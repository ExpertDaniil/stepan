from copy import deepcopy
import json
from pathlib import Path
import time
import uuid

from agent.config import ROOT
from agent.contracts import canonical
from agent.errors import AgentError
from agent.graph.artifacts import GraphArtifacts
from agent.graph.constructor import Constructor
from agent.graph.constraints import check as check_constraints
from agent.graph.context import WorkService
from agent.graph.export import GraphExport
from agent.graph.identity import dependency_closure, digest, solution
from agent.graph.recovery import GraphRecovery
from agent.observability.logging import CURRENT_CALL
from agent.search_policy import SearchPolicy, utility
from agent.store import comparable, is_better


def ident(prefix):
    return prefix + "-" + uuid.uuid4().hex[:16]


class GraphService(WorkService, SearchPolicy, GraphExport, GraphRecovery):
    """Local G/I facade on the SAME single-writer database as native C00-C24."""
    kind_schemas = {"graph_policy": "GPolicy", "graph_task_contract": "GTaskContract", "solution_manifest": "GSolutionManifest",
                    "team_checkpoint": "GCheckpoint", "graph_capsule": "GContextCapsule", "work_order": "GWorkOrder",
                    "snapshot_binding": "GSnapshotBinding", "proposal_record": "GProposalRecord", "component_idea": "GIdea",
                    "transfer_patch": "GTransferPatch", "component_trial": "GComponentTrial", "plateau_report": "GPlateau",
                    "search_decision": "GSearchDecision", "environment_manifest": "GEnvironmentManifest",
                    "component_catalog": "GCatalogManifest", "component_provenance": "GProvenance",
                    "graph_export": "GGraphExport", "ranking_report": "GRanking"}

    def __init__(self, controller, task_root):
        self.controller = controller
        for name in ("config", "bus", "budget", "artifacts", "store", "runner", "evaluator", "catalog", "spec", "validation", "log", "context"):
            setattr(self, name, getattr(controller, name))
        self.run_id, self.db, self.schemas = self.store.run_id, self.store.db, self.bus.schemas
        self.store._writer()
        exists = self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='g_schema_migrations'").fetchone()
        if not exists:
            try:
                self.db.executescript("BEGIN IMMEDIATE;\n" + Path(__file__).with_name("schema.sql").read_text(encoding="utf-8") + "\nCOMMIT;")
            except BaseException:
                self.db.rollback()
                raise
        self.files = GraphArtifacts(self, task_root)
        self.constructor = Constructor(self.catalog, self.artifacts, self.run_id)
        row = self.db.execute("SELECT policy_json FROM g_runs WHERE run_id=?", (self.run_id,)).fetchone()
        self.policy = json.loads(row[0]) if row else self.default_policy()
        self.budget.train_worker_limit = self.policy["max_train_workers"]
        if row:
            for name, key in (("environment", "environment_artifact_id"), ("catalog_manifest", "catalog_artifact_id"), ("provenance", "provenance_artifact_id")):
                art_id = self.metadata(key)
                meta = self.db.execute("SELECT * FROM g_artifacts WHERE run_id=? AND artifact_id=?", (self.run_id, art_id)).fetchone()
                if meta:
                    setattr(self.constructor, name, {k: meta[k] for k in ("artifact_id", "run_id", "kind", "relative_path", "sha256", "size_bytes", "media_type")})
        self.store.graph = self
        if row:
            self.apply_phase_caps()

    def default_policy(self):
        p = json.loads((ROOT / "recipes/graph-policy.json").read_text(encoding="utf-8"))
        p["constraint_mode"] = "catalog_strict" if self.config.research_mode == "catalog" else "generated_python"
        p['hard_decision_constraints'] = self.config.graph_hard_decisions
        p.update(peer_count=min(self.config.graph_peers, self.config.max_workers), max_train_workers=min(self.config.graph_train_workers, self.config.max_workers),
                 max_live_branches=self.config.graph_max_live_branches, max_total_branches=self.config.graph_max_total_branches)
        p["context"].update(max_input_tokens=min(self.config.max_context_tokens, 12000), max_output_tokens=self.config.llm_max_output_tokens,
                            max_api_request_bytes=self.config.graph_context_bytes)
        p["plateau"].update(window_full_evaluations=self.config.graph_plateau_window, min_distinct_solutions=self.config.graph_plateau_window,
                            min_elapsed_seconds=self.config.graph_plateau_seconds, absolute_epsilon=self.config.graph_plateau_absolute_epsilon,
                            relative_epsilon=self.config.graph_plateau_relative_epsilon)
        p["scheduling"]["exploit_steps"] = self.config.graph_exploit_steps
        p['scheduling']['min_branch_seconds'] = self.config.graph_branch_min_seconds
        p['scheduling'].update(refinement_enabled=self.config.graph_refinement_enabled, search_fraction=self.config.graph_search_fraction)
        p["promotion"]["min_effect"] = self.config.graph_min_effect
        p["retention"].update(max_artifact_bytes=self.config.graph_max_artifact_bytes, min_free_disk_bytes=self.config.graph_min_free_disk_bytes)
        return p

    def initialize_defaults(self):
        if self.revision():
            result = self.invoke("G00", {"task_spec_id": self.spec["task_spec_id"], "dataset_id": self.spec["dataset_id"],
                "validation_id": self.validation["validation_id"], "protocol_sha256": self.validation["protocol_sha256"],
                "environment_lock_artifact_id": self.metadata("environment_artifact_id"), "catalog_manifest_artifact_id": self.metadata("catalog_artifact_id"),
                "researcher_ids": self.metadata("researcher_ids"), "policy": self.policy}, key="graph-initialize")
            self.apply_phase_caps()
            return result
        environment, catalog = self.constructor.prepare()
        result = self.invoke("G00", {"task_spec_id": self.spec["task_spec_id"], "dataset_id": self.spec["dataset_id"],
            "validation_id": self.validation["validation_id"], "protocol_sha256": self.validation["protocol_sha256"],
            "environment_lock_artifact_id": environment["artifact_id"], "catalog_manifest_artifact_id": catalog["artifact_id"],
            "researcher_ids": ["researcher-"+str(i+1) for i in range(self.policy["peer_count"])], "policy": self.policy}, key="graph-initialize")

        self.apply_phase_caps()
        return result

    def elapsed(self):
        return max(0, int((time.monotonic()-self.budget.started)*1000))

    def revision(self):
        row = self.db.execute("SELECT graph_revision FROM g_runs WHERE run_id=?", (self.run_id,)).fetchone()
        return row[0] if row else 0

    def cut(self):
        return self.db.execute("SELECT COALESCE(MAX(event_seq),0) FROM g_events WHERE run_id=?", (self.run_id,)).fetchone()[0]

    def next_seq(self):
        return self.db.execute("SELECT COALESCE(MAX(event_seq),0)+1 FROM g_events").fetchone()[0]

    def metadata(self, name):
        row = self.db.execute("SELECT value FROM g_metadata WHERE run_id=? AND name=?", (self.run_id, name)).fetchone()
        if not row:
            raise AgentError("ENTITY_NOT_FOUND", "Graph metadata: "+name)
        return json.loads(row[0])

    def set_metadata_tx(self, name, value):
        self.db.execute("INSERT OR REPLACE INTO g_metadata VALUES(?,?,?)", (self.run_id, name, canonical(value)))

    def event(self, operation, entity_type, entity_id, record_schema, record):
        record = self.log.redactor.clean(record)
        self.schemas.validate(record, record_schema)
        seq = self.next_seq()
        value = {"event_schema": "graph-event-1.0", "run_id": self.run_id, "event_seq": seq, "event_id": ident("gevt"),
                 "elapsed_ms": self.elapsed(), "operation": operation, "request_id": CURRENT_CALL.get().get("request_id") or ident("internal"),
                 "entity_type": entity_type, "entity_id": entity_id, "record_schema": record_schema, "record": record}
        self.schemas.validate(value, "GEvent")
        self.db.execute("INSERT INTO g_events VALUES(?,?,?,?)", (seq, self.run_id, value["event_id"], canonical(value)))
        return seq

    def _cached(self, cid, payload, key):
        if key is None:
            return None
        row = self.db.execute("SELECT payload_sha256,response_json FROM g_idempotency WHERE run_id=? AND operation=? AND idempotency_key=?", (self.run_id, cid, key)).fetchone()
        if not row:
            return None
        if row[0] != digest(payload):
            raise AgentError("IDEMPOTENCY_CONFLICT", "Durable graph key reused with different payload")
        return json.loads(row[1])

    def commit(self, cid, payload, key, mutation):
        with self.store.transaction():
            cached = self._cached(cid, payload, key)
            if cached is not None:
                return cached
            data = mutation()
            self.db.execute("UPDATE g_runs SET graph_revision=graph_revision+1 WHERE run_id=?", (self.run_id,))
            if "graph_revision" in data:
                data["graph_revision"] = self.revision()
            self.schemas.validate({"schema_version": "1.0", "request_id": "validate-before-commit", "run_id": self.run_id,
                                   "ok": True, "data": data, "error": None}, cid+"_Response")
            if key is not None:
                self.db.execute("INSERT INTO g_idempotency VALUES(?,?,?,?,?)", (self.run_id, cid, key, digest(payload), canonical(data)))
        self.log.event("graph.committed", operation=cid, graph_revision=self.revision())
        return data

    def invoke(self, cid, payload, *, key=None):
        def dispatch(p):
            self.store._writer()
            cached = self._cached(cid, p, key)
            if cached is not None:
                return cached
            methods = {"G00": self.initialize, "G01": self.checkpoint, "G02": self.fork_decision, "G03": self.issue_work,
                       "G04": self.submit_proposal, "G05": self.commit_evaluation, "G06": self.register_idea,
                       "G07": self.assess_plateau, "G08": self.control_branch, "G09": self.files.register_staged,
                       "G10": lambda x, _: {"artifacts": [self.files.resolve(i, x["verify_bytes"]) for i in x["artifact_ids"]]},
                       "G11": self.export, "G12": self.recover, "G13": self.inspect, "G14": self.decide, "G15": self.start_refinement,
                       "I00": lambda x, _: self.constructor.verify(x), "I01": lambda x, _: check_constraints(x),
                       "I02": lambda x, _: self.resolve_lineage(x), "I03": lambda x, _: solution(x)}
            return methods[cid](p, key)
        return self.bus.call(cid, payload, dispatch, key=key)["data"]

    def branch(self, branch_id, expected=None):
        row = self.db.execute("SELECT * FROM g_branches WHERE run_id=? AND branch_id=?", (self.run_id, branch_id)).fetchone()
        if not row:
            raise AgentError("ENTITY_NOT_FOUND", "Branch: "+branch_id)
        if expected is not None and row["revision"] != expected:
            raise AgentError("BRANCH_REVISION_CONFLICT", "Branch policy epoch changed")
        constraints = {"required_capability_ids": [], "forbidden_capability_ids": []}
        for c in self.db.execute("SELECT capability_id,mode FROM g_constraints WHERE run_id=? AND branch_id=? ORDER BY capability_id", (self.run_id, branch_id)):
            constraints[("required" if c["mode"] == "required" else "forbidden")+"_capability_ids"].append(c["capability_id"])
        return {"branch_id": row["branch_id"], "parent_branch_id": row["parent_branch_id"], "fork_checkpoint_id": row["fork_checkpoint_id"],
                "decision_idea_id": row["decision_idea_id"], "decision": row["decision"], "state": row["state"],
                "branch_revision": row["revision"], "constraints": constraints, "created_event_seq": row["created_event_seq"]}

    def heads(self, branch_id):
        return [{"researcher_id": r["researcher_id"], "solution_id": r["solution_id"], "experiment_id": r["experiment_id"], "head_revision": r["revision"]}
                for r in self.db.execute("SELECT * FROM g_heads WHERE run_id=? AND branch_id=? ORDER BY researcher_id", (self.run_id, branch_id))]

    def head_satisfies(self, head, constraints):
        if not head["solution_id"]:
            return False
        manifest = self.manifest(head["solution_id"])
        if (constraints["required_capability_ids"] or constraints["forbidden_capability_ids"]) and manifest["verification_level"] != "catalog_verified":
            return False
        caps, _ = dependency_closure(manifest["core"]["components"])
        return set(constraints["required_capability_ids"]) <= set(caps) and not set(constraints["forbidden_capability_ids"]) & set(caps)

    def manifest(self, solution_id):
        row = self.db.execute("SELECT manifest_artifact_id FROM g_solutions WHERE run_id=? AND solution_id=?", (self.run_id, solution_id)).fetchone()
        if not row:
            raise AgentError("ENTITY_NOT_FOUND", "Solution: "+solution_id)
        self.files.resolve(row[0])
        return self.artifacts.read_json(row[0])

    def experiment_solution(self, experiment_id):
        self.store.get(experiment_id)
        row = self.db.execute("SELECT solution_id FROM g_experiment_links WHERE run_id=? AND experiment_id=?", (self.run_id, experiment_id)).fetchone()
        if not row:
            raise AgentError("ENTITY_NOT_FOUND", "Experiment has no graph solution")
        return row[0]

    def checkpoint_record(self, checkpoint_id):
        row = self.db.execute("SELECT snapshot_json FROM g_checkpoints WHERE run_id=? AND checkpoint_id=?", (self.run_id, checkpoint_id)).fetchone()
        if not row:
            raise AgentError("ENTITY_NOT_FOUND", "Checkpoint: "+checkpoint_id)
        return json.loads(row[0])

    def order(self, order_id):
        row = self.db.execute("SELECT order_json,state FROM g_work_orders WHERE run_id=? AND work_order_id=?", (self.run_id, order_id)).fetchone()
        if not row or not row[0]:
            raise AgentError("ENTITY_NOT_FOUND", "Work order: "+order_id)
        value = json.loads(row[0]);value["state"] = row[1]
        return value

    def initialize(self, p, key):
        if self.revision():
            raise AgentError("GRAPH_ALREADY_INITIALIZED", "Use the original durable key; never reset graph history")
        if (p["task_spec_id"], p["dataset_id"], p["validation_id"], p["protocol_sha256"]) != (self.spec["task_spec_id"], self.spec["dataset_id"], self.validation["validation_id"], self.validation["protocol_sha256"]):
            raise AgentError("STALE_VALIDATION", "Graph must use the trusted bootstrap protocol")
        peers = p["researcher_ids"]
        if len(set(peers)) != len(peers) or len(peers) != p["policy"]["peer_count"] or len(peers) > self.config.max_workers or p["policy"]["max_train_workers"] > self.config.max_workers:
            raise AgentError("INVALID_CONFIG", "Peer/execution limits disagree with C00")
        env = self.artifacts.read_json(p["environment_lock_artifact_id"])
        cat = self.artifacts.read_json(p["catalog_manifest_artifact_id"])
        self.schemas.validate(env, "GEnvironmentManifest");self.schemas.validate(cat, "GCatalogManifest")
        if set(cat["recipe_ids"]) != set(self.catalog.recipes):
            raise AgentError("MANIFEST_MISMATCH", "Catalogue disagrees with the current execution recipes")
        self.policy = deepcopy(p["policy"])
        policy_art = self.artifacts.write_json("graph/policy.json", self.policy, "graph_policy")
        task = {"task_spec": self.spec, "validation_id": p["validation_id"], "protocol_sha256": p["protocol_sha256"],
                "evaluator_implementation_id": "sklearn-metrics-v2", "selection_policy_id": "same-protocol-primary-v1",
                "train_rows_artifact_id": self.validation["train_rows_artifact_id"], "valid_rows_artifact_id": self.validation["valid_rows_artifact_id"],
                "labels_visibility": "evaluator_only"}
        self.schemas.validate(task, "GTaskContract")
        task_art = self.artifacts.write_json("graph/task_contract.json", task, "graph_task_contract")
        pins = [self.files.prepare_pin(i) for i in list(self.artifacts.entries)]
        root_id = ident("branch-root")
        def mutate():
            self.db.execute("INSERT INTO g_runs VALUES(?,?,?,?,?,?,?,?)", (self.run_id, p["task_spec_id"], p["dataset_id"], p["validation_id"], p["protocol_sha256"], canonical(self.policy), 0, 1))
            for prepared in pins:
                self.files.pin_tx(prepared)
            metadata = {"environment_artifact_id": p["environment_lock_artifact_id"], "catalog_artifact_id": p["catalog_manifest_artifact_id"],
                        "provenance_artifact_id": cat["provenance_artifact_ids"][0], "task_contract_artifact_id": task_art["artifact_id"],
                        "policy_artifact_id": policy_art["artifact_id"], "researcher_ids": peers, "root_branch_id": root_id,
                        "task_root": str(self.files.task_root)}
            if env.get("owned_runtime_artifact_id"):
                metadata["runtime_artifact_id"] = env["owned_runtime_artifact_id"]
            for name, value in metadata.items():
                self.set_metadata_tx(name, value)
            self.db.execute("INSERT INTO g_branches VALUES(?,?,?,?,?,?,?,?,?)", (self.run_id, root_id, None, None, None, "root", "active", 0, self.next_seq()))
            for who in peers:
                self.db.execute("INSERT INTO g_heads VALUES(?,?,?,?,?,?)", (self.run_id, root_id, who, None, None, 0))
            self.event("G00", "branch", root_id, "GBranch", self.branch(root_id))
            return {"root_branch": self.branch(root_id), "heads": self.heads(root_id), "policy_artifact_id": policy_art["artifact_id"], "graph_revision": 0}
        return self.commit("G00", p, key, mutate)

    def checkpoint(self, p, key):
        branch = self.branch(p["branch_id"], p["expected_branch_revision"])
        heads = self.heads(branch["branch_id"])
        for h in heads:
            if h["experiment_id"]:
                ev = self.store.experiment_evaluation(h["experiment_id"])
                if not ev or not ev["valid"] or self.store.get(h["experiment_id"])["state"] != "evaluated":
                    raise AgentError("INVALID_CHECKPOINT", "Every nonempty head must have a trusted evaluation")
        cp_id, art_id = ident("checkpoint"), "art-"+uuid.uuid4().hex
        cp = {"checkpoint_id": cp_id, "run_id": self.run_id, "branch_id": branch["branch_id"], "branch_revision": branch["branch_revision"],
              "store_revision": self.db.execute("SELECT revision FROM selections WHERE run_id=?", (self.run_id,)).fetchone()[0],
              "graph_revision": self.revision(), "captured_event_seq": self.cut(), "heads": heads, "constraints": branch["constraints"],
              "capsule_artifact_ids": [], "reason": p["reason"], "artifact_id": art_id}
        art = self.files.put_json(cp, "team_checkpoint", ident=art_id)
        def mutate():
            self.branch(branch["branch_id"], branch["branch_revision"])
            if self.heads(branch["branch_id"]) != heads:
                raise AgentError("HEAD_REVISION_CONFLICT", "Team changed while publishing checkpoint")
            self.db.execute("INSERT INTO g_checkpoints VALUES(?,?,?,?,?,?,?,?,?)", (self.run_id, cp_id, branch["branch_id"], cp["branch_revision"], cp["store_revision"], cp["graph_revision"], cp["captured_event_seq"], art_id, canonical(cp)))
            self.event("G01", "branch", branch["branch_id"], "GCheckpoint", cp)
            return {"checkpoint": cp, "artifact": art}
        return self.commit("G01", p, key, mutate)

    def _revoke_tx(self, branch_id):
        revoked = [r[0] for r in self.db.execute("SELECT work_order_id FROM g_work_orders WHERE run_id=? AND branch_id=? AND state IN ('preparing','issued')", (self.run_id, branch_id))]
        self.db.execute("UPDATE g_work_orders SET state='revoked' WHERE run_id=? AND branch_id=? AND state IN ('preparing','issued')", (self.run_id, branch_id))
        self.db.execute("UPDATE g_bindings SET invalidated=1 WHERE run_id=? AND work_order_id IN (SELECT work_order_id FROM g_work_orders WHERE run_id=? AND branch_id=?) AND state='pending'", (self.run_id, self.run_id, branch_id))
        jobs = []
        for job in self.runner.active():
            if job.phase == "train" and self.db.execute("SELECT 1 FROM g_bindings b JOIN g_work_orders w USING(run_id,work_order_id) WHERE b.run_id=? AND b.experiment_id=? AND w.branch_id=?", (self.run_id, job.experiment_id, branch_id)).fetchone():
                jobs.append(job.job_id)
        return revoked, jobs

    def _check_preidea(self, cp, capabilities):
        for h in cp["heads"]:
            if h["solution_id"]:
                caps, _ = dependency_closure(self.manifest(h["solution_id"])["core"]["components"])
                if set(caps) & set(capabilities):
                    raise AgentError("INVALID_FORK_CHECKPOINT", "Checkpoint heads already contain the new idea")

    def fork_decision(self, p, key):
        if self.refinement():
            raise AgentError('FINAL_REFINEMENT_ONLY', 'Final refinement does not create decision branches')
        parent = self.branch(p["parent_branch_id"], p["expected_branch_revision"])
        cp = self.checkpoint_record(p["checkpoint_id"])
        row = self.db.execute("SELECT record_json FROM g_ideas WHERE run_id=? AND idea_id=?", (self.run_id, p["idea_id"])).fetchone()
        if not row:
            raise AgentError("ENTITY_NOT_FOUND", "Idea: "+p["idea_id"])
        idea = json.loads(row[0])
        if idea["status"] != "confirmed" or idea["fork_checkpoint_id"] != cp["checkpoint_id"] or cp["branch_id"] != parent["branch_id"] or cp["branch_revision"] != parent["branch_revision"] or cp["constraints"] != parent["constraints"]:
            raise AgentError("INVALID_FORK_CHECKPOINT", "Fork needs a confirmed idea and its origin checkpoint")
        self._check_preidea(cp, idea["capability_ids"])
        required = set(parent["constraints"]["required_capability_ids"])
        forbidden = set(parent["constraints"]["forbidden_capability_ids"])
        if set(idea["capability_ids"]) & (required | forbidden):
            raise AgentError("CONSTRAINT_CONFLICT", "Fork must introduce a new unconstrained capability")
        children = {kind: ident("branch-"+kind) for kind in ("accept", "reject")}
        hard_decision = self.policy.get('hard_decision_constraints', False)
        archived = []
        # The causal checkpoint remains immutable. Explicit seeds carry this peer's
        # later measured knowledge into a child, rather than throwing warmups away.
        selection_cut = self.cut()
        seeds = {}
        for kind in children:
            constraints = {"required_capability_ids": sorted(required | (set(idea["capability_ids"]) if hard_decision and kind == "accept" else set())),
                           "forbidden_capability_ids": sorted(forbidden | (set(idea["capability_ids"]) if hard_decision and kind == "reject" else set()))}
            seeds[kind] = []
            for head in cp["heads"]:
                ids = list(dict.fromkeys(([head["experiment_id"]] if head["experiment_id"] else []) + self.visible_history(parent["branch_id"], head["researcher_id"])))
                choices = []
                expected = {"dataset_id": self.spec["dataset_id"], "validation_id": self.validation["validation_id"],
                            "metric_id": self.spec["metric"]["metric_id"], "direction": self.spec["metric"]["direction"], "fidelity": "full"}
                for eid in ids:
                    evaluation = self.store.experiment_evaluation(eid)
                    if (self.store.get(eid)["state"] == "evaluated" and evaluation and evaluation["valid"]
                            and not evaluation["violations"] and comparable(evaluation["primary_metric"], expected)):
                        choices.append({**head, "experiment_id": eid, "solution_id": self.experiment_solution(eid)})
                eligible = [h for h in choices if self.head_satisfies(h, constraints)]
                reason = "best_compatible_own_reference"
                if not hard_decision:
                    preferred = [h for h in eligible if (set(idea['capability_ids']) <= set(dependency_closure(
                        self.manifest(h['solution_id'])['core']['components'])[0])) == (kind == 'accept')]
                    if preferred:
                        eligible = preferred
                        reason = 'best_own_reference_matching_soft_preference'
                if not eligible and kind == "accept":
                    # A new feature may not have been tested on this peer yet.
                    eligible = [h for h in choices if self.head_satisfies(h, parent["constraints"])
                                and not set(dependency_closure(self.manifest(h["solution_id"])["core"]["components"])[0]) & set(idea["capability_ids"])]
                    reason = "pre_feature_own_reference"
                chosen = max(eligible, key=lambda h: utility(self.store.experiment_evaluation(h["experiment_id"]))) if eligible else head
                chosen = {**chosen, "head_revision": head["head_revision"] + int(chosen["experiment_id"] != head["experiment_id"])}
                seeds[kind].append({"branch_id": children[kind], "parent_branch_id": parent["branch_id"], "fork_checkpoint_id": cp["checkpoint_id"],
                                    "selection_cut": selection_cut, "head": chosen, "selection_reason": reason if eligible else "checkpoint_reference"})
        def mutate():
            current = self.branch(parent["branch_id"], parent["branch_revision"])
            if current["state"] != "active":
                raise AgentError("INVALID_BRANCH_STATE", "Only an active branch can fork")
            total = self.db.execute("SELECT count(*) FROM g_branches WHERE run_id=?", (self.run_id,)).fetchone()[0]
            victims = self.fork_victims(parent['branch_id'])
            if victims is None or total+2 > self.policy["max_total_branches"]:
                raise AgentError("BRANCH_LIMIT", "Both children must fit; a partial fork is forbidden")
            for bid in victims:
                self.db.execute("UPDATE g_branches SET state='archived',revision=revision+1 WHERE run_id=? AND branch_id=?", (self.run_id, bid))
                self._revoke_tx(bid)
                self.event('G02', 'branch', bid, 'GBranch', self.branch(bid))
                archived.append(bid)
            self.db.execute("UPDATE g_branches SET state='sealed',revision=revision+1 WHERE run_id=? AND branch_id=?", (self.run_id, parent["branch_id"]))
            revoked, jobs = self._revoke_tx(parent["branch_id"])
            for kind, child in children.items():
                self.db.execute("INSERT INTO g_branches VALUES(?,?,?,?,?,?,?,?,?)", (self.run_id, child, parent["branch_id"], cp["checkpoint_id"], idea["idea_id"], kind, "active" if kind == "accept" else "paused", 0, self.next_seq()))
                self.db.execute("INSERT INTO g_constraints SELECT run_id,?,capability_id,mode,origin_branch_id FROM g_constraints WHERE run_id=? AND branch_id=?", (child, self.run_id, parent["branch_id"]))
                for cap in idea["capability_ids"] if hard_decision else []:
                    self.db.execute("INSERT INTO g_constraints VALUES(?,?,?,?,?)", (self.run_id, child, cap, "required" if kind == "accept" else "forbidden", child))
                for seed in seeds[kind]:
                    h = seed["head"]
                    self.db.execute("INSERT INTO g_heads VALUES(?,?,?,?,?,?)", (self.run_id, child, h["researcher_id"], h["solution_id"], h["experiment_id"], h["head_revision"]))
                self.event("G02", "branch", child, "GBranch", self.branch(child))
                for seed in seeds[kind]:
                    self.event("G02", "branch", child, "GBranchSeed", seed)
            self.event("G02", "branch", parent["branch_id"], "GBranch", self.branch(parent["branch_id"]))
            return {"parent_branch": self.branch(parent["branch_id"]), "accept_branch": self.branch(children["accept"]), "reject_branch": self.branch(children["reject"]),
                    "revoked_work_order_ids": revoked, "origin_running_job_ids": jobs, "graph_revision": 0}
        result = self.commit("G02", p, key, mutate)
        for bid in archived:
            self.log.event('graph.branch_state_changed', branch_id=bid, action='archive', reason='Make room for a complete pair of decision children', state='archived',
                           eligibility='legacy_strict_policy' if hard_decision else 'current_plateau_or_no_executable_work')
        self.log.event("graph.branch_forked", parent_branch_id=parent["branch_id"], idea_id=idea["idea_id"],
                       checkpoint_id=cp["checkpoint_id"], accept_branch_id=result["accept_branch"]["branch_id"],
                       reject_branch_id=result["reject_branch"]["branch_id"], capability_ids=idea["capability_ids"],
                       decision_constraints='hard' if hard_decision else 'soft')
        for kind, branch_key in (("accept", "accept_branch"), ("reject", "reject_branch")):
            self.log.event("graph.branch_seeded", branch_id=result[branch_key]["branch_id"],
                           heads=self.heads(result[branch_key]["branch_id"]), selection_cut=selection_cut)
        return result

    def control_branch(self, p, key):
        def mutate():
            branch = self.branch(p["branch_id"], p["expected_branch_revision"])
            allowed = {"pause": {"active"}, "resume": {"paused"}, "archive": {"active", "paused"}}
            target = {"pause": "paused", "resume": "active", "archive": "archived"}[p["action"]]
            if branch["state"] not in allowed[p["action"]]:
                raise AgentError("INVALID_BRANCH_STATE", "Invalid branch control transition")
            self.db.execute("UPDATE g_branches SET state=?,revision=revision+1 WHERE run_id=? AND branch_id=?", (target, self.run_id, branch["branch_id"]))
            revoked, jobs = self._revoke_tx(branch["branch_id"])
            updated = self.branch(branch["branch_id"])
            self.event("G08", "branch", branch["branch_id"], "GBranch", updated)
            return {"branch": updated, "revoked_work_order_ids": revoked, "origin_running_job_ids": jobs, "graph_revision": 0}
        result = self.commit("G08", p, key, mutate)
        self.log.event("graph.branch_state_changed", branch_id=p["branch_id"], action=p["action"], reason=p["reason"],
                       state=result["branch"]["state"], branch_revision=result["branch"]["branch_revision"])
        return result

    def check_order(self, order_id):
        order = self.order(order_id)
        if order["state"] != "issued":
            raise AgentError("WORK_ORDER_NOT_ISSUED", "Work order is no longer issued")
        if self.elapsed() >= order["expires_elapsed_ms"]:
            with self.store.transaction():
                self.db.execute("UPDATE g_work_orders SET state='expired' WHERE run_id=? AND work_order_id=? AND state='issued'", (self.run_id, order_id))
            raise AgentError("WORK_ORDER_EXPIRED", "The proposal arrived after its assignment TTL")
        branch = self.branch(order["branch_id"], order["branch_revision"])
        if branch["state"] != "active":
            raise AgentError("INVALID_BRANCH_STATE", "The proposal's origin branch is not active")
        head = next((h for h in self.heads(order["branch_id"]) if h["researcher_id"] == order["researcher_id"]), None)
        if not head or head["head_revision"] != order["base_head_revision"] or head["experiment_id"] != order["base_experiment_id"] or head["solution_id"] != order["base_solution_id"]:
            raise AgentError("HEAD_REVISION_CONFLICT", "The proposal no longer matches its own peer head")
        if not self.db.execute("SELECT search_allowed FROM g_runs WHERE run_id=?", (self.run_id,)).fetchone()[0] or not self.budget.snapshot()["search_allowed"]:
            raise AgentError("SEARCH_CLOSED", "No new experiments may be submitted")
        return order

    def resolve_lineage(self, p):
        work, lineage, child = p["work_order"], p["lineage"], p["candidate_solution_id"]
        if work["run_id"] != self.run_id or self.order(work["work_order_id"]) != work:
            raise AgentError("STALE_WORK_ORDER", "Lineage must reference the trusted current work order")
        if lineage["base_experiment_id"] != work["base_experiment_id"]:
            raise AgentError("LINEAGE_MISMATCH", "Primary parent must be this peer's assigned base")
        donors = lineage["idea_donor_experiment_ids"] + lineage["ensemble_member_experiment_ids"]
        if not set(donors) <= set(work["allowed_donor_experiment_ids"]):
            raise AgentError("DONOR_NOT_ALLOWED", "Donor was not assigned to this work order")
        if lineage["ensemble_member_experiment_ids"]:
            raise AgentError("UNSUPPORTED_ACTION", "Ensemble compiler is not implemented in graph v1")
        for idea_id in lineage["source_idea_ids"]:
            row = self.db.execute("SELECT record_json FROM g_ideas WHERE run_id=? AND idea_id=?", (self.run_id, idea_id)).fetchone()
            if not row or json.loads(row[0])["source_experiment_id"] not in donors:
                raise AgentError("LINEAGE_MISMATCH", "Source idea must match an explicit donor")
        primary = [work["base_experiment_id"]] if work["base_experiment_id"] else []
        parents = list(dict.fromkeys(primary + donors))
        if parents != p["plan_parent_ids"]:
            raise AgentError("LINEAGE_MISMATCH", "Untyped C12 parent_ids must equal the canonical typed parents")
        existing = self.db.execute("SELECT 1 FROM g_solutions WHERE run_id=? AND solution_id=?", (self.run_id, child)).fetchone()
        edges = []
        for kind, ids in (("primary", primary), ("idea_donor", lineage["idea_donor_experiment_ids"])):
            for eid in ids:
                parent = self.experiment_solution(eid)
                if not existing and parent != child:
                    edge = {"parent_solution_id": parent, "child_solution_id": child, "kind": kind}
                    if edge not in edges:
                        edges.append(edge)
        return {"canonical_edges": edges, "alternate_provenance_only": bool(existing)}

    def submit_proposal(self, p, key):
        order = self.check_order(p["work_order_id"])
        proposal = deepcopy(p["proposal"])
        if proposal["researcher_id"] != order["researcher_id"]:
            raise AgentError("WORK_ORDER_MISMATCH", "Future ticket and proposal researcher disagree")
        if proposal["decision"] == "stop":
            if proposal["plan"] is not None or p["manifest"] is not None or p["lineage"] is not None:
                raise AgentError("INVALID_PROPOSAL", "A stop proposal has no plan/manifest/lineage")
            record = {"work_order_id": order["work_order_id"], "proposal": proposal, "manifest_artifact_id": None,
                      "lineage": None, "accepted": True, "rejection_code": None}
            self.files.put_json(record, "proposal_record")
            def stopped():
                self.check_order(order["work_order_id"])
                self.db.execute("UPDATE g_work_orders SET state='stopped' WHERE run_id=? AND work_order_id=?", (self.run_id, order["work_order_id"]))
                self.db.execute("INSERT INTO g_proposals VALUES(?,?,?,?,?)", (self.run_id, proposal["proposal_id"], order["work_order_id"], canonical(record), 1))
                self.event("G04", "work_order", order["work_order_id"], "GProposalRecord", record)
                return {"work_order_id": order["work_order_id"], "disposition": "stopped", "solution_id": None, "solution_disposition": None,
                        "lead_submission": None, "reused_evaluation_id": None, "graph_revision": 0}
            return self.commit("G04", p, key, stopped)
        if p["manifest"] is None or p["lineage"] is None or proposal["plan"] is None:
            raise AgentError("INVALID_PROPOSAL", "A proposed plan requires a manifest and typed lineage")
        plan = self.catalog.validate_plan(proposal["plan"], self.spec, self.validation["validation_id"], self.artifacts)
        if self.config.research_mode != 'adaptive' and self.refinement() and order['base_experiment_id']:
            base = self.store.get(order['base_experiment_id'])['plan']
            if base['model']['family'] != 'dummy' and plan['recipe_id'] != base['recipe_id']:
                raise AgentError('REFINEMENT_FAMILY_CHANGED', 'Final refinement keeps the measured model family')
            if base['model']['family'] != 'dummy' and any(plan['model']['hyperparameters'].get(feature['parameter']) != base['model']['hyperparameters'].get(feature['parameter']) for feature in self.constructor.registry['features'].values()):
                raise AgentError('REFINEMENT_COMPONENT_CHANGED', 'Final refinement preserves the measured feature switches')
        if plan["researcher_id"] != order["researcher_id"] or plan["fidelity"] != "full":
            raise AgentError("INVALID_PLAN", "Graph v1 executes full fidelity plans owned by the assigned peer")
        code_pin = self.files.prepare_pin(plan["code_artifact_id"])
        verified = self.invoke("I00", {"plan": plan, "manifest": p["manifest"], "code_artifact": code_pin[0],
                                      "catalog_manifest_artifact_id": self.metadata("catalog_artifact_id")})
        manifest = verified["manifest"]
        if (self.policy["constraint_mode"] == "catalog_strict" or order["constraints"]["required_capability_ids"] or order["constraints"]["forbidden_capability_ids"]) and manifest["verification_level"] != "catalog_verified":
            raise AgentError("UNVERIFIABLE_CONSTRAINT", "Hard restrictions require the exact owned constructor")
        self.invoke("I01", {"constraints": order["constraints"], "effective_capability_ids": verified["effective_capability_ids"]})
        lineage = self.invoke("I02", {"work_order": order, "lineage": p["lineage"], "plan_parent_ids": plan["parent_ids"], "candidate_solution_id": manifest["solution_id"]})
        self.invoke("I03", {"core": manifest["core"]})
        manifest_art = self.files.put_json(manifest, "solution_manifest")
        prepared = {"work_order_id": order["work_order_id"], "proposal": proposal,
                    "manifest_artifact_id": manifest_art["artifact_id"], "lineage": p["lineage"], "accepted": True, "rejection_code": None}
        self.schemas.validate(prepared, "GProposalRecord")
        self.files.put_json(prepared, "proposal_record")
        # The stored proposal is the exact submitted record, including alternate provenance on a cache hit.
        def mutate():
            self.check_order(order["work_order_id"])
            self.files.pin_tx(code_pin)
            old = self.db.execute("SELECT solution_hash FROM g_solutions WHERE run_id=? AND solution_id=?", (self.run_id, manifest["solution_id"])).fetchone()
            if old and old[0] != manifest["solution_hash"]:
                raise AgentError("SOLUTION_ID_CONFLICT", "Solution ID already identifies another core")
            created = not old
            if created:
                self.db.execute("INSERT INTO g_solutions VALUES(?,?,?,?,?,?)", (self.run_id, manifest["solution_id"], manifest["solution_hash"], manifest_art["artifact_id"], canonical(manifest["core"]), self.next_seq()))
                for edge in lineage["canonical_edges"]:
                    self.db.execute("INSERT INTO g_solution_edges VALUES(?,?,?,?)", (self.run_id, edge["parent_solution_id"], edge["child_solution_id"], edge["kind"]))
            for c in manifest["core"]["components"]:
                self.db.execute("INSERT OR IGNORE INTO g_components VALUES(?,?,?)", (self.run_id, c["component_id"], canonical(c)))
            submission = self.bus.call("C12", {"plan": plan}, lambda args: self.store.reserve_tx(args["plan"], code_pin[0]["sha256"]))["data"]
            existing_link = self.db.execute("SELECT solution_id FROM g_experiment_links WHERE run_id=? AND experiment_id=?", (self.run_id, submission["experiment_id"])).fetchone()
            if existing_link and existing_link[0] != manifest["solution_id"]:
                raise AgentError("MANIFEST_MISMATCH", "Native fingerprint already points to a different solution")
            self.db.execute("INSERT OR IGNORE INTO g_experiment_links VALUES(?,?,?)", (self.run_id, submission["experiment_id"], manifest["solution_id"]))
            self.db.execute("INSERT INTO g_proposals VALUES(?,?,?,?,?)", (self.run_id, proposal["proposal_id"], order["work_order_id"], canonical(prepared), 1))
            self.db.execute("INSERT INTO g_bindings VALUES(?,?,?,?,?,?,?)", (self.run_id, ident("binding"), order["work_order_id"], submission["experiment_id"], "pending", 0, None))
            self.db.execute("UPDATE g_work_orders SET state='submitted' WHERE run_id=? AND work_order_id=?", (self.run_id, order["work_order_id"]))
            # Use an existing closed schema for this event; the full proposal remains in g_proposals.
            updated_order = {**order, "state": "submitted"}
            self.event("G04", "work_order", order["work_order_id"], "GWorkOrder", updated_order)
            ev = self.store.experiment_evaluation(submission["experiment_id"])
            if submission["state"] in {"failed", "timed_out", "cancelled", "invalid"}:
                self.terminal_tx(submission["experiment_id"], submission["state"], "Cached terminal experiment", "C15")
            return {"work_order_id": order["work_order_id"], "disposition": "duplicate" if submission["disposition"] == "duplicate" else "submitted",
                    "solution_id": manifest["solution_id"], "solution_disposition": "created" if created else "existing", "lead_submission": submission,
                    "reused_evaluation_id": ev["evaluation_id"] if ev and submission["state"] == "evaluated" else None, "graph_revision": 0}
        result = self.commit("G04", p, key, mutate)
        if result["lead_submission"] and result["lead_submission"]["disposition"] == "created":
            self.log.event("experiment.reserved", experiment_id=result["lead_submission"]["experiment_id"], fingerprint=result["lead_submission"]["fingerprint"], researcher_id=order["researcher_id"])
        return result

    def commit_evaluation(self, p, key):
        ev = self.store.evaluation(p["evaluation_id"])
        exp = self.store.get(p["experiment_id"])
        metric = ev["primary_metric"]
        if not ev["valid"] or ev["experiment_id"] != exp["experiment_id"] or metric["validation_id"] != self.validation["validation_id"] or metric["dataset_id"] != self.spec["dataset_id"] or metric["metric_id"] != self.spec["metric"]["metric_id"] or metric["direction"] != self.spec["metric"]["direction"] or metric["fidelity"] != exp["plan"]["fidelity"]:
            raise AgentError("INVALID_EVALUATION", "Evaluation scope does not match the trusted task and experiment")
        solution_id = self.experiment_solution(exp["experiment_id"])
        diag_pin = self.files.prepare_pin(ev["diagnostics_artifact_id"])
        def mutate():
            self.files.pin_tx(diag_pin)
            lead = self.bus.call("C17", p, self.store.commit_evaluation_tx)["data"]
            results = []
            bindings = self.db.execute("SELECT * FROM g_bindings WHERE run_id=? AND experiment_id=? AND state='pending' ORDER BY rowid", (self.run_id, exp["experiment_id"])).fetchall()
            for binding in bindings:
                order = self.order(binding["work_order_id"])
                branch = self.branch(order["branch_id"])
                head = next(h for h in self.heads(order["branch_id"]) if h["researcher_id"] == order["researcher_id"])
                if branch["state"] != "active" or binding["invalidated"]:
                    disposition = "archived_branch_closed"
                elif branch["branch_revision"] != order["branch_revision"] or head["head_revision"] != order["base_head_revision"]:
                    disposition = "archived_stale"
                elif metric["fidelity"] != "full":
                    disposition = "kept_screening"
                elif not self.head_satisfies(head, branch["constraints"]) or is_better(metric, self.store.experiment_evaluation(head["experiment_id"])["primary_metric"]):
                    changed = self.db.execute("UPDATE g_heads SET solution_id=?,experiment_id=?,revision=revision+1 WHERE run_id=? AND branch_id=? AND researcher_id=? AND revision=?",
                        (solution_id, exp["experiment_id"], self.run_id, order["branch_id"], order["researcher_id"], order["base_head_revision"])).rowcount
                    if changed != 1:
                        raise AgentError("HEAD_REVISION_CONFLICT", "Peer head CAS failed")
                    disposition = "advanced"
                    head["head_revision"] += 1
                else:
                    disposition = "kept_not_better"
                result = {"work_order_id": order["work_order_id"], "branch_id": order["branch_id"], "researcher_id": order["researcher_id"],
                          "disposition": disposition, "solution_id": solution_id, "experiment_id": exp["experiment_id"], "head_revision": head["head_revision"]}
                seq = self.event("G05", "evaluation", ev["evaluation_id"], "GHeadResult", result)
                self.db.execute("INSERT OR IGNORE INTO g_branch_evaluations VALUES(?,?,?,?,?,?,?)", (self.run_id, order["branch_id"], exp["experiment_id"], ev["evaluation_id"], solution_id, seq, self.elapsed()))
                self.db.execute("UPDATE g_bindings SET state=?,outcome=? WHERE run_id=? AND binding_id=?", ("archived" if disposition.startswith("archived") else "applied", disposition, self.run_id, binding["binding_id"]))
                self.db.execute("UPDATE g_work_orders SET state='completed' WHERE run_id=? AND work_order_id=?", (self.run_id, order["work_order_id"]))
                results.append(result)
            if lead["should_prepare_submission"]:
                payload = {"experiment_id": exp["experiment_id"], "evaluation_id": ev["evaluation_id"], "task_spec_id": self.spec["task_spec_id"],
                           "mode": "refit_full_train", "expected_champion_version": self.store.champion()["champion_version"] if self.store.champion() else 0}
                self.db.execute("INSERT OR IGNORE INTO g_outbox VALUES(?,?,?,?,?,?)", (ident("outbox"), self.run_id, "prepare_submission", exp["experiment_id"]+"/"+ev["evaluation_id"], canonical(payload), "pending"))
            return {"lead_commit": lead, "head_results": results, "graph_revision": 0}
        return self.commit("G05", p, key, mutate)

    def capture_event_tx(self, event):
        row = self.db.execute("SELECT 1 FROM g_experiment_links WHERE run_id=? AND experiment_id=?", (self.run_id, event["experiment_id"])).fetchone()
        if not row:
            return
        attempt = {k: event[k] for k in ("attempt_no", "job_id", "lease_id", "fencing_token", "state", "output_artifact_ids", "duration_seconds", "peak_memory_mb", "error")}
        self.schemas.validate(attempt, "AAttemptRecord")
        self.db.execute("INSERT OR REPLACE INTO g_attempts VALUES(?,?,?,?)", (self.run_id, event["experiment_id"], event["attempt_no"], canonical(attempt)))
        self.event("C15", "evaluation", event["experiment_id"], "WorkerEvent", event)
        if event["state"] != "succeeded":
            self.terminal_tx(event["experiment_id"], event["state"], event["error"] or "Execution failed", "C15")

    def terminal_tx(self, experiment_id, state, error, operation):
        bindings = self.db.execute("SELECT binding_id,work_order_id FROM g_bindings WHERE run_id=? AND experiment_id=? AND state='pending'", (self.run_id, experiment_id)).fetchall()
        for b in bindings:
            self.db.execute("UPDATE g_bindings SET state='archived',outcome=? WHERE run_id=? AND binding_id=?", ("execution_"+state, self.run_id, b["binding_id"]))
            self.db.execute("UPDATE g_work_orders SET state='completed' WHERE run_id=? AND work_order_id=?", (self.run_id, b["work_order_id"]))
            self.event(operation, "work_order", b["work_order_id"], "Error", AgentError("EXPERIMENT_"+state.upper(), str(error)).as_dict())
        if bindings:
            self.db.execute("UPDATE g_runs SET graph_revision=graph_revision+1 WHERE run_id=?", (self.run_id,))

    def close_search(self, reason):
        with self.store.transaction():
            self.db.execute("UPDATE g_runs SET search_allowed=0,graph_revision=graph_revision+1 WHERE run_id=?", (self.run_id,))
            self.db.execute("UPDATE g_work_orders SET state='revoked' WHERE run_id=? AND state IN ('preparing','issued')", (self.run_id,))
            for row in self.db.execute("SELECT experiment_id FROM experiments WHERE run_id=? AND state='queued'", (self.run_id,)).fetchall():
                self.db.execute("UPDATE experiments SET state='cancelled',error=? WHERE experiment_id=?", (reason, row[0]))
                self.terminal_tx(row[0], "cancelled", reason, "C22")
            unfinished = self.db.execute("SELECT b.binding_id,b.work_order_id FROM g_bindings b JOIN experiments e USING(run_id,experiment_id) WHERE b.run_id=? AND b.state='pending' AND e.state<>'running'", (self.run_id,)).fetchall()
            for binding in unfinished:
                self.db.execute("UPDATE g_bindings SET state='archived',outcome='archived_search_closed' WHERE run_id=? AND binding_id=?", (self.run_id, binding["binding_id"]))
                self.db.execute("UPDATE g_work_orders SET state='completed' WHERE run_id=? AND work_order_id=?", (self.run_id, binding["work_order_id"]))

    def pin_event_outputs(self, event):
        for art_id in event["output_artifact_ids"]:
            self.files.pin(art_id)

    def acknowledge_publications(self):
        champion = self.store.champion()
        if not champion:
            return
        with self.store.transaction():
            rows = self.db.execute("SELECT outbox_id,payload_json FROM g_outbox WHERE run_id=? AND kind='prepare_submission' AND state='pending'", (self.run_id,)).fetchall()
            for row in rows:
                payload = json.loads(row[1])
                ev = self.store.evaluation(payload["evaluation_id"])
                if comparable(ev["primary_metric"], champion["primary_metric"]) and not is_better(ev["primary_metric"], champion["primary_metric"]):
                    self.db.execute("UPDATE g_outbox SET state=? WHERE outbox_id=?", ("delivered" if champion["experiment_id"] == payload["experiment_id"] else "superseded", row[0]))

    def track_job(self, job):
        attempt = {"attempt_no": job.attempt_no, "job_id": job.job_id, "lease_id": job.lease["lease_id"], "fencing_token": job.lease["fencing_token"],
                   "state": "running", "output_artifact_ids": [], "duration_seconds": None, "peak_memory_mb": None, "error": None}
        self.schemas.validate(attempt, "AAttemptRecord")
        with self.store.transaction():
            self.db.execute("INSERT INTO g_attempts VALUES(?,?,?,?)", (self.run_id, job.experiment_id, job.attempt_no, canonical(attempt)))

    def abandon(self, order_id, reason):
        with self.store.transaction():
            changed = self.db.execute("UPDATE g_work_orders SET state='revoked' WHERE run_id=? AND work_order_id=? AND state IN ('preparing','issued')", (self.run_id, order_id)).rowcount
            if changed:
                self.db.execute("UPDATE g_runs SET graph_revision=graph_revision+1 WHERE run_id=?", (self.run_id,))
                self.event("G03", "work_order", order_id, "Error", AgentError("PROPOSAL_REJECTED", reason).as_dict())
