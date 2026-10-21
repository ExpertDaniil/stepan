-- Graph additive migration for existing agent/schema.sql at e9a9d04.
-- Apply to the SAME SQLite database/connection used by C12/C15/C17/C21.
-- Native run/experiment references use existing tables; other cross-entity checks belong to the adapter.
PRAGMA foreign_keys=ON;
CREATE UNIQUE INDEX IF NOT EXISTS graph_native_exp_scope ON experiments(run_id,experiment_id);
CREATE TABLE IF NOT EXISTS g_schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL) STRICT;
CREATE TABLE g_runs(
 run_id TEXT PRIMARY KEY, task_spec_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
 validation_id TEXT NOT NULL, protocol_sha256 TEXT NOT NULL CHECK(length(protocol_sha256)=64),
 policy_json TEXT NOT NULL CHECK(json_valid(policy_json)), graph_revision INTEGER NOT NULL DEFAULT 0,
 search_allowed INTEGER NOT NULL CHECK(search_allowed IN(0,1)), CHECK(graph_revision>=0), FOREIGN KEY(run_id) REFERENCES runs(run_id)
) STRICT;
CREATE TABLE g_artifacts(
 run_id TEXT NOT NULL, artifact_id TEXT NOT NULL, kind TEXT NOT NULL,
 relative_path TEXT NOT NULL, sha256 TEXT NOT NULL CHECK(length(sha256)=64), size_bytes INTEGER NOT NULL CHECK(size_bytes>=0),
 media_type TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN('staged','ready','corrupt','missing')),
 PRIMARY KEY(run_id,artifact_id), FOREIGN KEY(run_id) REFERENCES g_runs(run_id)
) STRICT;
CREATE TABLE g_artifact_intents(
 run_id TEXT NOT NULL, intent_id TEXT NOT NULL, request_hash TEXT NOT NULL,
 intent_json TEXT NOT NULL CHECK(json_valid(intent_json)),
 state TEXT NOT NULL CHECK(state IN('prepared','file_published','ready','failed')),
 PRIMARY KEY(run_id,intent_id), FOREIGN KEY(run_id) REFERENCES g_runs(run_id)
) STRICT;
CREATE TABLE g_branches(
 run_id TEXT NOT NULL, branch_id TEXT NOT NULL, parent_branch_id TEXT,
 fork_checkpoint_id TEXT, decision_idea_id TEXT,
 decision TEXT NOT NULL CHECK(decision IN('root','accept','reject')),
 state TEXT NOT NULL CHECK(state IN('active','paused','sealed','archived')),
 revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0), created_event_seq INTEGER NOT NULL CHECK(created_event_seq>=1),
 PRIMARY KEY(run_id,branch_id), FOREIGN KEY(run_id) REFERENCES g_runs(run_id),
 FOREIGN KEY(run_id,parent_branch_id) REFERENCES g_branches(run_id,branch_id),
 FOREIGN KEY(run_id,fork_checkpoint_id) REFERENCES g_checkpoints(run_id,checkpoint_id),
 FOREIGN KEY(run_id,decision_idea_id) REFERENCES g_ideas(run_id,idea_id),
 UNIQUE(run_id,parent_branch_id,decision),
 CHECK((decision='root' AND parent_branch_id IS NULL AND fork_checkpoint_id IS NULL AND decision_idea_id IS NULL)
 OR (decision<>'root' AND parent_branch_id IS NOT NULL AND fork_checkpoint_id IS NOT NULL AND decision_idea_id IS NOT NULL)),
 CHECK(branch_id IS NOT parent_branch_id)
) STRICT;
CREATE UNIQUE INDEX g_one_root ON g_branches(run_id) WHERE decision='root';
CREATE TABLE g_constraints(
 run_id TEXT NOT NULL, branch_id TEXT NOT NULL, capability_id TEXT NOT NULL,
 mode TEXT NOT NULL CHECK(mode IN('required','forbidden')), origin_branch_id TEXT NOT NULL,
 PRIMARY KEY(run_id,branch_id,capability_id),
 FOREIGN KEY(run_id,branch_id) REFERENCES g_branches(run_id,branch_id),
 FOREIGN KEY(run_id,origin_branch_id) REFERENCES g_branches(run_id,branch_id)
) STRICT;
CREATE TABLE g_solutions(
 run_id TEXT NOT NULL, solution_id TEXT NOT NULL, solution_hash TEXT NOT NULL CHECK(length(solution_hash)=64),
 manifest_artifact_id TEXT NOT NULL, core_json TEXT NOT NULL CHECK(json_valid(core_json)), created_event_seq INTEGER NOT NULL,
 PRIMARY KEY(run_id,solution_id), UNIQUE(run_id,solution_hash),
 FOREIGN KEY(run_id) REFERENCES g_runs(run_id),
 FOREIGN KEY(run_id,manifest_artifact_id) REFERENCES g_artifacts(run_id,artifact_id)
) STRICT;
CREATE TABLE g_solution_edges(
 run_id TEXT NOT NULL, parent_solution_id TEXT NOT NULL, child_solution_id TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN('primary','idea_donor','ensemble_member')),
 PRIMARY KEY(run_id,parent_solution_id,child_solution_id,kind),
 FOREIGN KEY(run_id,parent_solution_id) REFERENCES g_solutions(run_id,solution_id),
 FOREIGN KEY(run_id,child_solution_id) REFERENCES g_solutions(run_id,solution_id),
 CHECK(parent_solution_id<>child_solution_id)
) STRICT;
CREATE UNIQUE INDEX g_one_primary_parent ON g_solution_edges(run_id,child_solution_id) WHERE kind='primary';
CREATE TRIGGER g_no_solution_cycle BEFORE INSERT ON g_solution_edges
WHEN EXISTS(
 WITH RECURSIVE reachable(id) AS (
   SELECT NEW.child_solution_id
   UNION
   SELECT e.child_solution_id FROM g_solution_edges e JOIN reachable r ON e.parent_solution_id=r.id WHERE e.run_id=NEW.run_id
 ) SELECT 1 FROM reachable WHERE id=NEW.parent_solution_id
) BEGIN SELECT RAISE(ABORT,'solution DAG cycle'); END;
CREATE TABLE g_experiment_links(
 run_id TEXT NOT NULL, experiment_id TEXT NOT NULL, solution_id TEXT NOT NULL,
 PRIMARY KEY(run_id,experiment_id), UNIQUE(run_id,experiment_id,solution_id),
 FOREIGN KEY(run_id,experiment_id) REFERENCES experiments(run_id,experiment_id),
 FOREIGN KEY(run_id,solution_id) REFERENCES g_solutions(run_id,solution_id)
) STRICT;
CREATE TABLE g_heads(
 run_id TEXT NOT NULL, branch_id TEXT NOT NULL, researcher_id TEXT NOT NULL,
 solution_id TEXT, experiment_id TEXT, revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0),
 PRIMARY KEY(run_id,branch_id,researcher_id),
 FOREIGN KEY(run_id,branch_id) REFERENCES g_branches(run_id,branch_id),
 FOREIGN KEY(run_id,solution_id) REFERENCES g_solutions(run_id,solution_id),
 FOREIGN KEY(run_id,experiment_id) REFERENCES g_experiment_links(run_id,experiment_id),
 FOREIGN KEY(run_id,experiment_id,solution_id) REFERENCES g_experiment_links(run_id,experiment_id,solution_id),
 CHECK((solution_id IS NULL)=(experiment_id IS NULL))
) STRICT;
CREATE TABLE g_checkpoints(
 run_id TEXT NOT NULL, checkpoint_id TEXT NOT NULL, branch_id TEXT NOT NULL,
 branch_revision INTEGER NOT NULL, store_revision INTEGER NOT NULL, graph_revision INTEGER NOT NULL, captured_event_seq INTEGER NOT NULL,
 artifact_id TEXT NOT NULL, snapshot_json TEXT NOT NULL CHECK(json_valid(snapshot_json)),
 PRIMARY KEY(run_id,checkpoint_id),
 FOREIGN KEY(run_id,branch_id) REFERENCES g_branches(run_id,branch_id),
 FOREIGN KEY(run_id,artifact_id) REFERENCES g_artifacts(run_id,artifact_id)
) STRICT;
CREATE TABLE g_work_orders(
 run_id TEXT NOT NULL, work_order_id TEXT NOT NULL, branch_id TEXT NOT NULL, researcher_id TEXT NOT NULL,
 branch_revision INTEGER NOT NULL, base_head_revision INTEGER NOT NULL,
 state TEXT NOT NULL CHECK(state IN('preparing','issued','submitted','stopped','revoked','expired','completed')),
 expires_elapsed_ms INTEGER NOT NULL CHECK(expires_elapsed_ms>0), order_json TEXT CHECK(order_json IS NULL OR json_valid(order_json)),
 PRIMARY KEY(run_id,work_order_id),
 FOREIGN KEY(run_id,branch_id,researcher_id) REFERENCES g_heads(run_id,branch_id,researcher_id)
) STRICT;
CREATE UNIQUE INDEX g_one_active_order_per_peer ON g_work_orders(run_id,researcher_id) WHERE state IN('preparing','issued','submitted');
CREATE TABLE g_snapshot_bindings(
 run_id TEXT NOT NULL, snapshot_id TEXT NOT NULL, work_order_id TEXT NOT NULL, binding_json TEXT NOT NULL CHECK(json_valid(binding_json)),
 PRIMARY KEY(run_id,snapshot_id), UNIQUE(run_id,work_order_id),
 FOREIGN KEY(run_id,work_order_id) REFERENCES g_work_orders(run_id,work_order_id)
) STRICT;
CREATE TABLE g_contexts(
 run_id TEXT NOT NULL, context_id TEXT NOT NULL, work_order_id TEXT NOT NULL,
 capsule_artifact_id TEXT NOT NULL, messages_artifact_id TEXT NOT NULL,
 PRIMARY KEY(run_id,context_id), UNIQUE(run_id,work_order_id),
 FOREIGN KEY(run_id,work_order_id) REFERENCES g_work_orders(run_id,work_order_id),
 FOREIGN KEY(run_id,capsule_artifact_id) REFERENCES g_artifacts(run_id,artifact_id),
 FOREIGN KEY(run_id,messages_artifact_id) REFERENCES g_artifacts(run_id,artifact_id)
) STRICT;
CREATE TABLE g_bindings(
 run_id TEXT NOT NULL, binding_id TEXT NOT NULL, work_order_id TEXT NOT NULL, experiment_id TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN('pending','applied','archived')), invalidated INTEGER NOT NULL DEFAULT 0 CHECK(invalidated IN(0,1)),
 outcome TEXT, PRIMARY KEY(run_id,binding_id), UNIQUE(run_id,work_order_id),
 FOREIGN KEY(run_id,work_order_id) REFERENCES g_work_orders(run_id,work_order_id),
 FOREIGN KEY(run_id,experiment_id) REFERENCES g_experiment_links(run_id,experiment_id)
) STRICT;
CREATE TABLE g_proposals(
 run_id TEXT NOT NULL, proposal_id TEXT NOT NULL, work_order_id TEXT NOT NULL,
 record_json TEXT NOT NULL CHECK(json_valid(record_json)), accepted INTEGER NOT NULL CHECK(accepted IN(0,1)),
 PRIMARY KEY(run_id,proposal_id),
 FOREIGN KEY(run_id,work_order_id) REFERENCES g_work_orders(run_id,work_order_id)
) STRICT;
CREATE TABLE g_ideas(
 run_id TEXT NOT NULL, idea_id TEXT NOT NULL, source_experiment_id TEXT NOT NULL,
 record_json TEXT NOT NULL CHECK(json_valid(record_json)), created_event_seq INTEGER NOT NULL,
 PRIMARY KEY(run_id,idea_id), FOREIGN KEY(run_id,source_experiment_id) REFERENCES g_experiment_links(run_id,experiment_id)
) STRICT;
CREATE TABLE g_assessments(
 run_id TEXT NOT NULL, assessment_id TEXT NOT NULL, branch_id TEXT NOT NULL,
 artifact_id TEXT NOT NULL, record_json TEXT NOT NULL CHECK(json_valid(record_json)),
 PRIMARY KEY(run_id,assessment_id), FOREIGN KEY(run_id,branch_id) REFERENCES g_branches(run_id,branch_id),
 FOREIGN KEY(run_id,artifact_id) REFERENCES g_artifacts(run_id,artifact_id)
) STRICT;
CREATE TABLE g_events(
 event_seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, event_id TEXT NOT NULL,
 record_json TEXT NOT NULL CHECK(json_valid(record_json)), UNIQUE(run_id,event_id),
 FOREIGN KEY(run_id) REFERENCES g_runs(run_id)
) STRICT;
CREATE INDEX g_events_run_seq ON g_events(run_id,event_seq);
CREATE TABLE g_idempotency(
 run_id TEXT NOT NULL, operation TEXT NOT NULL, idempotency_key TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
 response_json TEXT NOT NULL CHECK(json_valid(response_json)),
 PRIMARY KEY(run_id,operation,idempotency_key), FOREIGN KEY(run_id) REFERENCES g_runs(run_id)
) STRICT;
CREATE TABLE g_outbox(
 outbox_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN('prepare_submission','cancel_job','gc_blob')),
 dedup_key TEXT NOT NULL, payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
 state TEXT NOT NULL CHECK(state IN('pending','delivered','superseded')), UNIQUE(run_id,kind,dedup_key),
 FOREIGN KEY(run_id) REFERENCES g_runs(run_id)
) STRICT;
-- Immutable authoritative records. Recovery changes artifact status/intents, never solution history.
CREATE TRIGGER g_solution_immutable BEFORE UPDATE ON g_solutions BEGIN SELECT RAISE(ABORT,'immutable solution'); END;
CREATE TRIGGER g_checkpoint_immutable BEFORE UPDATE ON g_checkpoints BEGIN SELECT RAISE(ABORT,'immutable checkpoint'); END;
CREATE TRIGGER g_idea_immutable BEFORE UPDATE ON g_ideas BEGIN SELECT RAISE(ABORT,'immutable idea'); END;
CREATE TRIGGER g_event_immutable BEFORE UPDATE ON g_events BEGIN SELECT RAISE(ABORT,'immutable event'); END;
CREATE TRIGGER g_edge_immutable BEFORE UPDATE ON g_solution_edges BEGIN SELECT RAISE(ABORT,'immutable edge'); END;
CREATE TRIGGER g_artifact_ready_immutable BEFORE UPDATE OF artifact_id,run_id,relative_path,sha256,size_bytes,kind,media_type ON g_artifacts
 WHEN OLD.state='ready' BEGIN SELECT RAISE(ABORT,'immutable ready artifact'); END;
INSERT INTO g_schema_migrations VALUES(1,strftime('%Y-%m-%dT%H:%M:%fZ','now'));

CREATE TABLE g_artifact_locations(
 run_id TEXT NOT NULL, artifact_id TEXT NOT NULL, root_kind TEXT NOT NULL CHECK(root_kind IN('run','task')),
 root_relative_path TEXT NOT NULL, external INTEGER NOT NULL CHECK(external IN(0,1)),
 PRIMARY KEY(run_id,artifact_id), FOREIGN KEY(run_id,artifact_id) REFERENCES g_artifacts(run_id,artifact_id)
) STRICT;
CREATE TABLE g_components(
 run_id TEXT NOT NULL, component_id TEXT NOT NULL, record_json TEXT NOT NULL CHECK(json_valid(record_json)),
 PRIMARY KEY(run_id,component_id), FOREIGN KEY(run_id) REFERENCES g_runs(run_id)
) STRICT;
CREATE TABLE g_component_trials(
 run_id TEXT NOT NULL, component_trial_id TEXT NOT NULL, component_id TEXT NOT NULL,
 before_experiment_id TEXT NOT NULL, after_experiment_id TEXT NOT NULL,
 record_json TEXT NOT NULL CHECK(json_valid(record_json)), PRIMARY KEY(run_id,component_trial_id),
 FOREIGN KEY(run_id,component_id) REFERENCES g_components(run_id,component_id),
 FOREIGN KEY(run_id,before_experiment_id) REFERENCES g_experiment_links(run_id,experiment_id),
 FOREIGN KEY(run_id,after_experiment_id) REFERENCES g_experiment_links(run_id,experiment_id)
) STRICT;
CREATE TABLE g_decisions(
 run_id TEXT NOT NULL, decision_id TEXT NOT NULL, checkpoint_id TEXT NOT NULL, artifact_id TEXT NOT NULL,
 record_json TEXT NOT NULL CHECK(json_valid(record_json)), PRIMARY KEY(run_id,decision_id),
 FOREIGN KEY(run_id,checkpoint_id) REFERENCES g_checkpoints(run_id,checkpoint_id),
 FOREIGN KEY(run_id,artifact_id) REFERENCES g_artifacts(run_id,artifact_id)
) STRICT;
CREATE TABLE g_metadata(
 run_id TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL CHECK(json_valid(value)),
 PRIMARY KEY(run_id,name), FOREIGN KEY(run_id) REFERENCES g_runs(run_id)
) STRICT;
CREATE TABLE g_branch_evaluations(
 run_id TEXT NOT NULL, branch_id TEXT NOT NULL, experiment_id TEXT NOT NULL, evaluation_id TEXT NOT NULL,
 solution_id TEXT NOT NULL, event_seq INTEGER NOT NULL, elapsed_ms INTEGER NOT NULL,
 PRIMARY KEY(run_id,branch_id,experiment_id),
 FOREIGN KEY(run_id,branch_id) REFERENCES g_branches(run_id,branch_id),
 FOREIGN KEY(run_id,experiment_id) REFERENCES g_experiment_links(run_id,experiment_id),
 FOREIGN KEY(evaluation_id) REFERENCES evaluations(evaluation_id),
 FOREIGN KEY(run_id,solution_id) REFERENCES g_solutions(run_id,solution_id)
) STRICT;
CREATE TABLE g_attempts(
 run_id TEXT NOT NULL, experiment_id TEXT NOT NULL, attempt_no INTEGER NOT NULL, record_json TEXT NOT NULL CHECK(json_valid(record_json)),
 PRIMARY KEY(run_id,experiment_id,attempt_no),
 FOREIGN KEY(run_id,experiment_id) REFERENCES g_experiment_links(run_id,experiment_id)
) STRICT;
CREATE TABLE g_idea_candidates(
 run_id TEXT NOT NULL, source_experiment_id TEXT NOT NULL, record_json TEXT NOT NULL CHECK(json_valid(record_json)),
 PRIMARY KEY(run_id,source_experiment_id),
 FOREIGN KEY(run_id,source_experiment_id) REFERENCES g_experiment_links(run_id,experiment_id)
) STRICT;
CREATE INDEX g_binding_exp_state ON g_bindings(run_id,experiment_id,state);
CREATE INDEX g_branch_eval_order ON g_branch_evaluations(run_id,branch_id,event_seq);
CREATE TRIGGER g_no_delete_solution BEFORE DELETE ON g_solutions BEGIN SELECT RAISE(ABORT,'immutable solution'); END;
CREATE TRIGGER g_no_delete_checkpoint BEFORE DELETE ON g_checkpoints BEGIN SELECT RAISE(ABORT,'immutable checkpoint'); END;
CREATE TRIGGER g_no_delete_idea BEFORE DELETE ON g_ideas BEGIN SELECT RAISE(ABORT,'immutable idea'); END;
CREATE TRIGGER g_no_delete_event BEFORE DELETE ON g_events BEGIN SELECT RAISE(ABORT,'immutable event'); END;
CREATE TRIGGER g_no_delete_edge BEFORE DELETE ON g_solution_edges BEGIN SELECT RAISE(ABORT,'immutable edge'); END;
CREATE TRIGGER g_constraint_no_update BEFORE UPDATE ON g_constraints BEGIN SELECT RAISE(ABORT,'immutable inherited constraint'); END;
CREATE TRIGGER g_constraint_no_delete BEFORE DELETE ON g_constraints BEGIN SELECT RAISE(ABORT,'immutable inherited constraint'); END;
CREATE TRIGGER g_trial_no_update BEFORE UPDATE ON g_component_trials BEGIN SELECT RAISE(ABORT,'immutable component trial'); END;
CREATE TRIGGER g_trial_no_delete BEFORE DELETE ON g_component_trials BEGIN SELECT RAISE(ABORT,'immutable component trial'); END;
