"""Inspect LLM token/latency metadata from a run directory or ZIP; never load CSVs/models."""
from pathlib import Path
import argparse
import json
import math
import statistics
import zipfile
from datetime import datetime


def job_occupancy(events):
    """Wall intervals with a train/refit job; this is not measured CPU utilization."""
    if not events:
        return {}
    if any('timestamp' not in e for e in events):
        return {'timing_available': False, 'interpretation': 'This metadata has no complete timestamp history; wall occupancy is unknown.'}
    start, end = (datetime.fromisoformat(events[i]['timestamp']).timestamp() for i in (0, -1))
    opened, intervals = {}, []
    for event in events:
        stamp = datetime.fromisoformat(event['timestamp']).timestamp()
        if event['event'] == 'job.started':
            opened[event['job_id']] = stamp
        elif event['event'] == 'job.finished' and event['job_id'] in opened:
            intervals.append((opened.pop(event['job_id']), stamp))
    intervals.extend((stamp, end) for stamp in opened.values())
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    busy = sum(right-left for left, right in merged)
    return {'elapsed_seconds': round(end-start, 3), 'job_present_seconds': round(busy, 3),
            'no_train_or_refit_seconds': round(max(0, end-start-busy), 3),
            'unfinished_jobs': len(opened), 'interpretation': 'Job presence from saved events, not CPU utilization.'}


def research_summary(events):
    """Summarize measured development, not tokens as a proxy for intelligence."""
    evaluations = {}
    for event in events:
        if event.get('event') == 'experiment.evaluated' and event.get('experiment_id'):
            evaluations[event['experiment_id']] = event
    by_recipe = {}
    owned_features = constant = 0
    for event in evaluations.values():
        recipe = event.get('recipe_id', 'unknown')
        row = by_recipe.setdefault(recipe, {'evaluations': 0, 'best_score': None, 'best_experiment_id': None})
        row['evaluations'] += 1
        score = event.get('score')
        if isinstance(score, (int, float)) and math.isfinite(score):
            if row['best_score'] is None or ((score < row['best_score']) if event.get('direction') == 'minimize' else (score > row['best_score'])):
                row.update(best_score=score, best_experiment_id=event['experiment_id'])
        owned_features += bool(event.get('hyperparameters', {}).get('feature_program'))
        constant += bool(event.get('prediction_behavior', {}).get('constant_like'))
    actions = {}
    for event in events:
        if event.get('event') == 'research.action_proposed':
            kind = event.get('action', 'unknown')
            actions[kind] = actions.get(kind, 0)+1
    return {'evaluated_configurations': len(evaluations), 'by_recipe': by_recipe,
            'proposed_llm_actions': actions, 'evaluated_owned_feature_programs': owned_features,
            'observed_constant_like_outputs': constant,
            'completed_data_analyses': sum(e.get('event') == 'research.analysis_completed' for e in events),
            'rejected_reference_optimizations': sum(e.get('code') == 'REFERENCE_ONLY_RECIPE' and e.get('event') == 'research.proposal_failed' for e in events),
            'interpretation': 'Proposals are not executions. Counts describe recorded behavior; a feature program can regress, '
                'and custom pipelines without an owned program need their code/fitted report inspected.'}


def inspect_run(source):
    source = Path(source)
    if source.is_dir():
        names = [p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file() and not p.is_symlink()]
        def read(name):
            path = source/name
            if path.stat().st_size > 8*1024*1024:
                raise ValueError("Metadata file too large")
            return path.read_text(encoding="utf-8")
        archive = None
        sizes = {name: (source/name).stat().st_size for name in names}
    else:
        archive = zipfile.ZipFile(source)
        names = archive.namelist()
        sizes = {item.filename: item.file_size for item in archive.infolist() if not item.is_dir()}
        def read(name):
            if archive.getinfo(name).file_size > 8*1024*1024:
                raise ValueError("Metadata file too large")
            return archive.read(name).decode("utf-8")
    try:
        event_files = sorted(n for n in names if Path(n).name == "events.jsonl")
        results = []
        for event_file in event_files:
            prefix = event_file[:-len("events.jsonl")]
            events = [json.loads(line) for line in read(event_file).splitlines() if line.strip()]
            completed = {e["call_no"]:e for e in events if e.get("event")=="llm.call_completed" and "call_no" in e}
            rows = []
            for name in sorted(n for n in names if n.startswith(prefix+"llm/") and n.endswith(".response.json")):
                number = int(Path(name).name.split(".")[0])
                data = json.loads(read(name))
                request_name = name.replace(".response.json", ".request.json")
                request = json.loads(read(request_name)) if request_name in names else {}
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                usage = data.get("usage") or {}
                broken_tools = False
                for call in message.get("tool_calls") or []:
                    try:
                        json.loads(call["function"]["arguments"])
                    except (ValueError, KeyError, TypeError):
                        broken_tools = True
                at_cap = bool(request.get("max_tokens")) and usage.get("completion_tokens", 0) >= request["max_tokens"]
                duration = completed.get(number,{}).get("duration_ms")
                reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                rows.append({"call_no":number,"requested_output_tokens":request.get("max_tokens"),
                             "finish_reason":choice.get("finish_reason"),"output_tokens":usage.get("completion_tokens"),
                             "input_tokens":usage.get("prompt_tokens"),
                             "cached_input_tokens":(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                             "reasoning_tokens_reported":reasoning,
                             "usage_inconsistent":isinstance(reasoning,int) and reasoning > usage.get("completion_tokens",0),
                             "content_chars":len(message.get("content") or ""),"invalid_tool_arguments":broken_tools,
                             "truncated":choice.get("finish_reason")=="length" or (at_cap and broken_tools),
                             "duration_seconds":duration/1000 if duration is not None else None})
            maximum = max((r["requested_output_tokens"] or 0 for r in rows),default=0)
            censored = sum(r["truncated"] for r in rows)
            initial = max(16384, 2**math.ceil(math.log2(maximum+1))) if censored else max(16384,maximum)
            initial = min(initial,131072)
            valid_outputs = [r["output_tokens"] for r in rows if not r["truncated"] and isinstance(r["output_tokens"],int)]
            state = next((e.get("state") for e in reversed(events) if e.get("event")=="run.finished"),
                         'failed' if any(e.get('event') == 'run.failed' for e in events) else 'incomplete')
            categories = {}
            for name, size in sizes.items():
                if not name.startswith(prefix):
                    continue
                relative = name[len(prefix):]
                category = "input_cache" if relative.startswith('data-cache/') else "model_snapshots" if relative.endswith(".joblib") else "worker_logs" if relative.endswith(".log") else "predictions_csv" if relative.endswith(".csv") else "other_metadata"
                categories[category] = categories.get(category, 0)+size
            storage = {"total_bytes": sum(categories.values()), "by_category": categories,
                       "largest_files": [{"path": n[len(prefix):], "size_bytes": size} for n, size in sorted(sizes.items(), key=lambda p: p[1], reverse=True) if n.startswith(prefix)][:10]}
            results.append({"run":events[0].get("run_id",prefix.rstrip('/')) if events else prefix,
                            "state":state,"api_attempts":sum(e.get("event")=="llm.call_started" for e in events),
                            "responses":len(rows),"truncated_responses":censored,
                            "reported_output_tokens":sum(r["output_tokens"] or 0 for r in rows),
                            "reported_input_tokens":sum(r["input_tokens"] or 0 for r in rows),
                            "reported_cached_input_tokens":sum(r["cached_input_tokens"] or 0 for r in rows),
                            "uncensored_output_median":statistics.median(valid_outputs) if valid_outputs else None,
                            "suggested_initial_output_tokens":initial,"suggested_output_ceiling":min(131072,initial*2),
                            "budget_note":"Client-side starting estimate, not a verified provider maximum or a RUB conversion. Truncated samples are censored; exact token need is unknown.",
                            "research_decisions":sum(e.get('event') == 'research.decision_saved' for e in events),
                            "feedback_deliveries":sum(e.get('event') == 'research.feedback_delivered' for e in events),
                            "job_occupancy":job_occupancy(events),
                            "research_summary":research_summary(events),
                            'scientific_research': {
                                'test_plans': sum(e.get('event') == 'research.decision_saved' and e.get('test_plan_supplied', False) for e in events),
                                'followups': sum(e.get('event') == 'research.decision_saved' and e.get('continuation') == 'followup' for e in events),
                                'oof_reports': sum(e.get('event') == 'diagnostics.oof_completed' for e in events),
                                'oof_unavailable': sum(e.get('event') == 'diagnostics.oof_unavailable' for e in events),
                                'preflight_deferrals': sum(e.get('event') == 'research.preflight_reviewed' and e.get('admission') == 'deferred' for e in events),
                                'extended_dialogues': sum(e.get('event') == 'llm.deliberation_budget' and e.get('extended_funded', False) for e in events),
                                'reported_reasoning_tokens': sum(e.get('reasoning_tokens') or 0 for e in events if e.get('event') == 'llm.call_completed'),
                                'calls_without_reasoning_usage': sum(e.get('event') == 'llm.call_completed' and e.get('reasoning_tokens') is None for e in events),
                                'calls_with_reasoning_field': sum(e.get('event') == 'llm.call_completed' and e.get('reasoning_field_present', False) for e in events),
                                'provider_thinking_application': 'unverified; request flags and reported usage do not measure decision quality'},
                            "dialogues":[e for e in events if e.get('event') == 'research.dialogue_summary'],
                            "feature_seed_proposals":sum(e.get('event') == 'features.seed_proposed' for e in events),
                            "overlap_deferrals":sum(e.get('event') == 'research.admission_reconsidered' for e in events),
                            "rolling_peer_resumptions":sum(e.get('event') == 'research.peer_resumed_while_waiting' for e in events),
                            "search_stop_reason":next((e.get('stop_reason') for e in reversed(events) if e.get('event') == 'search.finished'), None),
                            "fallback_seconds":next((e.get('fallback_seconds') for e in reversed(events) if e.get('event') == 'search.finished'), None),
                            "final_confirmation":next((e for e in reversed(events) if e.get('event') == 'validation.finalists_confirmed'), None),
                            "published_pipeline":next(({'experiment_id': e.get('experiment_id'), 'selection_basis': e.get('selection_basis'),
                                'confirmation_artifact_id': e.get('confirmation_artifact_id')} for e in reversed(events) if e.get('event') == 'submission.published'), None),
                            "storage":storage, "calls":rows})
        return results
    finally:
        if archive is not None:
            archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--json",action="store_true")
    args = parser.parse_args()
    reports = inspect_run(args.source)
    if args.json:
        print(json.dumps(reports,ensure_ascii=False,indent=2))
    else:
        for report in reports:
            print(f"{report['run']} state={report['state']} attempts={report['api_attempts']} responses={report['responses']}")
            print(f"Output reported={report['reported_output_tokens']}; input={report['reported_input_tokens']}; cached input={report['reported_cached_input_tokens']}; truncated={report['truncated_responses']}")
            print('Storage: '+json.dumps(report['storage'], ensure_ascii=False))
            print('Research: '+json.dumps({k: report[k] for k in ('research_decisions', 'feedback_deliveries',
                'rolling_peer_resumptions', 'search_stop_reason', 'fallback_seconds', 'published_pipeline', 'research_summary')}, ensure_ascii=False))
            for row in report['calls']:
                print('  '+json.dumps(row,ensure_ascii=False))
            print(f"Suggested initial/ceiling: {report['suggested_initial_output_tokens']}/{report['suggested_output_ceiling']}")
            print(report['budget_note'])
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
