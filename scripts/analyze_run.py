"""Inspect LLM token/latency metadata from a run directory or ZIP; never load CSVs/models."""
from pathlib import Path
import argparse
import json
import math
import statistics
import zipfile


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
    else:
        archive = zipfile.ZipFile(source)
        names = archive.namelist()
        def read(name):
            if archive.getinfo(name).file_size > 8*1024*1024:
                raise ValueError("Metadata file too large")
            return archive.read(name).decode("utf-8")
    try:
        event_files = sorted(n for n in names if n.endswith("events.jsonl"))
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
            state = next((e.get("state") for e in reversed(events) if e.get("event")=="run.finished"),"failed_or_incomplete")
            results.append({"run":events[0].get("run_id",prefix.rstrip('/')) if events else prefix,
                            "state":state,"api_attempts":sum(e.get("event")=="llm.call_started" for e in events),
                            "responses":len(rows),"truncated_responses":censored,
                            "reported_output_tokens":sum(r["output_tokens"] or 0 for r in rows),
                            "reported_input_tokens":sum(r["input_tokens"] or 0 for r in rows),
                            "reported_cached_input_tokens":sum(r["cached_input_tokens"] or 0 for r in rows),
                            "uncensored_output_median":statistics.median(valid_outputs) if valid_outputs else None,
                            "suggested_initial_output_tokens":initial,"suggested_output_ceiling":min(131072,initial*2),
                            "budget_note":"Client-side starting estimate, not a verified provider maximum or a RUB conversion. Truncated samples are censored; exact token need is unknown.",
                            "calls":rows})
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
            for row in report['calls']:
                print('  '+json.dumps(row,ensure_ascii=False))
            print(f"Suggested initial/ceiling: {report['suggested_initial_output_tokens']}/{report['suggested_output_ceiling']}")
            print(report['budget_note'])
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
