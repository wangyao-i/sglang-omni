"""Read a saved GDB log; never import torch or attach to a live process."""

import argparse
import json
import re
from pathlib import Path

CATEGORIES = {
    "stream_access": r"NPUStream::stream|MakeSureQueueEmpty",
    "queue_consumer": r"StartConsume|Repository::(?:Dequeue|ReadQueue)|ConsumeTask|TaskQueue",
    "gil": r"take_gil|PyEval_RestoreThread|PyGILState_Ensure|acquire_timed",
    "runtime_wait": r"aclrtSynchronize|StreamSynchronize|EventSynchronize|rtStreamSynchronize",
    "graph_update": r"graph_task_update|CaptureTaskUpdate",
    "copy": r"at::native::to\b|copy_|Memcpy",
}


def summarize(text):
    # Match backtrace headers, not rows in `info threads`.
    blocks = re.split(r"(?m)(?=^Thread \d+\b)", text)
    rows = []
    for block in blocks:
        if not re.match(r"Thread \d+\b", block):
            continue
        header = block.splitlines()[0]
        lwp = re.search(r"LWP\s+(\d+)", header)
        categories = [
            key for key, pattern in CATEGORIES.items() if re.search(pattern, block)
        ]
        frames = re.findall(r"(?m)^#\d+\b.*", block)
        rows.append(
            {
                "header": header,
                "lwp": int(lwp.group(1)) if lwp else None,
                "categories": categories,
                "frames_observed": len(frames),
                "stack": block if categories else None,
            }
        )
    return {
        "backtrace_blocks_observed": len(rows),
        "category_counts": {
            key: sum(key in row["categories"] for row in rows) for key in CATEGORIES
        },
        "threads": rows,
        "limitation": "Matches are triage hints, not ownership proof. Missing blocks may reflect GDB timeout.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.log.read_text(errors="replace"))
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "threads"}, indent=2))


if __name__ == "__main__":
    main()
