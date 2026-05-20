"""Resumable, batched ETL — convert Djinni + LinkedIn postings via the atlm_teacher agent.

Both sources are interleaved with a fixed-seed shuffle, so every batch is a
Djinni+LinkedIn mix; each output record carries a `source` field, so they
stay separable (Djinni -> train, LinkedIn -> OOD test).

Append-only and resumable: progress is the output file itself. Re-run to
continue where the last batch stopped. Each run is bounded by --count and/or
--minutes (whichever is hit first); a SIGINT/SIGTERM finishes in-flight work
and exits cleanly.

  .venv_atlm_pro/bin/python src/run_etl.py --count 2500
  .venv_atlm_pro/bin/python src/run_etl.py --minutes 30
  .venv_atlm_pro/bin/python src/run_etl.py            # -> remaining, no limit
"""
import argparse
import json
import queue
import random
import re
import signal
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DJINNI = ROOT / "data/jobs/djinni/train-00000-of-00001.parquet"
LINKEDIN = ROOT / "data/jobs/linkedin/postings.csv"
OUT_DIR = ROOT / "data/processed"
CONVERTED = OUT_DIR / "converted.jsonl"   # successful records (the asset)
SKIPPED = OUT_DIR / "skipped.jsonl"       # teacher said <SKIP> (final, not retried)

ENDPOINT = "http://localhost:7701/v1/chat/completions"
AGENT = "atlm_teacher"
SEED = 42
MIN_CHARS = 150          # drop stub postings
WORKERS = 8              # matches llama-vision's 8 slots
TIMEOUT = 300

SECTIONS = ["## Summary", "## Required Skills",
            "## Responsibilities", "## Requirements"]


# ----------------------------------------------------------------- data
def build_work_list():
    """All postings from both sources, lightly filtered, interleaved (seeded)."""
    items = []

    dj = pd.read_parquet(DJINNI, columns=["id", "Position", "Long Description"])
    dj.columns = ["sid", "title", "body"]
    dj = dj[dj["body"].str.len() >= MIN_CHARS]
    for r in dj.itertuples(index=False):
        items.append({"id": f"djinni:{r.sid}", "source": "djinni",
                      "source_id": str(r.sid), "title": str(r.title),
                      "body": str(r.body)})

    lk = pd.read_csv(LINKEDIN, usecols=["job_id", "title", "description"])
    lk = lk[lk["description"].notna()]
    lk["description"] = lk["description"].astype(str)
    lk = lk[lk["description"].str.len() >= MIN_CHARS]
    lk = lk.drop_duplicates(subset=["description"])      # ~16k reposts removed
    for r in lk.itertuples(index=False):
        title = "" if pd.isna(r.title) else str(r.title)
        items.append({"id": f"linkedin:{r.job_id}", "source": "linkedin",
                      "source_id": str(r.job_id), "title": title,
                      "body": str(r.description)})

    random.Random(SEED).shuffle(items)
    return items


def load_done():
    """IDs already converted or skipped — used to resume."""
    done = set()
    for path in (CONVERTED, SKIPPED):
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:  # noqa: BLE001
                        pass
    return done


# ------------------------------------------------------------- teacher
def parse_teacher_output(text):
    text = (text or "").strip()
    m = re.search(r"<SKIP>(.*?)</SKIP>", text, re.S)
    if m:
        return {"skip": True, "reason": m.group(1).strip()}
    q = re.search(r"<QUERIES>(.*?)</QUERIES>", text, re.S)
    jd = re.search(r"<JOB_DESCRIPTION>(.*?)(?:</JOB_DESCRIPTION>|\Z)", text, re.S)
    queries = []
    if q:
        for line in q.group(1).strip().splitlines():
            mm = re.match(r"\s*\d+[.)]\s*(.+)", line)
            if mm:
                queries.append(mm.group(1).strip())
    jd_text = jd.group(1).strip() if jd else ""
    ok = (len(queries) == 3 and jd_text
          and re.search(r"^#\s+\S", jd_text, re.M)
          and all(s in jd_text for s in SECTIONS))
    return {"skip": False, "ok": bool(ok), "queries": queries, "jd": jd_text}


def call_teacher(text):
    payload = {"model": AGENT, "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        resp = json.load(r)
    return resp["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=0,
                    help="stop after N successful conversions this run")
    ap.add_argument("--minutes", type=float, default=0,
                    help="stop after M minutes this run")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building work list (reading both sources)...", flush=True)
    t_build = time.time()
    work = build_work_list()
    done = load_done()
    todo = [w for w in work if w["id"] not in done]
    print(f"total postings : {len(work):,}", flush=True)
    print(f"already done   : {len(done):,}", flush=True)
    print(f"remaining      : {len(todo):,}  (work list built in "
          f"{time.time() - t_build:.0f}s)", flush=True)
    if not todo:
        print("nothing to do — everything is converted.", flush=True)
        return

    limit = args.count or len(todo)
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    print(f"this batch: up to {limit:,} records"
          + (f" or {args.minutes:g} min" if deadline else "")
          + f", {args.workers} workers\n", flush=True)

    q = queue.Queue()
    for w in todo:
        q.put(w)

    stop = threading.Event()
    lock = threading.Lock()
    stats = {"ok": 0, "skip": 0, "err": 0}
    t0 = time.time()

    def handle_sig(*_):
        print("\n[signal] finishing in-flight work and stopping...", flush=True)
        stop.set()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    conv_f = open(CONVERTED, "a", encoding="utf-8")
    skip_f = open(SKIPPED, "a", encoding="utf-8")

    def worker():
        while not stop.is_set():
            if deadline and time.time() > deadline:
                stop.set()
                return
            try:
                w = q.get_nowait()
            except queue.Empty:
                return
            text = f"{w['title']}\n\n{w['body']}"
            try:
                parsed = parse_teacher_output(call_teacher(text))
            except Exception:  # noqa: BLE001 — HTTP/timeout: leave for next run
                with lock:
                    stats["err"] += 1
                continue
            now = datetime.now(timezone.utc).isoformat()
            with lock:
                if parsed["skip"]:
                    skip_f.write(json.dumps(
                        {"id": w["id"], "source": w["source"],
                         "reason": parsed["reason"], "at": now},
                        ensure_ascii=False) + "\n")
                    skip_f.flush()
                    stats["skip"] += 1
                elif parsed["ok"]:
                    conv_f.write(json.dumps(
                        {"id": w["id"], "source": w["source"],
                         "source_id": w["source_id"], "queries": parsed["queries"],
                         "job_description": parsed["jd"], "converted_at": now},
                        ensure_ascii=False) + "\n")
                    conv_f.flush()
                    stats["ok"] += 1
                else:
                    stats["err"] += 1   # malformed output -> retried next run
                n = stats["ok"]
                if n and n % 50 == 0:
                    el = time.time() - t0
                    print(f"  {n} ok | {stats['skip']} skip | {stats['err']} err "
                          f"| {n / el:.2f}/s | {el / 60:.1f} min", flush=True)
                if stats["ok"] >= limit:
                    stop.set()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    conv_f.close()
    skip_f.close()

    el = time.time() - t0
    total = len(done) + stats["ok"] + stats["skip"]
    remaining = len(work) - total
    rate = stats["ok"] / el if el else 0
    print("\n=== batch done ===", flush=True)
    print(f"  converted this run : {stats['ok']}", flush=True)
    print(f"  skipped            : {stats['skip']}", flush=True)
    print(f"  errors (retry next): {stats['err']}", flush=True)
    print(f"  time               : {el / 60:.1f} min  ({rate:.2f}/s)", flush=True)
    print(f"  total done overall : {total:,} / {len(work):,}  "
          f"({remaining:,} remaining)", flush=True)
    if rate and remaining:
        print(f"  est. to finish all : ~{remaining / rate / 3600:.1f} h "
              f"at this rate", flush=True)


if __name__ == "__main__":
    main()
