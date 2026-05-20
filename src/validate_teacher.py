"""Validate the atlm_teacher agent BEFORE the bulk ETL run.

Runs the teacher on a small, deliberately diverse sample of postings
(Djinni + LinkedIn, varied keywords and lengths), parses each output,
checks format compliance, measures latency, and writes a human-reviewable
report to data/processed/teacher_validation.md.

Run:  .venv_atlm_pro/bin/python src/validate_teacher.py
"""
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------- config
AGENT_SERVER = "http://localhost:7701"
AGENT = "atlm_teacher"                       # call by agent name (one-call A1)
ENDPOINT = f"{AGENT_SERVER}/v1/chat/completions"
WORKERS = 3
TIMEOUT = 300
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DJINNI = ROOT / "data/jobs/djinni/train-00000-of-00001.parquet"
LINKEDIN = ROOT / "data/jobs/linkedin/postings.csv"
OUT_DIR = ROOT / "data/processed"
OUT_MD = OUT_DIR / "teacher_validation.md"
OUT_JSONL = OUT_DIR / "teacher_validation_raw.jsonl"

DJINNI_KEYWORDS = ["JavaScript", "Java", "DevOps", "Python", "QA Automation",
                   ".NET", "Node.js", "PHP", "Project Manager", "HR",
                   "Design", "Marketing"]
SECTIONS = ["## Summary", "## Required Skills",
            "## Responsibilities", "## Requirements"]


# ------------------------------------------------------------ sampling
def build_sample():
    """Return a list of dicts: {source, tag, title, body}."""
    items = []

    dj = pd.read_parquet(
        DJINNI, columns=["Position", "Long Description", "Primary Keyword"])
    dlen = dj["Long Description"].str.len()

    # one normal-length posting per keyword, spread across the tech mix
    for kw in DJINNI_KEYWORDS:
        pool = dj[(dj["Primary Keyword"] == kw) & dlen.between(600, 2500)]
        if len(pool):
            r = pool.sample(1, random_state=SEED).iloc[0]
            items.append(dict(source="djinni", tag=kw,
                              title=r["Position"], body=r["Long Description"]))

    # stress cases: very short and very long postings
    for _, r in dj[dlen.between(200, 400)].sample(2, random_state=SEED).iterrows():
        items.append(dict(source="djinni", tag="SHORT",
                          title=r["Position"], body=r["Long Description"]))
    for _, r in dj[dlen.between(5000, 10000)].sample(2, random_state=SEED).iterrows():
        items.append(dict(source="djinni", tag="LONG",
                          title=r["Position"], body=r["Long Description"]))

    # out-of-domain: LinkedIn cross-industry postings
    lk = pd.read_csv(LINKEDIN, usecols=["title", "description"])
    lk = lk[lk["description"].notna()]
    lk = lk[lk["description"].str.len().between(800, 4000)]
    for _, r in lk.sample(4, random_state=SEED).iterrows():
        items.append(dict(source="linkedin", tag="cross-industry",
                          title=str(r["title"]), body=str(r["description"])))

    return items


# ------------------------------------------------------------ teacher call
def call_teacher(posting_text):
    payload = {"model": AGENT,
               "messages": [{"role": "user", "content": posting_text}]}
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            resp = json.load(r)
        out = resp["choices"][0]["message"]["content"]
        return {"output": out, "latency": time.time() - t0, "error": None}
    except urllib.error.HTTPError as e:
        return {"output": "", "latency": time.time() - t0,
                "error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:  # noqa: BLE001
        return {"output": "", "latency": time.time() - t0, "error": repr(e)}


# ------------------------------------------------------------ parsing
def parse_output(text):
    text = (text or "").strip()
    m = re.search(r"<SKIP>(.*?)</SKIP>", text, re.S)
    if m:
        return dict(is_skip=True, skip_reason=m.group(1).strip(),
                    queries=[], jd="", has_q=False, has_jd=False)
    q = re.search(r"<QUERIES>(.*?)</QUERIES>", text, re.S)
    jd = re.search(r"<JOB_DESCRIPTION>(.*?)(?:</JOB_DESCRIPTION>|\Z)", text, re.S)
    queries = []
    if q:
        for line in q.group(1).strip().splitlines():
            mm = re.match(r"\s*\d+[.)]\s*(.+)", line)
            if mm:
                queries.append(mm.group(1).strip())
    return dict(is_skip=False, skip_reason="", queries=queries,
                jd=jd.group(1).strip() if jd else "",
                has_q=bool(q), has_jd=bool(jd))


def evaluate(parsed):
    if parsed["is_skip"]:
        return dict(verdict="SKIP", issues=[], leaks=[])
    issues, jd = [], parsed["jd"]
    if not parsed["has_q"]:
        issues.append("no <QUERIES> block")
    if len(parsed["queries"]) != 3:
        issues.append(f"{len(parsed['queries'])} queries (expected 3)")
    if not parsed["has_jd"]:
        issues.append("no <JOB_DESCRIPTION> block")
    if not re.search(r"^#\s+\S", jd, re.M):
        issues.append("no H1 title")
    for s in SECTIONS:
        if s not in jd:
            issues.append(f"missing '{s}'")
    leaks = []
    if re.search(r"[\w.]+@[\w.]+\.\w+", jd):
        leaks.append("email")
    if re.search(r"https?://", jd):
        leaks.append("url")
    if re.search(r"\bapply\b", jd, re.I):
        leaks.append("'apply'")
    return dict(verdict="OK" if not issues else "FAIL", issues=issues, leaks=leaks)


# ------------------------------------------------------------ main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = build_sample()
    print(f"sample: {len(sample)} postings — calling '{AGENT}' "
          f"(concurrency {WORKERS})...")

    def work(item):
        text = f"{item['title']}\n\n{item['body']}"
        res = call_teacher(text)
        parsed = parse_output(res["output"])
        ev = evaluate(parsed)
        print(f"  [{ev['verdict']:4}] {item['source']:8} {item['tag']:18} "
              f"{res['latency']:5.1f}s"
              + (f"  ERROR {res['error']}" if res["error"] else ""))
        return {**item, **res, "parsed": parsed, "eval": ev}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(work, sample))
    wall = time.time() - t0

    # ---- aggregate ----
    n = len(results)
    ok = sum(r["eval"]["verdict"] == "OK" for r in results)
    skip = sum(r["eval"]["verdict"] == "SKIP" for r in results)
    fail = sum(r["eval"]["verdict"] == "FAIL" for r in results)
    errors = sum(r["error"] is not None for r in results)
    leaks = sum(bool(r["eval"]["leaks"]) for r in results)
    lat = sorted(r["latency"] for r in results if r["error"] is None)
    med = lat[len(lat) // 2] if lat else 0.0
    rate = n / wall if wall else 0.0          # postings / second (measured)
    usable_frac = (ok) / n if n else 0.0

    sect = {s: sum(s in r["parsed"]["jd"] for r in results
                   if not r["parsed"]["is_skip"]) for s in SECTIONS}
    non_skip = n - skip

    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY  ({n} postings)")
    print("=" * 60)
    print(f"  OK              : {ok}/{n}")
    print(f"  SKIP            : {skip}/{n}")
    print(f"  FAIL            : {fail}/{n}")
    print(f"  call errors     : {errors}/{n}")
    print(f"  leak suspects   : {leaks}/{n}  (email/url/'apply' in description)")
    print(f"  section presence (of {non_skip} non-skip):")
    for s, c in sect.items():
        print(f"    {s:24}: {c}/{non_skip}")
    print(f"  latency (s)     : median {med:.1f} | "
          f"min {lat[0]:.1f} | max {lat[-1]:.1f}" if lat else "  latency: n/a")
    print(f"  wall time       : {wall:.0f}s for {n} calls @ concurrency {WORKERS}")
    if rate:
        per_h = rate * 3600
        for hrs in (6, 8, 10):
            post = int(per_h * hrs)
            pairs = int(post * usable_frac * 3)
            print(f"  -> {hrs}h overnight: ~{post:,} postings "
                  f"-> ~{pairs:,} training pairs (3 queries each, OK only)")

    # ---- write artifacts ----
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({k: r[k] for k in
                    ("source", "tag", "title", "output", "latency", "error")},
                    ensure_ascii=False) + "\n")

    lines = [f"# atlm_teacher — output validation\n",
             f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
             f"Sample: {n} postings ({non_skip} processed, {skip} skipped), "
             f"concurrency {WORKERS}.\n",
             "## Summary\n",
             "| metric | value |", "|---|---|",
             f"| OK | {ok}/{n} |", f"| SKIP | {skip}/{n} |",
             f"| FAIL | {fail}/{n} |", f"| call errors | {errors}/{n} |",
             f"| leak suspects | {leaks}/{n} |",
             f"| median latency | {med:.1f}s |", ""]
    lines.append("## Samples\n")
    for i, r in enumerate(results, 1):
        ev, p = r["eval"], r["parsed"]
        lines.append(f"### {i}. {r['source']} / {r['tag']} — **{ev['verdict']}**"
                     f"  ({r['latency']:.1f}s)\n")
        if r["error"]:
            lines.append(f"**call error:** `{r['error']}`\n")
        excerpt = f"{r['title']}\n\n{r['body']}"[:600]
        lines.append("**Input** (excerpt):\n")
        lines.append("```\n" + excerpt + "\n```\n")
        if p["is_skip"]:
            lines.append(f"**SKIPPED:** {p['skip_reason']}\n")
        else:
            lines.append("**Parsed queries:**\n")
            for j, qq in enumerate(p["queries"], 1):
                lines.append(f"{j}. {qq}")
            lines.append("\n**Parsed job description:**\n")
            lines.append("```markdown\n" + (p["jd"] or "(empty)") + "\n```\n")
        if ev["issues"]:
            lines.append(f"**issues:** {', '.join(ev['issues'])}\n")
        if ev["leaks"]:
            lines.append(f"**leak suspects:** {', '.join(ev['leaks'])}\n")
        lines.append("---\n")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport : {OUT_MD}")
    print(f"raw    : {OUT_JSONL}")


if __name__ == "__main__":
    main()
