#!/usr/bin/env bash
# Build the FP8/FP16 accuracy corpora from real, held-out text.
set -uo pipefail
C=/home/sg/acc/corpora
mkdir -p "$C"
cd "$C"

py() { /home/sg/sglang-v100-venv/bin/python "$@"; }

# ---- English: a Wikipedia article (real prose, unrelated to this project)
for T in "Electric power distribution" "Transformer" "Optical fiber"; do
  curl -sS -m 60 "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&redirects=1&titles=$(printf %s "$T" | sed 's/ /%20/g')" \
    -o "wiki_en_$(echo "$T" | tr ' ' '_').json" 2>/dev/null
done
py - <<'PY'
import glob, json, os
os.chdir("/home/sg/acc/corpora")
parts = []
for f in sorted(glob.glob("wiki_en_*.json")):
    try:
        d = json.load(open(f))
        for p in d["query"]["pages"].values():
            if "extract" in p and len(p["extract"]) > 5000:
                parts.append(p["extract"])
    except Exception as e:
        print("  skip", f, e)
txt = "\n\n".join(parts)
open("en_full.txt", "w").write(txt)
print("  en_full.txt: %d chars" % len(txt))
PY

# ---- Chinese: Chinese Wikipedia, fall back to a local Chinese doc
for T in "电力系统" "配电网" "光纤"; do
  curl -sS -m 60 "https://zh.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&redirects=1&titles=$(printf %s "$T" | python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.stdin.read().strip()))')" \
    -o "wiki_zh_$(echo "$T" | tr -d '\n').json" 2>/dev/null
done
py - <<'PY'
import glob, json, os
os.chdir("/home/sg/acc/corpora")
parts = []
for f in sorted(glob.glob("wiki_zh_*.json")):
    try:
        d = json.load(open(f))
        for p in d["query"]["pages"].values():
            if "extract" in p and len(p["extract"]) > 3000:
                parts.append(p["extract"])
    except Exception as e:
        print("  skip", f, e)
txt = "\n\n".join(parts)
open("zh_full.txt", "w").write(txt)
print("  zh_full.txt: %d chars" % len(txt))
PY

# ---- Long document: real SGLang source, deterministic order, many distinct files
py - <<'PY'
import glob, os
files = sorted(glob.glob("/home/sg/sglang-sxm2/python/sglang/srt/**/*.py", recursive=True))
out, total, used = [], 0, 0
for f in files:
    if "__pycache__" in f or os.path.getsize(f) < 4096:
        continue
    try:
        s = open(f, encoding="utf8", errors="ignore").read()
    except Exception:
        continue
    out.append("\n# ==== FILE: %s ====\n" % os.path.relpath(f, "/home/sg/sglang-sxm2"))
    out.append(s)
    total += len(s); used += 1
    if total > 1_700_000:
        break
txt = "".join(out)
open("/home/sg/acc/corpora/long.txt", "w").write(txt)
print("  long.txt: %d chars from %d files" % (len(txt), used))
PY

# ---- Fixed-size short corpora for T1
py - <<'PY'
import os
os.chdir("/home/sg/acc/corpora")
en = open("en_full.txt").read()
zh = open("zh_full.txt").read()
if len(zh) < 8000:                       # fall back to a local Chinese document
    for cand in ("/home/sg/SGLANG-GLM53-HANDOFF.md", "/home/sg/GLM53-SM70-FP8KV-DESIGN.md"):
        if os.path.exists(cand):
            zh += "\n\n" + open(cand, encoding="utf8").read()
long_ = open("long.txt").read()
code_start = long_.index("# ==== FILE:", 400000)
open("en.txt", "w").write(en[:16000])
open("zh.txt", "w").write(zh[:7000])
open("code.txt", "w").write(long_[code_start:code_start + 14000])
for n in ("en.txt", "zh.txt", "code.txt", "long.txt"):
    print("  %-10s %8d chars" % (n, os.path.getsize(n)))
PY
ls -la "$C"
