import argparse
import csv
import io
import json
import os
import re
import string
import zipfile
from pathlib import Path
from collections import Counter
import math
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from typing import Tuple, List, Dict  # <-- Add this line


# ---------------------------
# Paths / Project structure
# ---------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
STATIC_DIR = ROOT / "static"
DATA_DIR.mkdir(exist_ok=True, parents=True)
DOCS_DIR.mkdir(exist_ok=True, parents=True)
STATIC_DIR.mkdir(exist_ok=True, parents=True)

LEXICON_PATH = DATA_DIR / "lexicon_from_datasets.csv"
EVAL_PATH = DOCS_DIR / "evaluation.json"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

# ---------------------------
# Normalization helpers
# ---------------------------
LEET_MAP = str.maketrans({
    '@': 'a', '4': 'a',
    '!': 'i', '1': 'i',
    '3': 'e',
    '0': 'o',
    '$': 's', '5': 's',
    '7': 't', '+': 't',
    '€': 'e',
    '£': 'l'
})

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = t.translate(LEET_MAP)
    # collapse 3+ repeated characters into 2 (e.g., *cooool → cool*)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = ZERO_WIDTH_RE.sub("", t)
    return t

# ---------------------------
# Zip extraction + file probing
# ---------------------------
def extract_zip(zpath: str, outdir: Path) -> None:
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(outdir)

def find_olid_files(search_dir: Path) -> Tuple[str, str]:
    """Return (train_tsv, test_tsv_or_None). We only strictly need train."""
    train_tsv = None
    test_tsv = None
    for root, _, files in os.walk(search_dir):
        for f in files:
            lf = f.lower()
            p = str(Path(root) / f)
            if lf.endswith(".tsv") and "olid" in lf and "train" in lf:
                train_tsv = p
            if lf.endswith(".tsv") and "olid" in lf and "test" in lf and "gold" not in lf:
                test_tsv = p
    return train_tsv, test_tsv

def is_jigsaw_df(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return "comment_text" in cols or "comment" in cols

def find_jigsaw_train_csv(search_dir: Path) -> str:
    """Try to detect Jigsaw train.csv by reading small samples and checking columns."""
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith(".csv") and "train" in f.lower():
                cand = str(Path(root) / f)
                try:
                    df = pd.read_csv(cand, nrows=20)
                    if is_jigsaw_df(df):
                        return cand
                except Exception:
                    pass
    # fallback: return first train.csv if present
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower() == "train.csv":
                return str(Path(root) / f)
    return ""

# ---------------------------
# Dataset loading
# ---------------------------
def load_olid_train(olid_train_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(olid_train_tsv, sep="\t", header=0, quoting=3, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    # expected columns: id, tweet, subtask_a (OFF/NOT), subtask_b, subtask_c
    if "tweet" not in df.columns or "subtask_a" not in df.columns:
        raise ValueError(f"Unexpected OLID columns: {df.columns.tolist()}")
    return df

def load_jigsaw_train(jigsaw_train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(jigsaw_train_csv)
    cols = {c.lower(): c for c in df.columns}
    # standard columns: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
    if "comment_text" not in {c.lower() for c in df.columns} and "comment" in cols:
        df = df.rename(columns={cols["comment"]: "comment_text"})
    if "comment_text" not in {c.lower() for c in df.columns}:
        raise ValueError(f"Could not find 'comment_text' in Jigsaw CSV columns: {df.columns.tolist()}")
    return df

# ---------------------------
# Lexicon mining (contrastive)
# ---------------------------
STOP = set("""
the a an and or to of for in on at is it this that you your yours i me my we our they them their be are was were am as with so but if by from not no
""".split())

TOKEN_RE = re.compile(r"\w+|[^\w\s]")

def normalize_token(t: str) -> str:
    t = t.lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = t.strip()
    return t

def extract_terms(series: pd.Series, top_k: int = 500) -> list[str]:
    cnt = Counter()
    for s in series.astype(str):
        for tok in TOKEN_RE.findall(s):
            nt = normalize_token(tok)
            if nt and nt not in STOP and len(nt) > 2:
                cnt[nt] += 1
    return [w for w, _ in cnt.most_common(top_k)]

def build_lexicon_from_datasets(olid_df: pd.DataFrame, jigsaw_df: pd.DataFrame,
                                top_olid=400, top_jigsaw=700,
                                min_count=25, logodds_thresh=2.0) -> pd.DataFrame:
    """
    Build a *contrastive* lexicon: keep tokens with strong association to OFF/toxic.
    - Uses log-odds with add-k smoothing against NOT/non-toxic pools.
    - Applies a stronger stoplist and min_count filter.
    - Only categories that we can trust to "censor" are profanity/slurs/violence; others default to "flag".
    """

    # Helper: tokenize & normalize
    def tok_norm(series: pd.Series) -> list[str]:
        tokens = []
        for s in series.astype(str):
            for t in TOKEN_RE.findall(s):
                nt = normalize_token(t)
                if nt and len(nt) > 2:
                    tokens.append(nt)
        return tokens

    # OLID: split OFF vs NOT
    olid_off = olid_df[olid_df["subtask_a"].astype(str).str.upper() == "OFF"]["tweet"]
    olid_not = olid_df[olid_df["subtask_a"].astype(str).str.upper() == "NOT"]["tweet"]
    toks_off = [w for w in tok_norm(olid_off) if w not in STOP]
    toks_not = [w for w in tok_norm(olid_not) if w not in STOP]

    from collections import Counter
    c_off = Counter(toks_off)
    c_not = Counter(toks_not)

    # Jigsaw: toxic vs non-toxic
    cols = {c.lower(): c for c in jigsaw_df.columns}
    label_names = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    label_cols = [cols[n] for n in label_names if n in cols]
    if not label_cols and "toxic" in cols:
        label_cols = [cols["toxic"]]
    j_toxic = jigsaw_df[jigsaw_df[label_cols].sum(axis=1) > 0][cols.get("comment_text","comment_text")]
    j_nontox = jigsaw_df[jigsaw_df[label_cols].sum(axis=1) == 0][cols.get("comment_text","comment_text")]

    toks_tox = [w for w in tok_norm(j_toxic) if w not in STOP]
    toks_non = [w for w in tok_norm(j_nontox) if w not in STOP]
    c_tox = Counter(toks_tox)
    c_non = Counter(toks_non)

    # merge four counters into two pooled groups (positive vs negative)
    pos = c_off + c_tox
    neg = c_not + c_non

    # compute log-odds ratio with add-k smoothing
    k = 1.0
    V = len(set(list(pos.keys()) + list(neg.keys())))
    N_pos = sum(pos.values()) + k*V
    N_neg = sum(neg.values()) + k*V

    candidates = []
    for w, fpos in pos.items():
        if fpos < min_count:
            continue
        fneg = neg.get(w, 0)
        # log-odds (simple add-k)
        p_pos = (fpos + k) / N_pos
        p_neg = (fneg + k) / N_neg
        lo = math.log(p_pos / p_neg)
        if lo >= logodds_thresh:
            candidates.append((w, fpos, fneg, lo))

    # sort by log-odds then by positive freq
    candidates.sort(key=lambda x: (x[3], x[1]), reverse=True)

    # Heuristic categorization + action policy
    rows = []
    for w, fpos, fneg, lo in candidates:
        cat = "dataset_inferred"
        sev = 2
        censorable = False

        if re.search(r"(kill|die|murder|rape|bomb|hang|shoot|gun)", w):
            cat, sev, censorable = "violence", 3, True
        elif re.search(r"(idiot|stupid|moron|dumb|fool|retard|loser|trash|pig)", w):
            cat, sev = "insult", 2
        elif re.search(r"(nigger|nigga|chink|spic|kike|paki)", w):
            cat, sev, censorable = "slur_ethnicity", 3, True
        elif re.search(r"(fag|dyke|tranny|lesbo|homo)", w):
            cat, sev, censorable = "slur_sexuality", 3, True
        elif re.search(r"(bitch|whore|slut)", w):
            cat, sev, censorable = "slur_gendered", 3, True
        elif re.search(r"(fuck|shit|asshole|bastard|dick|cunt|motherfucker)", w):
            cat, sev, censorable = "profanity", 2, True

        action = "censor" if censorable else "flag"

        rows.append({
            "term": w,
            "category": cat,
            "severity": sev,
            "action": action,
            "pattern_type": "literal",
            "regex": "",
            "locale": "en",
            "source": "OLID+Jigsaw",
            "pos_freq": fpos,
            "neg_freq": fneg,
            "log_odds": lo
        })

    # Keep a reasonable cap
    df_out = pd.DataFrame(rows).sort_values(["log_odds","pos_freq"], ascending=False).head(2000)
    return df_out

# ---------------------------
# Pattern engine
# ---------------------------
def build_patterns(lex_df: pd.DataFrame):
    patterns = []
    # word-boundaries using unicode letters/numbers; safer than \b with emojis
    boundary = r"(?<![A-Za-z0-9_]){term}(?![A-Za-z0-9_])"
    for _, row in lex_df.iterrows():
        term = str(row["term"]).lower()
        pat = re.compile(boundary.format(term=re.escape(term)))
        patterns.append((pat, row.to_dict()))
    return patterns

def normalize(text: str) -> str:
    # Normalize the text by converting to lowercase and removing leetspeak.
    LEET_MAP = str.maketrans({
        '@': 'a', '4': 'a',
        '!': 'i', '1': 'i',
        '3': 'e',
        '0': 'o',
        '$': 's', '5': 's',
        '7': 't', '+': 't',
        '€': 'e',
        '£': 'l'
    })
    text = text.lower()
    text = text.translate(LEET_MAP)
    return text

def detect_and_censor(text: str, patterns) -> [str, list]:
    norm = normalize(text)  # Normalize the input text
    censored_text = list(text)  # Convert to list for mutability
    matches = []

    # Loop through each pattern and check for matches in the normalized text
    for rx, meta in patterns:
        # Ensure that the action is always "censor"
        meta["action"] = "censor"
        
        for m in rx.finditer(norm):
            s, e = m.start(), m.end()
            span_text = text[s:e]
            matches.append({
                "match": span_text,
                "start": s,
                "end": e,
                "term": meta["term"],
                "category": meta["category"],
                "severity": meta["severity"],
                "action": meta["action"]
            })
            
            # Censor the matched word(s) by replacing them with asterisks
            if meta["action"] == "censor":
                for i in range(s, e):
                    if not text[i].isspace():  # Don't replace spaces
                        censored_text[i] = "*"

    return "".join(censored_text), matches  # Return the censored text and matches

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_olid(olid_train_tsv: str, patterns) -> Dict:
    try:
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    except Exception:
        return {"error": "scikit-learn not installed; run: pip install scikit-learn"}

    df = pd.read_csv(olid_train_tsv, sep="\t", header=0, engine="python")
    df.columns = [c.lower() for c in df.columns]
    y_true = (df["subtask_a"].astype(str).str.upper() == "OFF").astype(int).values

    y_pred = []
    for t in df["tweet"].astype(str):
        _, ms = detect_and_censor(t, patterns)
        y_pred.append(1 if len(ms) > 0 else 0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"precision": p, "recall": r, "f1": f, "accuracy": acc, "n": len(y_true)}

def evaluate_jigsaw(jigsaw_train_csv: str, patterns, sample_size: int = 20000) -> Dict:
    try:
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    except Exception:
        return {"error": "scikit-learn not installed; run: pip install scikit-learn"}

    df = pd.read_csv(jigsaw_train_csv)
    cols = {c.lower(): c for c in df.columns}
    if "comment_text" not in {c.lower() for c in df.columns} and "comment" in cols:
        df = df.rename(columns={cols["comment"]: "comment_text"})

    label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    label_cols = [cols[n] for n in label_names if n in cols]
    if not label_cols and "toxic" in cols:
        label_cols = [cols["toxic"]]
    df["target_y"] = (df[label_cols].sum(axis=1) > 0).astype(int)

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    y_true = df["target_y"].values
    y_pred = []
    for t in df["comment_text"].astype(str):
        _, ms = detect_and_censor(t, patterns)
        y_pred.append(1 if len(ms) > 0 else 0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"precision": p, "recall": r, "f1": f, "accuracy": acc, "n": len(y_true)}

# ---------------------------
# Flask app (UI)
# ---------------------------
def ensure_index_html():
    if INDEX_HTML_PATH.exists():
        return
    INDEX_HTML_PATH.write_text("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pattern Censor: OLID + Jigsaw</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; }
    .card { border: 1px solid #ddd; border-radius: 16px; padding: 1rem 1.25rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
    textarea { width: 100%; min-height: 160px; font-size: 1rem; padding: .75rem; border-radius: 12px; border: 1px solid #ccc; }
    button { padding: .75rem 1rem; border-radius: 12px; border: 0; cursor: pointer; font-weight: 600; background: black; color: white; }
    pre { background: #f7f7f7; padding: .5rem; border-radius: 8px; overflow:auto; }
    table { border-collapse: collapse; width: 100%; }
    th, td { padding: .5rem .6rem; border-bottom: 1px solid #eee; text-align: left; }
    .pill { display:inline-block; padding: .2rem .5rem; border-radius: 999px; font-size: .8rem; background:#eee; }
  </style>
</head>
<body>
  <h1>Detect & Censor Harmful/Offensive Language (OLID + Jigsaw)</h1>
  <p class="muted">Type text below and click "Censor". The lexicon was inferred from OLID & Jigsaw offensive/toxic examples.</p>
  <div class="card">
    <textarea id="input" placeholder="Try: You idiot!! This is so dumb..."></textarea>
    <div style="margin-top: .75rem;"><button onclick="run()">Censor</button></div>
    <div style="margin-top: 1rem;">
      <h3>Output</h3>
      <p><strong>Censored:</strong></p>
      <pre id="censored"></pre>
      <p><strong>Matches:</strong></p>
      <table id="matches"><thead><tr><th>Match</th><th>Category</th><th>Severity</th><th>Action</th></tr></thead><tbody></tbody></table>
    </div>
  </div>
  <script>
    async function run(){
      const text = document.getElementById('input').value;
      const res = await fetch('/api/censor', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({text})
      });
      const data = await res.json();
      document.getElementById('censored').textContent = data.censored;
      const tbody = document.querySelector('#matches tbody');
      tbody.innerHTML = '';
      data.matches.forEach(m=>{
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${m.match}</td><td><span class="pill">${m.category}</span></td><td>${m.severity}</td><td>${m.action}</td>`;
        tbody.appendChild(tr);
      });
    }
  </script>
</body>
</html>""", encoding="utf-8")

def create_app(lexicon_path: Path):
    app = Flask(__name__, static_folder=str(STATIC_DIR))

    # Load lexicon once
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon not found at {lexicon_path}. Run with --build first.")
    lex_df = pd.read_csv(lexicon_path)
    patterns = build_patterns(lex_df)

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.post("/api/censor")
    def api_censor():
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "") or ""
        censored, matches = detect_and_censor(text, patterns)
        return jsonify({"input": text, "censored": censored, "matches": matches})

    return app

# ---------------------------
# Orchestration
# ---------------------------
def build_pipeline(olid_zip: str, jigsaw_zip: str) -> None:
    # 1) Extract
    if olid_zip:
        extract_zip(olid_zip, DATA_DIR)
    if jigsaw_zip:
        extract_zip(jigsaw_zip, DATA_DIR)

    # 2) Find files
    olid_train, _ = find_olid_files(DATA_DIR)
    if not olid_train:
        raise FileNotFoundError("Could not find OLID train TSV after extraction.")
    jigsaw_train = find_jigsaw_train_csv(DATA_DIR)
    if not jigsaw_train:
        raise FileNotFoundError("Could not find Jigsaw train.csv after extraction.")

    # 3) Load
    olid_df = load_olid_train(olid_train)
    jigsaw_df = load_jigsaw_train(jigsaw_train)

    # 4) Build lexicon and write CSV
    lex_df = build_lexicon_from_datasets(olid_df, jigsaw_df, top_olid=400, top_jigsaw=700)
    lex_df.to_csv(LEXICON_PATH, index=False)
    print(f"[BUILD] Lexicon written to {LEXICON_PATH} (rows={len(lex_df)})")

def run_eval() -> None:
    # Need OLID and Jigsaw train files for evaluation
    olid_train, _ = find_olid_files(DATA_DIR)
    if not olid_train:
        print("[EVAL] Skipped: OLID train TSV not found in ./data")
        return
    jigsaw_train = find_jigsaw_train_csv(DATA_DIR)
    if not jigsaw_train:
        print("[EVAL] Skipped: Jigsaw train.csv not found in ./data")
        return
    if not LEXICON_PATH.exists():
        print("[EVAL] Skipped: Lexicon missing. Run with --build first.")
        return

    lex_df = pd.read_csv(LEXICON_PATH)
    patterns = build_patterns(lex_df)

    olid_metrics = evaluate_olid(olid_train, patterns)
    jigsaw_metrics = evaluate_jigsaw(jigsaw_train, patterns, sample_size=20000)

    EVAL_PATH.write_text(json.dumps({"olid": olid_metrics, "jigsaw": jigsaw_metrics}, indent=2), encoding="utf-8")
    print(f"[EVAL] Results written to {EVAL_PATH}")
    print(json.dumps({"olid": olid_metrics, "jigsaw": jigsaw_metrics}, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Pattern Censor using OLID + Jigsaw")
    parser.add_argument("--olid_zip", type=str, default="", help="Path to OLIDv1.0.zip (uploaded)")
    parser.add_argument("--jigsaw_zip", type=str, default="", help="Path to Jigsaw zip (uploaded Kaggle mirror)")
    parser.add_argument("--build", action="store_true", help="Extract zips, mine lexicon, write CSV")
    parser.add_argument("--eval", action="store_true", help="Evaluate simple rule-based model on OLID & Jigsaw")
    parser.add_argument("--serve", action="store_true", help="Run Flask server with UI")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.build:
        if not args.olid_zip or not args.jigsaw_zip:
            print("For --build, provide both --olid_zip and --jigsaw_zip")
            return
        build_pipeline(args.olid_zip, args.jigsaw_zip)

    if args.eval:
        run_eval()

    if args.serve:
        ensure_index_html()
        app = create_app(LEXICON_PATH)
        app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
