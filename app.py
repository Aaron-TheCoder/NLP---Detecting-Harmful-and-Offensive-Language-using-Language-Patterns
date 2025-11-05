import pandas as pd
import re
from flask import Flask, request, jsonify, send_from_directory
import pickle

# Load pre-trained model and vectorizer
with open('toxic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# ---------------------------
# Pattern-based detection (censorship)
# ---------------------------
def normalize(text: str) -> str:
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

def load_lexicon():
    # Load the lexicon from the CSV (lexicon_from_datasets.csv)
    lex_df = pd.read_csv('lexicon_from_datasets.csv')
    
    # Compile regex patterns from the lexicon terms
    patterns = []
    for _, row in lex_df.iterrows():
        term = row['term']
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        patterns.append((pattern, row.to_dict()))
    return patterns

def detect_and_censor(text: str, patterns) -> tuple[str, list]:
    norm = normalize(text)
    censored_text = list(text)
    matches = []
    for rx, meta in patterns:
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
            if meta["action"] == "censor":
                for i in range(s, e):
                    if not text[i].isspace():
                        censored_text[i] = "*"
    return "".join(censored_text), matches

def predict_with_model(text: str) -> int:
    X_input = vectorizer.transform([text])
    return model.predict(X_input)[0]  # 1 = offensive, 0 = not offensive

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, static_folder='static')

# Load lexicon (patterns)
patterns = load_lexicon()

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/api/censor")
def api_censor():
    data = request.get_json(force=True)
    text = data.get("text", "")
    
    # Step 1: Pattern matching (fast detection)
    censored_text, matches = detect_and_censor(text, patterns)

    # Step 2: If no match is found, fallback to neural model prediction
    if len(matches) == 0:
        neural_pred = predict_with_model(text)
        if neural_pred == 1:
            matches.append({
                "match": text, "start": 0, "end": len(text), "term": "neural_model", "category": "neural_toxic",
                "severity": 3, "action": "censor"
            })
            censored_text = "*" * len(text)  # Censor the entire text if neural model detects toxicity

    return jsonify({"input": text, "censored": censored_text, "matches": matches})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)
