#!/usr/bin/env python
import argparse, re, random, json, hashlib
from collections import Counter, defaultdict
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

RNG = random.Random(42)

# -------- Text cleaning helpers --------
PMID_PAT = re.compile(r'\b(PMID|PMCID|DOI|ClinicalTrials\.gov|MeSH|Trial registration|NCT\d+)\b.*$', re.I)
SPACE_PAT = re.compile(r'\s+')

def clean_sentence(s: str, max_words=25):
    if not s: return ""
    # take the first sentence-ish chunk
    s = s.replace("\n", " ").strip()
    # chop at obvious citation tails
    s = PMID_PAT.split(s)[0].strip()
    # end at first full stop if present
    if ". " in s:
        s = s.split(". ")[0].strip()
    # collapse spaces, trim
    s = SPACE_PAT.sub(" ", s).strip()
    # word limit
    toks = s.split(" ")
    s = " ".join(toks[:max_words]).strip().rstrip(",;:")
    if s and not s.endswith("."): s += "."
    return s

def canon_label(x: str):
    if not x: return "Uncertain"
    x = x.strip().lower()
    if x in {"yes", "y"}: return "Yes"
    if x in {"no", "n"}: return "No"
    # pubmedqa uses "maybe"
    if "maybe" in x or x in {"uncertain", "not sure"}: return "Uncertain"
    return "Uncertain"

def hash_q(q: str):
    return hashlib.sha256(q.strip().lower().encode()).hexdigest()

def make_text(question, label, rationale):
    label = canon_label(label)
    rat = clean_sentence(rationale, max_words=22)
    if not rat: rat = "Evidence mixed or context-dependent."
    return f"Q: {question.strip()}\nA: {label}. Rationale: {rat}"

# -------- Loaders --------
def load_pubmedqa_labeled():
    # pqa_labeled has 1k expert-labeled items, train-only
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")  # fields include final_decision/long_answer
    def map_row(r):
        return {
            "question": r["question"],
            "label": canon_label(r.get("final_decision", "")),
            "rationale_src": r.get("long_answer", "") or "",
            "source": "pqa_labeled",
        }
    return ds.map(map_row, remove_columns=ds.column_names)
    
def load_pubmedqa_artificial(sample_cap=None):
    # large, noisier; has final_decision labels too
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    if sample_cap:
        # quick deterministic sample (approx. stratified below)
        ds = ds.shuffle(seed=42).select(range(min(sample_cap, len(ds))))
    def map_row(r):
        return {
            "question": r["question"],
            "label": canon_label(r.get("final_decision", "")),
            "rationale_src": r.get("long_answer", "") or "",
            "source": "pqa_artificial",
        }
    return ds.map(map_row, remove_columns=ds.column_names)

def try_load_bioasq_yesno():
    """
    bigbio/bioasq_task_b -> keep only yes/no questions.
    Schema can vary; we defensively extract question text and yes/no label.
    If unavailable, return empty dataset.
    """
    try:
        ds = load_dataset("bigbio/bioasq_task_b", split="train")
    except Exception:
        return Dataset.from_list([])
    # Heuristic extraction
    rows = []
    for r in ds:
        # try common keys
        q = r.get("question", "") or r.get("body", "") or ""
        # possible locations for answer/type
        ans = None
        for k in ("yesno", "answer", "exact_answer"):
            v = r.get(k, None)
            if isinstance(v, str) and v.lower() in {"yes", "no"}:
                ans = v
                break
        # filter only explicit yes/no
        if q and ans in {"yes", "no"}:
            rows.append({
                "question": q,
                "label": canon_label(ans),
                "rationale_src": "From BioASQ snippet(s).",  # contexts exist but we keep rationale short
                "source": "bioasq_yesno",
            })
    return Dataset.from_list(rows)

# -------- Balance, split, export --------
def stratified_split(df, val_frac=0.1, test_frac=0.1, seed=42):
    RNG.seed(seed)
    parts = {}
    for label in sorted(df.label.unique()):
        sub = df[df.label == label]
        idx = list(sub.index)
        RNG.shuffle(idx)
        n = len(idx)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        val_idx = idx[:n_val]
        test_idx = idx[n_val:n_val+n_test]
        train_idx = idx[n_val+n_test:]
        parts.setdefault("train", []).extend(train_idx)
        parts.setdefault("validation", []).extend(val_idx)
        parts.setdefault("test", []).extend(test_idx)
    return (
        df.loc[parts["train"]].reset_index(drop=True),
        df.loc[parts["validation"]].reset_index(drop=True),
        df.loc[parts["test"]].reset_index(drop=True),
    )

def class_balance(df, per_class=None, seed=42):
    RNG.seed(seed)
    out = []
    labels = sorted(df.label.unique())
    if per_class is None:
        # choose min count among classes
        per_class = min(df[df.label == y].shape[0] for y in labels)
    for y in labels:
        sub = df[df.label == y]
        idx = list(sub.index)
        RNG.shuffle(idx)
        pick = idx[:per_class]
        out.append(sub.loc[pick])
    return pd.concat(out, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

def drop_dupe_questions(df):
    seen = set(); keep = []
    for i, q in enumerate(df.question):
        h = hash_q(q)
        if h in seen: continue
        seen.add(h); keep.append(True)
    df = df.iloc[[i for i, q in enumerate(df.question) if hash_q(q) in seen]]
    # The quick comprehension above marks all; redo properly:
    seen.clear(); mask=[]
    for q in df.question:
        h = hash_q(q); mask.append(h not in seen); seen.add(h)
    return df[mask].reset_index(drop=True)

def attach_text(df):
    texts = []
    for q, lab, rat in zip(df.question, df.label, df.rationale_src):
        texts.append(make_text(q, lab, rat))
    df["text"] = texts
    return df

def save_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "text": row["text"],
                "label": row["label"],
                "question": row["question"],
                "rationale_src": row["rationale_src"],
                "source": row["source"],
            }, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pqa_artificial_cap", type=int, default=80000,
                    help="Max examples to take from pqa_artificial (before balancing).")
    ap.add_argument("--include_bioasq", action="store_true",
                    help="Include BioASQ yes/no if available.")
    ap.add_argument("--target_per_class", type=int, default=5000,
                    help="Target number of samples per class (default: 5000, yielding ~15k total for 3 classes).")
    ap.add_argument("--drop_uncertain", action="store_true",
                    help="Drop 'Uncertain' class if it has insufficient samples (use only Yes/No).")
    ap.add_argument("--uncertain_min_threshold", type=int, default=1000,
                    help="Minimum samples needed for Uncertain class; if below, auto-drop it.")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./biomed_yesno_dataset")
    args = ap.parse_args()

    print("Loading PubMedQA labeled…")
    pqaL = load_pubmedqa_labeled()     # 1k expert-labeled
    print("Loading PubMedQA artificial…")
    pqaA = load_pubmedqa_artificial(sample_cap=args.pqa_artificial_cap)

    print("Optional BioASQ yes/no…")
    bio = try_load_bioasq_yesno() if args.include_bioasq else Dataset.from_list([])

    print("Concatenating…")
    ds = concatenate_datasets([pqaL, pqaA, bio]) if len(bio) else concatenate_datasets([pqaL, pqaA])
    pdf = pd.DataFrame(ds)

    # Clean and dedupe
    pdf["question"] = pdf["question"].astype(str).str.strip()
    pdf["rationale_src"] = pdf["rationale_src"].astype(str)
    pdf["label"] = pdf["label"].astype(str)

    # drop empties
    pdf = pdf[(pdf.question.str.len() > 0) & (pdf.label.isin(["Yes","No","Uncertain"]))].reset_index(drop=True)

    # dedupe by question
    # (eager pandas filter to keep first occurrence)
    pdf = pdf.drop_duplicates(subset=["question"], keep="first").reset_index(drop=True)

    # Attach training text
    pdf = attach_text(pdf)

    # Balance classes
    counts = Counter(pdf.label)
    print("Counts before balance:", counts)
    
    # Check if we should drop Uncertain class
    if "Uncertain" in counts:
        if args.drop_uncertain or counts["Uncertain"] < args.uncertain_min_threshold:
            print(f"Dropping 'Uncertain' class ({counts['Uncertain']} samples < {args.uncertain_min_threshold} threshold)")
            pdf = pdf[pdf.label != "Uncertain"].reset_index(drop=True)
            counts = Counter(pdf.label)
            print("Counts after dropping Uncertain:", counts)
    
    # Use target_per_class, but cap at minimum available if not enough data
    per_class = min(args.target_per_class, min(counts.values()))
    print(f"Target per class: {args.target_per_class}, actual per class: {per_class}")
    pdf_bal = class_balance(pdf, per_class=per_class, seed=args.seed)
    print("Counts after balance:", Counter(pdf_bal.label))

    # Stratified split
    train_df, val_df, test_df = stratified_split(pdf_bal, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    # Export
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    save_jsonl(train_df, f"{args.out_dir}/train.jsonl")
    save_jsonl(val_df,   f"{args.out_dir}/validation.jsonl")
    save_jsonl(test_df,  f"{args.out_dir}/test.jsonl")

    # Small manifest
    manifest = {
        "total": len(pdf_bal),
        "splits": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "class_counts": Counter(pdf_bal.label),
        "sources": dict(Counter(pdf_bal.source)),
        "format": "Q/A text with Yes|No|Uncertain and short rationale; unstructured (no chat template)."
    }
    with open(f"{args.out_dir}/MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
