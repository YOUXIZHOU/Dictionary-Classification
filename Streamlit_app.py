"""
Streamlit app that classifies statements in an uploaded CSV according to user‑defined dictionaries.
Features
--------
* **Upload CSV** – expects a column named ``Statement``.
* **Editable dictionaries** – a JSON editor in the sidebar lets users add, remove, or tweak categories/terms.
* **Run classification** – generates boolean flags and a ``labels`` column just like the original script.
* **Download results** – returns a CSV with the new columns.
Run with: ``streamlit run streamlit_dictionary_classifier.py``
"""
from __future__ import annotations
import json
import re
import unicodedata
import csv
from typing import Dict, List, Set
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# --------------------------- Default dictionaries --------------------------
# ---------------------------------------------------------------------------
DEFAULT_DICTIONARIES: Dict[str, Set[str]] = {
   "urgency_marketing": {
       "limited", "limited time", "limited run", "limited edition", "order now", "last chance", "hurry",
       "while supplies last", "before they're gone", "selling out", "selling fast", "act now",
       "don't wait", "today only", "expires soon", "final hours", "almost gone",
   },
   "exclusive_marketing": {
       "exclusive", "exclusively", "exclusive offer", "exclusive deal", "members only", "vip", "special access",
       "invitation only", "premium", "privileged", "limited access", "select customers", "insider",
       "private sale", "early access",
   },
   "personalized_service_product": {
       "custom", "monogram",
   },
}

# ---------------------------------------------------------------------------
# ------------------------------ Helper logic -------------------------------
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
   text = unicodedata.normalize("NFKD", text)
   text = re.sub(r"[\u2013\u2014-]", " ", text)
   text = re.sub(r"[\u2019\u2018`]", "'", text)
   text = re.sub(r"[!?.,:;()\[\]]", " ", text)
   return text.lower()

def contains_term(text: str, term: str) -> bool:
   pattern = r"\\b" + re.sub(r"\\s+", r"\\\\s+", re.escape(term.lower())) + r"\\b"
   return bool(re.search(pattern, text))

def category_matches(text: str, terms: Set[str]) -> bool:
   text = normalize(text)
   return any(contains_term(text, t) for t in terms)

def classify(df: pd.DataFrame, dictionaries: Dict[str, Set[str]], col_name: str = "Statement") -> pd.DataFrame:
   if col_name not in df.columns:
       raise KeyError(f"Expected a '{col_name}' column in the input CSV.")
   out = df.copy()
   out["labels"] = [[] for _ in range(len(out))]
   for cat in dictionaries:
       out[cat] = False
   for idx, text in out[col_name].items():
       matched_categories = [cat for cat, terms in dictionaries.items() if category_matches(str(text), terms)]
       for cat in matched_categories:
           out.at[idx, cat] = True
       out.at[idx, "labels"] = matched_categories
   return out

def sniff_delimiter(file) -> str:
   try:
       file.seek(0)
       sample = file.read(2048).decode("utf-8", errors="ignore")
       return csv.Sniffer().sniff(sample).delimiter
   except Exception:
       return ','  # fallback

# ---------------------------------------------------------------------------
# ------------------------------- App layout --------------------------------
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dictionary‑based Text Classifier", page_icon="📄", layout="wide")
st.title("📄 Dictionary‑based Text Classifier")
st.markdown(
   """
Upload a CSV containing a **Statement** column and specify the dictionaries that mark up your text.\
When you click **Run Classification**, new boolean columns and a **labels** list will be added.\
Download the enriched CSV with the button at the bottom.
"""
)

# -- Sidebar dictionary editor ------------------------------------------------
st.sidebar.header("🔧 Dictionaries")
if "dictionaries" not in st.session_state:
   st.session_state["dictionaries"] = DEFAULT_DICTIONARIES.copy()

raw_dict_json = st.sidebar.text_area(
   "Edit the dictionaries as JSON (category → list of terms)",
   value=json.dumps({k: sorted(list(v)) for k, v in st.session_state["dictionaries"].items()}, indent=2),
   height=400,
   key="dict_editor",
)

if st.sidebar.button("✅ Apply changes"):
   try:
       loaded = json.loads(raw_dict_json)
       st.session_state["dictionaries"] = {k: set(v) for k, v in loaded.items()}
       st.sidebar.success("Dictionaries updated ✔️")
   except json.JSONDecodeError as exc:
       st.sidebar.error(f"Invalid JSON: {exc}")

# -- Main area: file upload & preview ----------------------------------------
uploaded_file = st.file_uploader("📄 Upload your CSV", type=["csv"])
if uploaded_file is not None:
   try:
       delimiter = sniff_delimiter(uploaded_file)
       uploaded_file.seek(0)

       try:
           df_input = pd.read_csv(uploaded_file, encoding="utf-8", sep=delimiter)
       except UnicodeDecodeError:
           uploaded_file.seek(0)
           df_input = pd.read_csv(uploaded_file, encoding="ISO-8859-1", sep=delimiter)

       df_input.columns = [col.lstrip('\ufeff') for col in df_input.columns]
       st.write("📁 Detected columns:", df_input.columns.tolist())

       classification_col = st.selectbox("📝 Column to classify", df_input.columns, index=df_input.columns.get_loc("Statement") if "Statement" in df_input.columns else 0)
       ground_truth_col = st.selectbox("✅ Ground truth column (for evaluation)", df_input.columns)

   except Exception as e:
       st.error(f"❌ Failed to parse CSV: {e}")
       st.stop()

   if st.button("🚀 Run Classification"):
       with st.spinner("Classifying…"):
           df_out = classify(df_input, st.session_state["dictionaries"], col_name=classification_col)
           df_out["labels_set"] = df_out["labels"].apply(set)
           df_out["true_labels_set"] = df_out[ground_truth_col].apply(lambda x: set(eval(x)) if isinstance(x, str) and x.startswith("[") else set())

           def get_metrics(row):
               return {
                   "tp": len(row["labels_set"] & row["true_labels_set"]),
                   "fp": len(row["labels_set"] - row["true_labels_set"]),
                   "fn": len(row["true_labels_set"] - row["labels_set"]),
               }

           metrics = df_out.apply(get_metrics, axis=1).tolist()
           total_tp = sum(m["tp"] for m in metrics)
           total_fp = sum(m["fp"] for m in metrics)
           total_fn = sum(m["fn"] for m in metrics)
           total_tn = len(df_out) - sum(1 for m in metrics if m["tp"] + m["fp"] + m["fn"] > 0)

           precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
           recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
           f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
           accuracy = (total_tp + total_tn) / len(df_out)

       st.success("✅ Classification Complete")
       st.markdown("## 📊 Classification Results Summary")
       st.columns(4)[0].metric("Accuracy", f"{accuracy:.2%}")
       st.columns(4)[1].metric("Precision", f"{precision:.2%}")
       st.columns(4)[2].metric("Recall", f"{recall:.2%}")
       st.columns(4)[3].metric("F1 Score", f"{f1:.2%}")

       st.markdown(f"""
       True Positives: {total_tp} | False Positives: {total_fp} | False Negatives: {total_fn} | True Negatives: {total_tn}
       """)

       fp_rows = df_out[[len(m["labels_set"] - m["true_labels_set"]) > 0 for m in metrics]]
       fn_rows = df_out[[len(m["true_labels_set"] - m["labels_set"]) > 0 for m in metrics]]

       st.markdown("### ❌ False Positives (Incorrectly Classified as Positive)")
       if not fp_rows.empty:
           st.dataframe(fp_rows[[classification_col, "labels", ground_truth_col]])
       else:
           st.info("No false positives")

       st.markdown("### ⚠️ False Negatives (Missed Positive Cases)")
       if not fn_rows.empty:
           st.dataframe(fn_rows[[classification_col, "labels", ground_truth_col]])
       else:
           st.info("No false negatives")

       st.markdown("## 🧠 Step 4: Keyword Impact Analysis")
       all_terms = {term for terms in st.session_state["dictionaries"].values() for term in terms}
       keyword_stats = []
       for term in sorted(all_terms):
           pattern = re.compile(r"\\b" + re.sub(r"\\s+", r"\\\\s+", re.escape(term.lower())) + r"\\b")
           tp = fp = fn = 0
           for _, row in df_out.iterrows():
               normalized_text = normalize(str(row[classification_col]))
               predicted = pattern.search(normalized_text) is not None
               actual = any(term in normalize(l) for l in row[ground_truth_col]) if isinstance(row[ground_truth_col], list) else False
               if predicted and actual:
                   tp += 1
               elif predicted and not actual:
                   fp += 1
               elif not predicted and actual:
                   fn += 1
           prec = tp / (tp + fp) if (tp + fp) else 0
           rec = tp / (tp + fn) if (tp + fn) else 0
           f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
           keyword_stats.append((term, tp, fp, fn, prec, rec, f1_score))

       keyword_stats.sort(key=lambda x: x[-1], reverse=True)
       for idx, (term, tp, fp, fn, prec, rec, f1_score) in enumerate(keyword_stats, 1):
           with st.container():
               st.markdown(f"#### #{idx} `{term}`")
               st.markdown(
                   f"**Recall:** {rec:.1%} | **Precision:** {prec:.1%} | **F1:** {f1_score:.1%}  \n"
                   f"**True Positives ({tp})**, **False Positives ({fp})**, **False Negatives ({fn})**"
               )
else:
   st.info("👈 Upload a CSV file to begin.")
