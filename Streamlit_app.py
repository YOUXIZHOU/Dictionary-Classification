"""
Streamlit app that classifies statements in an uploaded CSV according to user‑defined dictionaries.
Features
--------
* **Upload CSV** – expects a column named `Statement.
* **Editable dictionaries** – a JSON editor in the sidebar lets users add, remove, or tweak categories/terms.
* **Run classification** – generates boolean flags and a `labels column just like the original script.
* **Download results** – returns a CSV with the new columns.
Run with: `streamlit run streamlit_dictionary_classifier.py
"""
from __future__ import annotations
import json
import re
import unicodedata
import csv
from typing import Dict, List, Set
import pandas as pd
import streamlit as st
from collections import Counter

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
   text = re.sub(r"[–—-]", " ", text)
   text = re.sub(r"[’‘]", "'", text)
   text = re.sub(r"[!?.,:;()\[\]]", " ", text)
   return text.lower()

def contains_term(text: str, term: str) -> bool:
   pattern = r"\b" + re.sub(r"\s+", r"\\s+", re.escape(term.lower())) + r"\b"
   return bool(re.search(pattern, text))

def category_matches(text: str, terms: Set[str]) -> bool:
   text = normalize(text)
   return any(contains_term(text, t) for t in terms)

def classify(df: pd.DataFrame, dictionaries: Dict[str, Set[str]]) -> pd.DataFrame:
   if "Statement" not in df.columns:
       raise KeyError("Expected a 'Statement' column in the input CSV.")
   out = df.copy()
   out["labels"] = [[] for _ in range(len(out))]
   for cat in dictionaries:
       out[cat] = False
   for idx, text in out["Statement"].items():
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
uploaded_file = st.file_uploader("📤 Upload your CSV", type=["csv"])
if uploaded_file is not None:
   try:
       delimiter = sniff_delimiter(uploaded_file)
       uploaded_file.seek(0)

       try:
           df_input = pd.read_csv(uploaded_file, encoding="utf-8", sep=delimiter)
       except UnicodeDecodeError:
           uploaded_file.seek(0)
           df_input = pd.read_csv(uploaded_file, encoding="ISO-8859-1", sep=delimiter)

       # 🔧 Remove BOM if present in column names
       df_input.columns = [col.lstrip('\ufeff') for col in df_input.columns]

       # 🧪 Show columns to debug
       st.write("📑 Detected columns:", df_input.columns.tolist())

   except Exception as e:
       st.error(f"❌ Failed to parse CSV: {e}")
       st.stop()

   st.subheader("🔍 Input preview")
   st.dataframe(df_input.head(10), use_container_width=True)

   if "Statement" not in df_input.columns:
       st.warning("The CSV must contain a **Statement** column. Upload another file.")
       st.stop()

   if st.button("🚀 Run Classification"):
       with st.spinner("Classifying…"):
           df_out = classify(df_input, st.session_state["dictionaries"])
       st.success("Done!")
       st.subheader("📊 Results (first 20 rows)")
       st.dataframe(df_out.head(20), use_container_width=True)

       # -- Analysis Summary Section -------------------------------------------
       st.header("📊 4. Analysis Results")

       st.subheader("📊 Category Analysis")
       category_data = []
       total_posts = len(df_out)
       for cat in st.session_state["dictionaries"].keys():
           count = df_out[cat].sum()
           percent = f"{(count / total_posts * 100):.1f}%"
           category_data.append({"Category": cat.replace("_", " ").title(), "Posts": count, "Percentage": percent})
       category_df = pd.DataFrame(category_data)

       # -- Top keywords
       all_text = " ".join(df_out["Statement"].astype(str).tolist())
       words = normalize(all_text).split()
       top_keywords = Counter(words).most_common(10)
       keywords_df = pd.DataFrame(top_keywords, columns=["Keyword", "Frequency"])

       col1, col2 = st.columns(2)
       with col1:
           st.markdown("**Category Frequency:**")
           st.dataframe(category_df, use_container_width=True)
       with col2:
           st.markdown("**Top Keywords Overall:**")
           st.dataframe(keywords_df, use_container_width=True)

       # -- Sample Posts Section -----------------------------------------------
       st.header("📝 Sample Tagged Posts")
       category_options = list(st.session_state["dictionaries"].keys())
       selected_category = st.selectbox("Select category to view sample posts:", category_options, format_func=lambda x: x.replace("_", " ").title())

       sample_posts = df_out[df_out[selected_category]]["Statement"].head(3).reset_index()
       for i, row in sample_posts.iterrows():
           st.markdown(f"**Post {row['index']}:**")
           st.text_area("", row["Statement"], height=80)

       # -- Download CSV Button ------------------------------------------------
       export_df = df_out.copy()
       export_df["labels"] = export_df["labels"].apply(json.dumps)
       csv_bytes = export_df.to_csv(index=False).encode("utf-8")
       st.download_button(
           label="💾 Download classified CSV",
           data=csv_bytes,
           mime="text/csv",
           file_name="classified_output.csv",
       )
else:
   st.info("👈 Upload a CSV file to begin.")
