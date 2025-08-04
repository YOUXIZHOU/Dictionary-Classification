from __future__ import annotations
import json
import re
import unicodedata
import csv
from typing import Dict, Set
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
   "personal_milestone": {
       "promotion", "new job", "graduated", "retired", "married", "baby", "new house", "milestone"
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

def classify(df: pd.DataFrame, target_col: str, dictionaries: Dict[str, Set[str]]) -> pd.DataFrame:
   out = df.copy()
   out["labels"] = [[] for _ in range(len(out))]

   for cat in dictionaries:
       out[f"has_{cat}"] = 0
       out[f"found_{cat}_terms"] = [[] for _ in range(len(out))]

   for idx, text in out[target_col].items():
       norm_text = normalize(str(text))
       for cat, terms in dictionaries.items():
           matched_terms = [t for t in terms if contains_term(norm_text, t)]
           if matched_terms:
               out.at[idx, f"has_{cat}"] = 1
               out.at[idx, f"found_{cat}_terms"] = matched_terms
               out.at[idx, "labels"].append(cat)
   return out

def sniff_delimiter(file) -> str:
   try:
       file.seek(0)
       sample = file.read(2048).decode("utf-8", errors="ignore")
       return csv.Sniffer().sniff(sample).delimiter
   except Exception:
       return ','

# ---------------------------------------------------------------------------
# ------------------------------- App layout --------------------------------
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dictionary‑based Text Classifier", page_icon="📄", layout="wide")
st.title("📄 Dictionary Classifier Creation")
st.markdown(
   """
Upload a CSV file and choose a text column to analyze using dictionaries you define.\n
After classification, you’ll see the matched categories, terms, and can download full results.
"""
)

# Sidebar – dictionary editor
st.sidebar.header("🔧 Dictionaries")
if "dictionaries" not in st.session_state:
   st.session_state["dictionaries"] = DEFAULT_DICTIONARIES.copy()

raw_dict_json = st.sidebar.text_area(
   "Edit dictionaries as JSON (category → list of terms)",
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

# File upload
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

       df_input.columns = [col.lstrip('\ufeff') for col in df_input.columns]

   except Exception as e:
       st.error(f"❌ Failed to parse CSV: {e}")
       st.stop()

   st.subheader("🔍 Input preview")
   st.dataframe(df_input.head(10), use_container_width=True)

   target_column = st.selectbox("🎯 Select the column to analyze", df_input.columns)

   if st.button("🚀 Run Anaylsis"):
       with st.spinner("Analyzing…"):
           df_out = classify(df_input, target_column, st.session_state["dictionaries"])
           st.session_state["df_out"] = df_out
           st.session_state["target_column"] = target_column
       st.success("Done!")
       st.subheader("📊 Results (first 20 rows)")
       st.dataframe(df_out.head(20), use_container_width=True)

# -------------------- Post-classification display -------------------------
if "df_out" in st.session_state:
   df_out = st.session_state["df_out"]

   st.header("📊 4. Analysis Results")

   st.subheader("📊 Category Analysis")
   category_data = []
   total_posts = len(df_out)
   for cat in st.session_state["dictionaries"].keys():
       count = df_out[f"has_{cat}"].sum()
       percent = f"{(count / total_posts * 100):.1f}%"
       category_data.append({"Category": cat.replace("_", " ").title(), "Posts": count, "Percentage": percent})
   category_df = pd.DataFrame(category_data)

   # Filter words to only dictionary-defined terms
   all_dict_terms = set()
   for term_set in st.session_state["dictionaries"].values():
       all_dict_terms.update(t.lower() for t in term_set)

   all_text = " ".join(df_out[st.session_state["target_column"]].astype(str).tolist())
   words = normalize(all_text).split()
   filtered_words = [w for w in words if w in all_dict_terms]
   top_keywords = Counter(filtered_words).most_common(10)
   keywords_df = pd.DataFrame(top_keywords, columns=["Keyword", "Frequency"])

   col1, col2 = st.columns(2)
   with col1:
       st.markdown("**Category Frequency:**")
       st.dataframe(category_df, use_container_width=True)
   with col2:
       st.markdown("**Top Keywords Overall:**")
       st.dataframe(keywords_df, use_container_width=True)

   st.header("📝 Sample Tagged Posts")
   category_options = list(st.session_state["dictionaries"].keys())
   selected_category = st.selectbox("Select category to view sample posts:", category_options, format_func=lambda x: x.replace("_", " ").title())

   sample_posts = df_out[df_out[f"has_{selected_category}"] == 1][st.session_state["target_column"]].head(3).reset_index()
   for i, row in sample_posts.iterrows():
       st.markdown(f"**Post {row['index']}:**")
       st.text_area("", row[st.session_state["target_column"]], height=80, disabled=True)

   # Download section
   st.subheader("📥 Download Full Results")
   export_df = df_out.copy()
   export_df["labels"] = export_df["labels"].apply(json.dumps)
   for col in export_df.columns:
       if isinstance(export_df[col].iloc[0], list):
           export_df[col] = export_df[col].apply(json.dumps)
   csv_bytes = export_df.to_csv(index=False).encode("utf-8")
   st.download_button(
       label="💾 Download classified CSV with term matches",
       data=csv_bytes,
       mime="text/csv",
       file_name="classified_output_detailed.csv",
   )
else:
   st.info("👈 Upload a CSV file and run classification to begin.")
