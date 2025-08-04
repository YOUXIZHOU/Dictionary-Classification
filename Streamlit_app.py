import streamlit as st
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

st.title("Keyword-Based Text Classification")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='latin1')

    st.subheader("Step 1: Choose Columns")
    text_column = st.selectbox("Select the column containing statements:", df.columns, index=df.columns.get_loc("Statement") if "Statement" in df.columns else 0)
    label_column = st.selectbox("Select the column for ground truth labels:", df.columns)

    st.subheader("Step 2: Keyword Setup")
    keyword_input = st.text_area("Enter keywords separated by commas:", "custom, customized, customization")
    keywords = [kw.strip().lower() for kw in keyword_input.split(",") if kw.strip()]

    if st.button("Run Classification"):
        df['__predicted'] = df[text_column].astype(str).str.lower().apply(
            lambda x: any(kw in x for kw in keywords))

        df['__true'] = df[label_column].astype(bool)

        precision = precision_score(df['__true'], df['__predicted'], zero_division=0)
        recall = recall_score(df['__true'], df['__predicted'], zero_division=0)
        f1 = f1_score(df['__true'], df['__predicted'], zero_division=0)
        accuracy = (df['__true'] == df['__predicted']).mean()

        st.subheader("Classification Results Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy*100:.2f}%")
        col2.metric("Precision", f"{precision*100:.2f}%")
        col3.metric("Recall", f"{recall*100:.2f}%")
        col4.metric("F1 Score", f"{f1*100:.2f}%")

        # Error breakdown
        st.markdown(f"**True Positives**: {(df['__true'] & df['__predicted']).sum()} | "
                    f"**False Positives**: {(~df['__true'] & df['__predicted']).sum()} | "
                    f"**False Negatives**: {(df['__true'] & ~df['__predicted']).sum()} | "
                    f"**True Negatives**: {(~df['__true'] & ~df['__predicted']).sum()}")

        if (~df['__true'] & df['__predicted']).any():
            st.error("False Positives (Incorrectly Classified as Positive)")
            st.write(df.loc[~df['__true'] & df['__predicted'], text_column])

        if (df['__true'] & ~df['__predicted']).any():
            st.warning("False Negatives (Missed Positive Cases)")
            st.write(df.loc[df['__true'] & ~df['__predicted'], text_column])

        st.subheader("Step 4: Keyword Impact Analysis")
        st.markdown("Analyze keywords by different metrics to find the optimal set for your classification needs")

        keyword_analysis = []
        for kw in keywords:
            matched = df[text_column].astype(str).str.lower().str.contains(kw)
            tp = ((df['__true'] == True) & matched).sum()
            fp = ((df['__true'] == False) & matched).sum()
            fn = ((df['__true'] == True) & ~matched).sum()

            kw_precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            kw_recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            kw_f1 = 2 * kw_precision * kw_recall / (kw_precision + kw_recall) if kw_precision + kw_recall > 0 else 0.0

            keyword_analysis.append({
                'Keyword': kw,
                'True Positives': tp,
                'False Positives': fp,
                'Recall': kw_recall,
                'Precision': kw_precision,
                'F1': kw_f1
            })

        sort_option = st.radio("Sort keywords by:", ["Recall", "Precision", "F1"], horizontal=True)
        keyword_analysis = sorted(keyword_analysis, key=lambda x: x[sort_option], reverse=True)

        for i, entry in enumerate(keyword_analysis, 1):
            st.markdown(f"### #{i} `{entry['Keyword']}`")
            col1, col2 = st.columns(2)
            col1.success(f"True Positives ({entry['True Positives']})")
            col2.error(f"False Positives ({entry['False Positives']})")
            st.markdown(f"**Recall:** {entry['Recall']*100:.1f}% | "
                        f"**Precision:** {entry['Precision']*100:.1f}% | "
                        f"**F1:** {entry['F1']*100:.1f}%")
