import streamlit as st
import pandas as pd

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

        tp = ((df['__true'] == True) & (df['__predicted'] == True)).sum()
        fp = ((df['__true'] == False) & (df['__predicted'] == True)).sum()
        fn = ((df['__true'] == True) & (df['__predicted'] == False)).sum()
        tn = ((df['__true'] == False) & (df['__predicted'] == False)).sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        st.subheader("Classification Results Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy*100:.2f}%")
        col2.metric("Precision", f"{precision*100:.2f}%")
        col3.metric("Recall", f"{recall*100:.2f}%")
        col4.metric("F1 Score", f"{f1*100:.2f}%")

        st.markdown(f"**True Positives**: {tp} | **False Positives**: {fp} | **False Negatives**: {fn} | **True Negatives**: {tn}")

        if fp > 0:
            st.error("False Positives (Incorrectly Classified as Positive)")
            st.write(df.loc[~df['__true'] & df['__predicted'], text_column])

        if fn > 0:
            st.warning("False Negatives (Missed Positive Cases)")
            st.write(df.loc[df['__true'] & ~df['__predicted'], text_column])

        st.subheader("Step 4: Keyword Impact Analysis")
        st.markdown("Analyze keywords by different metrics to find the optimal set for your classification needs")

        keyword_analysis = []
        for kw in keywords:
            matched = df[text_column].astype(str).str.lower().str.contains(kw)
            tp_k = ((df['__true'] == True) & matched).sum()
            fp_k = ((df['__true'] == False) & matched).sum()
            fn_k = ((df['__true'] == True) & ~matched).sum()

            kw_precision = tp_k / (tp_k + fp_k) if tp_k + fp_k > 0 else 0.0
            kw_recall = tp_k / (tp_k + fn_k) if tp_k + fn_k > 0 else 0.0
            kw_f1 = 2 * kw_precision * kw_recall / (kw_precision + kw_recall) if kw_precision + kw_recall > 0 else 0.0

            keyword_analysis.append({
                'Keyword': kw,
                'True Positives': tp_k,
                'False Positives': fp_k,
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
