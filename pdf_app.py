
import streamlit as st
import pandas as pd
import traceback
import user_code as uc

st.set_page_config(page_title="Document RAG App", page_icon="📘", layout="wide")

st.title("📘RAG - Question Answering App")

# Old code
#with st.sidebar:
#    st.header("⚙️ Options")
#    st.write("Upload a PDF to build vector DB (only once needed).")

#    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
#    if uploaded_pdf:
#        with open("uploaded.pdf", "wb") as f:
#            f.write(uploaded_pdf.read())
#        with st.spinner("Building vector DB..."):
#            uc.build_vector_store("uploaded.pdf")
#        st.success("Vector DB created successfully!")


with st.sidebar:
    st.header("⚙️ Options")
    st.write("Upload a PDF or Word document to build vector DB (only once needed).")

    uploaded_file = st.file_uploader("Upload File", type=["pdf", "docx"])
    if uploaded_file:
        # file extension check
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".pdf"):
            save_path = "uploaded.pdf"
        elif file_name.endswith(".docx"):
            save_path = "uploaded.docx"
        else:
            st.error("Unsupported file type.")
            save_path = None

        if save_path:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            
            with st.spinner("Building vector DB..."):
                result = uc.build_vector_store(save_path)
                # uc.build_vector_store(save_path)

            if result is None:
                st.warning("⚠️ This file (or another document) has already been vectorized. Using existing Vector DB.")
            else:
                st.session_state["uploaded_file_path"] = save_path  # 🔥 yahan save kar diya
                st.success("Vector DB created successfully!")
            
            

# Tabs
# tab1, tab2, tab3 = st.tabs(["🔤 Ask a Question", "📄 CSV Batch", "📝 Summarize"])
tab1, tab2 = st.tabs(["🔤 Ask a Question", "📄 CSV Batch"])

with tab1:
    st.subheader("Please ask a question from the provided document")
    question = st.text_area("Enter your question:", height=120, key="qa_question")
    if st.button("Get Answer", type="primary"):
        try:
            # answer = uc.predict(question)
            answer = uc.summarize_document(question)
            st.session_state["last_question"] = question  # save for tab3
            st.write("### Answer")
            st.success(answer)
        except Exception:
            st.error("Error running prediction.")
            st.code(traceback.format_exc())


with tab2:
    st.subheader("Batch QA using CSV")
    st.caption("Upload CSV with a column named **input**, **question**, or **query**")

    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview:", df.head())

        # flexible validation
        if not any(col in df.columns for col in ["input", "question", "query"]):
            st.error("❌ Invalid CSV format. Please upload a file with a column named 'input', 'question', or 'query'.")
        else:
            if st.button("Run Batch", type="primary"):
                try:
                    out_df = uc.predict_batch(df)
                    st.write(out_df.head())
                    st.download_button("Download results", out_df.to_csv(index=False), "results.csv")
                except Exception:
                    st.error("Batch processing failed.")
                    st.code(traceback.format_exc())


#with tab2:
#    st.subheader("Batch QA using CSV")
#    st.caption("Upload CSV with a column named **input**")

#    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
#    if uploaded_csv:
#        df = pd.read_csv(uploaded_csv)
#        st.write("Preview:", df.head())

#        # ✅ Validation check
#        if "input" not in df.columns:
#            st.error("❌ Invalid CSV format. Please upload a file with a column named 'input'.")
#        else:
#            if st.button("Run Batch", type="primary"):
#                try:
#                    out_df = uc.predict_batch(df)
#                    st.write(out_df.head())
#                    st.download_button("Download results", out_df.to_csv(index=False), "results.csv")
#                except Exception:
#                    st.error("Batch processing failed.")
#                    st.code(traceback.format_exc())



#with tab2:
#    st.subheader("Batch QA using CSV")
#    st.caption("Upload CSV with a column named **input**")
#    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
#    if uploaded_csv:
#        df = pd.read_csv(uploaded_csv)
#        st.write("Preview:", df.head())
#        if st.button("Run Batch", type="primary"):
#            try:
#                out_df = uc.predict_batch(df)
#                st.write(out_df.head())
#                st.download_button("Download results", out_df.to_csv(index=False), "results.csv")
#            except Exception:
#                st.error("Batch processing failed.")
#                st.code(traceback.format_exc())


# with tab3:
#    st.subheader("Summarize Hamlet")
#    if st.button("Summarize Document", type="primary"):
#        try:
#            if "last_question" in st.session_state and "uploaded_file_path" in st.session_state:
#                summary = uc.summarize_document(
#                    st.session_state["last_question"],
#                    st.session_state["uploaded_file_path"]
#                )
#                st.success(summary)
#            else:
#                st.warning("Please upload a file and ask a question in Tab 1 first.")
#        except Exception:
#            st.error("Summarization failed.")
#            st.code(traceback.format_exc())
