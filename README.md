# 📘 RAG - Question Answering App  

This project is a **Retrieval-Augmented Generation (RAG)** application built with **Streamlit** and **LangChain**.  
It allows users to:  
- Upload a **PDF or Word document**, which is vectorized and stored in a FAISS database.  
- Ask **questions interactively** about the uploaded document.  
- Run **batch question answering** using a CSV file.  

---

## 🚀 Features  
- **Document Upload**: Supports **PDF** and **DOCX**.  
- **Vector Store**: Automatically builds a **FAISS index** for retrieval.  
- **Question Answering**:  
  - Interactive Q&A (Tab 1).  
  - Batch QA from CSV with flexible column names (`input`, `question`, or `query`).  
- **CSV Validation**: Prevents wrong file uploads by checking column names.  
- **Re-upload Protection**: Detects if a document is already vectorized.  
- **Download Results**: Export batch QA results as a CSV.  

---

## 🛠️ Tech Stack  
- **Frontend**: [Streamlit](https://streamlit.io/)  
- **Backend**: [LangChain](https://www.langchain.com/)  
- **Embeddings**: HuggingFace MiniLM (`all-MiniLM-L6-v2`)  
- **Vector Store**: FAISS  
- **LLM**: Groq (`llama3-70b-8192`)  

---

## 📂 Project Structure  
```
rag-app/
├── pdf_app.py        # Frontend (Streamlit UI)
├── user_code.py      # Backend (Vector DB, QA, Batch Processing)
├── requirements.txt  # Project dependencies
├── README.md         # Documentation
└── faiss_db/         # Local FAISS vector DB (auto-created)
```

---

## ⚙️ Installation  

1. **Clone the repository**  
```bash
git clone https://github.com/jnoorsolutions/rag-app.git
cd rag-qa-app
```

2. **Create a virtual environment**  
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**  
Create a `.env` file in the project root:  
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Usage  

Run the Streamlit app:  
```bash
streamlit run pdf_app.py
```

- **Tab 1 (Interactive QA):** Ask questions directly about the uploaded document.  
- **Tab 2 (Batch QA):** Upload a CSV with a column named `input`, `question`, or `query` for batch processing.  

---

## 📝 Example CSV Format  
```csv
input
What is the main topic of the document?
Summarize the introduction.
List the key points in section 2.
```

✔️ Supports alternative column names: `input`, `question`, or `query`.  

---

## 📦 Requirements  

See [requirements.txt](./requirements.txt)  

---

## 🔮 Future Enhancements  
- Support for multiple document uploads.  
- UI improvements with chat history.  
- Option to choose between **different embedding models**.  

---

## 👨‍💻 Author  
Developed by **Junaid Noor Siddiqui** ✨  
