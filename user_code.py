import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader,  Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq

# Load API keys
load_dotenv()

# Global resources
# llm = ChatGroq(model="llama3-8b-8192")  # 🔥 Groq model
llm = ChatGroq(model='llama3-70b-8192')   # 🔥 Groq model
parser = StrOutputParser()
_DB_PATH = "faiss_db/document"

# EMBED_MODEL = "D:/Projects/JSolution/LearnEasily/AI/all-MiniLM-L6-v2"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ✅ HuggingFace online model
# NOTE: local path agar use karna ho to EMBED_MODEL = "D:/Projects/JSolution/LearnEasily/AI/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# embedding = OpenAIEmbeddings()



# Prompt template
prompt = PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer ONLY from the provided context.
        If the context is insufficient, say you don't know.

        {context}
        Question: {question}
    """,
    input_variables=['context', 'question']
)


# ---------------- Document Loader ----------------
def load_document(file_path):
    """Load PDF or DOCX document"""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or DOCX.")
    return loader.load()


# ---------------- Build Vector Store ----------------
def build_vector_store(file_path: str):
    """Load PDF or Word DOCX, split, embed and save FAISS DB."""

    # 🔹 Agar vector DB already exist karta hai to wapas message chala jay ga
    if os.path.exists(_DB_PATH):
        return None  # Signal ke pehle se ban chuka hai
    
    # Loader choose based on extension
    #if file_path.lower().endswith(".pdf"):
    #    loader = PyPDFLoader(file_path)
    #elif file_path.lower().endswith(".docx"):
    #    loader = Docx2txtLoader(file_path)
    #else:
    #    raise ValueError("Unsupported file type. Please upload PDF or DOCX.")

    # load document
    docs = load_document(file_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # create vector store
    if not os.path.exists(_DB_PATH):
        os.makedirs(_DB_PATH, exist_ok=True)

    # Embed + build FAISS
    vector_store = FAISS.from_documents(split_docs, embedding)
    #os.makedirs(_DB_PATH, exist_ok=True)
    vector_store.save_local(_DB_PATH)
    return vector_store


# ---------------- Load Existing Vector Store ----------------
def load_vector_store():
    """Load existing FAISS DB."""
    if not os.path.exists(_DB_PATH):
        raise FileNotFoundError("Vector DB not found. Upload and process a PDF first.")
    return FAISS.load_local(
        _DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True
    )


# 1st tab # ---------------- Summarize / Answer ----------------
def summarize_document(question: str):
    """Query the vector DB with a question"""
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type='similarity', kwargs={'k': 5})

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | llm | parser
    result = final_chain.invoke(question)
    return result


# 2nd tab # ---------------- Batch Prediction ----------------
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Batch QA on CSV with flexible column mapping."""
    # allowed alternative names
    alternatives = ["input", "question", "query"]

    # find which column exists
    found_col = None
    for alt in alternatives:
        if alt in df.columns:
            found_col = alt
            break

    if not found_col:
        raise ValueError("CSV must contain a column named one of: 'input', 'question', or 'query'.")

    # rename to 'input' for consistency
    df = df.rename(columns={found_col: "input"}).copy()

    # run predictions
    # df["output"] = df["input"].apply(lambda q: predict(str(q)))
    df["output"] = df["input"].apply(lambda q: summarize_document(str(q)))
    return df
