
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
llm = ChatGroq(model='llama3-70b-8192')
parser = StrOutputParser()
#embedding = OpenAIEmbeddings()
# embedding = HuggingFaceEmbeddings(model_name="D:/Projects/JSolution/LearnEasily/AI/all-MiniLM-L6-v2")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

_DB_PATH = "faiss_db/document"

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


def build_vector_store(file_path: str):
    """Load PDF or Word DOCX, split, embed and save FAISS DB."""

    # 🔹 Agar vector DB already exist karta hai to wapas message chala jay ga
    if os.path.exists(_DB_PATH):
        return None  # Signal ke pehle se ban chuka hai
    
    # Loader choose based on extension
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload PDF or DOCX.")

    # Load docs
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # Embed + build FAISS
    vector_store = FAISS.from_documents(split_docs, embedding)
    os.makedirs(_DB_PATH, exist_ok=True)
    vector_store.save_local(_DB_PATH)
    return vector_store


# OLd code
#def build_vector_store(pdf_path: str):
#    """Load PDF, split, embed and save FAISS DB."""
#    loader = PyPDFLoader(pdf_path)
#    docs = loader.load()

#    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#    split_docs = splitter.split_documents(docs)

#    vector_store = FAISS.from_documents(split_docs, embedding)
#    os.makedirs("faiss_db", exist_ok=True)
#    vector_store.save_local(_DB_PATH)
#    return vector_store


def load_vector_store():
    """Load existing FAISS DB."""
    if not os.path.exists(_DB_PATH):
        raise FileNotFoundError("Vector DB not found. Upload and process a PDF first.")
    return FAISS.load_local(
        _DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True
    )


# 1st tab
def summarize_document(question: str):
    # """Run summarization chain on whole document."""
    """Single question answering."""
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


# 2nd tab
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





def predict(question: str):
    """Single question answering."""
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type='similarity', kwargs={'k': 3})

    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": question})

    answer = llm.invoke(final_prompt)
    return answer.content


def predict_batch_old(df: pd.DataFrame) -> pd.DataFrame:
    """Batch QA on CSV with column 'input'."""
    if "input" not in df.columns:
        raise ValueError("CSV must contain an 'input' column.")
    df = df.copy()
    df["output"] = df["input"].apply(lambda q: predict(str(q)))
    return df


def summarize_document_old(question: str, file_path: str):
    """Run summarization chain: retriever docs + predict answer as context."""
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type='similarity', kwargs={'k': 15})

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(lambda _: predict(question)),
        'question': RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | llm | parser
    result = final_chain.invoke(
        f"can you summarize the {file_path} document."
    )
    
    return result