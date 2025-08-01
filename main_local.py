import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from tqdm import tqdm

load_dotenv()
PDF_PATH = "pdfs/sample.pdf"
FAISS_INDEX_PATH = "vectorstore/faiss_index_yollama"

# Loads a PDF file and splits its text into chunks.
def load_and_splitpdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, chunk_overlap = 50, add_start_index=True
    )
    all_splits = []
    # Use tqdm to show progress bar
    # No tqdm version: 
    # all_splits = text_splitter.split_documents(docs)
    for doc in tqdm(docs, desc="Splitting PDF into chunks"):
        all_splits.extend(text_splitter.split_documents([doc]))

    return all_splits

# Creates or loads a FAISS vectorstore from the text chunks.
def create_or_load_vectorstore(all_splits):
    if os.path.exists(FAISS_INDEX_PATH):
        print("🔁 Loading existing FAISS index...")
        # Set allow_dangerous_deserialization to True when you trust the source
        return FAISS.load_local(FAISS_INDEX_PATH, 
                                OllamaEmbeddings(model="nomic-embed-text:latest"), 
                                allow_dangerous_deserialization=True)
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        vectorstore = FAISS.from_documents(all_splits, embeddings)

        # This line is just for progress display, and does not embed the documents.
        # It only iterates over the documents to show how many there are. 
        # It does not actually show the progress of the embedding process itself
        _ = list(tqdm(all_splits, desc="Embedding Docs"))

        vectorstore.save_local(FAISS_INDEX_PATH)
        return vectorstore

# Creates a QA chain from the vectorstore.
def built_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    # Set model to yollama:latest, you may change it.
    llm = ChatOllama(model="yollama:latest", base_url="http://localhost:11434")

    template = """
    You are a Document Q&A assistant called Yollama. You are created by Yiding. Answer the following question concisely.
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Using the pipe operator (|) to compose a runnable sequence
    # where the output of prompt.invoke() becomes the input to llm.invoke()
    llm = prompt | llm

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        chain_type = "stuff"
    )
    return qa

if __name__ == "__main__":
    print("📄 PDF Q&A Bot Initialized.")
    print("🔍 Using file:", PDF_PATH)

    docs = load_and_splitpdf(PDF_PATH)
    vectorstore = create_or_load_vectorstore(docs)
    qa_chain = built_qa_chain(vectorstore)

    print("✅ Ready. Type your question about the PDF (type 'exit' to quit).")

    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        # The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. 
        # Use :meth:`~invoke` instead.
        # result = qa_chain.run(query)
        result = qa_chain.invoke(query)
        print("DocQA:", result["result"])