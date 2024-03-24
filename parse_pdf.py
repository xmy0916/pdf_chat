from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS

def create_db(pdfs):
    total_parges = []
    for pdf in pdfs:
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()

        total_parges.extend(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(total_parges, embeddings)

    return db

def search_most_similarity_content(db, query):
    docs = db.similarity_search(query)
    content = "\n".join([x.page_content for x in docs])
    qa_prompt = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.----------------"
    input_text = qa_prompt+"\nContext:"+content+"\nUser question:\n"+query
    return input_text
