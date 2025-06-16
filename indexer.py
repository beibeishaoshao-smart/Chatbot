from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

class Indexer:
    """
    Split documents into chunks and build a FAISS index.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = OpenAIEmbeddings()

    def build(self, docs: list[Document]) -> FAISS:
        chunks = self.splitter.split_documents(docs)
        return FAISS.from_documents(chunks, self.embedder)