
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.schema import Document

class DataLoader:
    """
    Load PDF and DOCX documents from disk.
    """
    def __init__(self, pdf_path: str, docx_path: str):
        self.pdf_path = pdf_path
        self.docx_path = docx_path

    def load(self) -> list[Document]:
        pdf_loader = PyPDFLoader(self.pdf_path)
        docx_loader = Docx2txtLoader(self.docx_path)
        return pdf_loader.load() + docx_loader.load()

