from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

class QARetriever:
    """
    A retrieval-based QA system.
    """
    def __init__(self, store: FAISS, temperature: float = 0.0):
        self.llm = ChatOpenAI(temperature=temperature)
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="map_reduce",
            retriever=store.as_retriever()
        )

    def ask(self, question: str) -> str:
        return self.chain.run(question)