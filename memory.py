from langchain.memory import ConversationBufferMemory

class MemoryManager:
    """
    Manages conversation memory to retain chat history.
    """
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def get_memory(self) -> ConversationBufferMemory:
        return self.memory