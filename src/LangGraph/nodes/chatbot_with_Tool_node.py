from src.LangGraph.state.state import State
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class ChatbotWithToolNode:
    """
    Chatbot logic Enhanced with tool integration.
    """
    def __init__(self, model):
        self.llm = model
    
    def create_chatbot(self, tools):
        """
        Creates a chatbot node with bind tools.
        """
        chatbot_pronpt = ChatPromptTemplate.from_messages(
            [
            ("system", 
            "You are a helpful, honest, and professional assistant with access to tools. "
            "If a tool is used, always base your response strictly on the information the tool returns. "
            "Do not make up any extra information or speculate. "
            "Present the answer in a clear, confident, and professional tone."),
            MessagesPlaceholder(variable_name="messages")
            ]
        )
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):
            """
            Chatbot logic for processing the input state and returning a response.
            """
            chatbot_runable = chatbot_pronpt | llm_with_tools
            return {"messages": [chatbot_runable.invoke(state['messages'])]}
        
        return chatbot_node