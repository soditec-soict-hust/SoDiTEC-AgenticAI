from langgraph.graph import StateGraph
from src.LangGraph.state.state import State
from src.LangGraph.state.state import StateRAG
from langgraph.graph import START,END
from src.LangGraph.nodes.basic_chatbot_node import BasicChatbotNode
from src.LangGraph.tools.search_tool import get_tools, create_tool_node
from langgraph.prebuilt import tools_condition,ToolNode
from src.LangGraph.nodes.chatbot_with_Tool_node import ChatbotWithToolNode
from src.LangGraph.nodes.ai_news_node import AiNewsNode
from src.LangGraph.nodes.chatbot_rag import ChatbotWithRAG
from langgraph.checkpoint.memory import MemorySaver

class GraphBuilder:
    def __init__(self,model):
        self.llm=model
        self.graph_builder=StateGraph(State)

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """

        self.basic_chatbot_node=BasicChatbotNode(self.llm)

        # Add the basic chatbot node to the graph
        self.graph_builder.add_node("chatbot",self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot",END)

    def chatbot_with_tools_build_graph(self):
        """
        Placeholder for building a chatbot graph with tools.
        This method can be expanded to include additional nodes and edges
        for a more complex chatbot setup.
        """
        tools = get_tools()
        tool_node = create_tool_node(tools)

        # Define the LLM
        llm = self.llm

        # Define the chatbot node
        obj_chatbot_with_tool_node = ChatbotWithToolNode(llm)
        chatbot_node = obj_chatbot_with_tool_node.create_chatbot(tools)

        # Add nodes
        self.graph_builder.add_node("chatbot",chatbot_node)
        self.graph_builder.add_node("tools", tool_node)

        # Add edges
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools", "chatbot")
        self.graph_builder.add_edge("chatbot", END)

    def ai_news_builder_graph(self):

        ai_news_node = AiNewsNode(self.llm)

        # Add nodes
        self.graph_builder.add_node("fetch_news", ai_news_node.fetch_news)
        self.graph_builder.add_node("summarize_news", ai_news_node.summarize_news)
        self.graph_builder.add_node("save_result", ai_news_node.save_result)

        # Add edges
        self.graph_builder.set_entry_point("fetch_news")
        self.graph_builder.add_edge("fetch_news", "summarize_news")
        self.graph_builder.add_edge("summarize_news", "save_result")
        self.graph_builder.add_edge("save_result", END)

    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()
        elif usecase == "Chatbot With WebTool":
            self.chatbot_with_tools_build_graph()
        elif usecase == "AI News":
            self.ai_news_builder_graph()
        return self.graph_builder.compile()

class RAGGraphBuilder(GraphBuilder):
    def __init__(self,model, vectorstore_path, embedding_model):
        self.llm=model
        self.graph_builder=StateGraph(State)
        self.rag_graph_builder=StateGraph(StateRAG)
        self.vectorstore_path = vectorstore_path
        self.embedding_model = embedding_model

    def ai_chatbot_with_rag_builder_graph(self):

        self.rag_chatbot_node = ChatbotWithRAG(self.llm, self.vectorstore_path, self.embedding_model)

        self.rag_graph_builder.add_node("retrieve_internal", self.rag_chatbot_node.internal_retrieve_node)
        self.rag_graph_builder.add_node("check_recall", self.rag_chatbot_node.check_recall_node)
        self.rag_graph_builder.add_node("use_rag", self.rag_chatbot_node.rag_node)
        self.rag_graph_builder.add_node("use_tavily", self.rag_chatbot_node.tavily_tool_node)

        # self.rag_graph_builder.set_entry_point("retrieve_internal")
        self.rag_graph_builder.add_edge(START, "retrieve_internal")
        self.rag_graph_builder.set_entry_point("retrieve_internal")
        self.rag_graph_builder.add_edge("retrieve_internal", "check_recall")
        self.rag_graph_builder.add_conditional_edges(
            "check_recall", 
            self.rag_chatbot_node.route_decision, {
                "use_rag": "use_rag",
                "use_tavily": "use_tavily"
            }
        )
        self.rag_graph_builder.add_edge("use_rag", END)
        self.rag_graph_builder.add_edge("use_tavily", END)
    
    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        memory = MemorySaver()
        if usecase == "Chatbot with RAG":
            self.ai_chatbot_with_rag_builder_graph()
        return self.rag_graph_builder.compile(checkpointer=memory)