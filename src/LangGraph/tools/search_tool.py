from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

def get_tools():
    """
    Returns the list of tools to be used in the chatbot.
    """
    tools = [TavilySearchResults(max_results=5)]
    return tools

def create_tool_node(tools):
    """
    Creates and returns a tool node for the graph.
    """
    return ToolNode(tools=tools)

    