from typing_extensions import TypedDict,List
from langgraph.graph.message import add_messages
from typing import Annotated


class State(TypedDict):
    """
    Represent the structure of the state used in graph
    """
    messages: Annotated[List,add_messages]

class StateRAG():
    messages: Annotated[List,add_messages]
    retrieve_docs: List[str] = []
    tavily_results: list[str] = []
    answer: str = ""
    recall_check_result: str = ""