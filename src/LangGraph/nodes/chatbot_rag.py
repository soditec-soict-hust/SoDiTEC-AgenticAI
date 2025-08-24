from src.LangGraph.state.state import StateRAG
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.LangGraph.tools.search_tool import get_tools, create_tool_node
from src.LangGraph.vectorstore.file_ingestion import get_retriever

class ChatbotWithRAG:
    def __init__(self, model, vectorstore_path, embedding_model):
        self.llm = model
        self.retriever = get_retriever(vectorstore_path, embedding_model)
        self.tools = get_tools()  # Load các external tools như Tavily

    def internal_retrieve_node(self, state: StateRAG) -> StateRAG:
        # Lấy nội dung của message cuối cùng từ Human làm query
        messages = state.get("messages", [])
        human_messages = [msg for msg in messages if hasattr(msg, 'content') and msg.__class__.__name__ == 'HumanMessage']
        query = human_messages[-1].content if human_messages else ""
        
        docs = self.retriever.invoke(query)
        state["retrieve_docs"] = [doc.page_content for doc in docs]
        return state

    def check_recall_node(self, state: StateRAG) -> StateRAG:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý AI giúp đánh giá khả năng trả lời câu hỏi."),
            ("human", "Dưới đây là thông tin truy xuất được:\n\n{docs}\n\nCâu hỏi:\n\n{question}\n\nThông tin trên có đủ để trả lời không? Trả lời 'yes' hoặc 'no' thôi.")
        ])
        chain = prompt | self.llm

        messages = state.get("messages", [])
        human_messages = [msg for msg in messages if hasattr(msg, 'content') and msg.__class__.__name__ == 'HumanMessage']
        question = human_messages[-1].content if human_messages else ""
        joined_docs = "\n\n".join(state.get("retrieve_docs", []))
        result = chain.invoke({"docs": joined_docs, "question": question})
        state["recall_check_result"] = result.content.strip().lower()
        return state

    def route_decision(self, state: StateRAG) -> str:
        if state.get("recall_check_result") == "yes":
            return "use_rag"
        return "use_tavily"

    def rag_node(self, state: StateRAG) -> StateRAG:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý AI. Dưới đây là thông tin tham khảo để trả lời câu hỏi:\n\n{context}"),
            MessagesPlaceholder(variable_name="messages")
        ])
        chain = prompt | self.llm

        context = "\n\n".join(state.get("retrieve_docs", []))
        result = chain.invoke({"context": context, "messages": state["messages"]})

        state["answer"] = result.content.strip()
        return state

    def tavily_tool_node(self, state: StateRAG) -> StateRAG:
        tool_node = create_tool_node(self.tools)
        updated_state = tool_node.invoke(state)

        # Đảm bảo cập nhật answer + kết quả tìm kiếm nếu có
        state["answer"] = updated_state.get("answer", "")
        state["tavily_results"] = updated_state.get("tavily_results", [])
        return state


    