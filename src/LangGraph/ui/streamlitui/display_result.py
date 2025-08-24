import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import json
from src.LangGraph.state.state import StateRAG

class DisplayResultStreamlit:
    def __init__(self,usecase,graph,user_message):
        self.usecase= usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        usecase= self.usecase
        graph = self.graph
        user_message = self.user_message
        print(user_message)
        if usecase =="Basic Chatbot":
            for event in graph.stream({'messages':("user",user_message)}):
                print(event.values())
                for value in event.values():
                    print(value['messages'])
                    with st.chat_message("user"):
                        st.write(user_message)
                    with st.chat_message("assistant"):
                        st.write(value["messages"].content)
        elif usecase =="Chatbot With WebTool":
            # Prepare the state and invoke the graph
            initial_state = {'messages': [user_message]}
            res = graph.invoke(initial_state)
            for message in res['messages']:
                if type(message) == HumanMessage:
                    with st.chat_message("user"):
                        st.write(message.content)
                elif type(message) == ToolMessage:
                    print(message.content)
                elif type(message) == AIMessage and message.content:
                    with st.chat_message("assistant"):
                        st.write(message.content)
        elif usecase == "AI News":
            frequency = self.user_message
            with st.spinner("Fetching and summarizing news..."):
                result = graph.invoke({"messages": frequency})
                try:
                    # Read the markdown file
                    AI_NEWS_PATH = f"./AINews/{frequency.lower()}_summary.md"
                    with open(AI_NEWS_PATH, "r") as file:
                        markdown_content = file.read()

                    # Display the markdown content in Streamlit
                    st.markdown(markdown_content, unsafe_allow_html=True)
                except FileNotFoundError:
                    st.error(f"News Not Generated or File not found: {AI_NEWS_PATH}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        elif usecase == "Chatbot with RAG":
            # Chuẩn bị state ban đầu với message từ người dùng
            initial_state = StateRAG(
                messages=[HumanMessage(content=user_message)]
            )

            # Gọi graph, truyền thread_id để MemorySaver nhớ được
            result_state = graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": "rag_session_1"}}
            )

            # Hiển thị message người dùng
            with st.chat_message("user"):
                st.write(user_message)

            # Hiển thị câu trả lời cuối cùng từ RAG/Tavily
            with st.chat_message("assistant"):
                st.write(result_state.answer)

                # Nếu có nguồn tài liệu từ retrieve_docs
                if result_state.retrieve_docs:
                    with st.expander("Nguồn từ RAG"):
                        for doc in result_state.retrieve_docs:
                            st.markdown(f"- {doc}")

                # Nếu có kết quả tìm kiếm từ Tavily
                if result_state.tavily_results:
                    with st.expander("Kết quả từ Tavily"):
                        for res in result_state.tavily_results:
                            st.markdown(f"- {res}")