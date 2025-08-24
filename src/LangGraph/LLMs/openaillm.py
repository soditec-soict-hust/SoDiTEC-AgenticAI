import os
from langchain_openai import ChatOpenAI

class OpenAILLM:
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        try:
            openai_api_key = self.user_controls_input["API_KEY"]
            sellected_openai_model = self.user_controls_input["selected_openai_model"]
            if openai_api_key == '' and os.environ.get("OPENAI_API_KEY", '') == '':
                raise ValueError("Please Enter the API KEY")
            llm = ChatOpenAI(api_key=openai_api_key, model=sellected_openai_model)

        except Exception as e:
            raise ValueError(f"Error Occurred With Exception: {e}")
        return llm