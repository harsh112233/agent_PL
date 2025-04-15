from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.utils.function_calling import convert_to_openai_tool
import json

load_dotenv()
search_tool = TavilySearchResults(k=3)

@tool
def web_search(query: str) -> str:
    """Search the web for the given query using Tavily."""
    return search_tool.invoke(query)

tools = [convert_to_openai_tool(web_search)]

llm = ChatOpenAI(model="gpt-4o-mini", model_kwargs={"tools":tools}, temperature=0)

class FaqQueryAgent(Runnable[dict, dict]):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs["user_input"]

        messages = [
            SystemMessage(content=( "You are a smart home-buying assistant that helps users with general questions about real estate, property laws, home services, and best practices.\n"
                "Your job is to answer questions directly using your knowledge or call the web_search tool if external information is needed (e.g., for local prices, recommendations, or laws).\n\n"
                "If you're unsure about something, use the web_search tool with an appropriate query.\n"
                "Always return answers that are well-structured, helpful, and easy to understand. Keep your answer short and crisp.\n\n"
                "Example queries:\n"
                "- 'Is buying property in London a good investment?'\n"
                "- 'What are tenant rights in the UK?'\n"
                "- 'Average plumber charges in Manchester'\n"
                "- 'Best areas to rent for families in Leeds'"
                "Use memory to understand the context of the query and respond with consideration of previous message if applicable"
                "memory: {memory}"
                )),
            HumanMessage(content= user_input)
        ]

        response = self.llm.invoke(messages)
                
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_outputs = []
            for tool_call in response.tool_calls:
                tool_func = web_search
                tool_args = tool_call['args']
                tool_result = tool_func.invoke(tool_args["query"])

                #include the tool result for LLM
                messages.append(response)
                messages.append(ToolMessage(content=json.dumps(tool_result), tool_call_id=tool_call["id"]))

            final_response = self.llm.invoke(messages)
            return {"output": final_response.content}
        
        return {"output": response.content}
    

faq_agent = FaqQueryAgent(llm=llm)

        
        










