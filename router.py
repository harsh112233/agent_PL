from typing import TypedDict, Optional
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from agent_2 import faq_agent
from agent_1 import home_inspection_agent
import time

router_llm = ChatOpenAI(model="gpt-4o", temperature=0)

class AgentState(TypedDict):
    input: str                      # the raw input from user
    image_path: Optional[str]       # path if image is sent
    agent_response: Optional[str]   # output from whichever agent is called
    memory: Optional[list[str]]     # shared memory (just text for now)
    route: Optional[str]            # route decision made by router

router_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a stict router that decides whether to send user input to a Home Inspection Agent or a FAQ search agent or to clarify agent.\n"
     "If the input contains a problem description or an image, route to 'home_inspection_agent'.\n"
     "If the input asks for general information or recommendations, or anything related to home buying, home management and property buying (like 'plumber in London') or (which area are good for renting a house?) or (what are laws related to renting a house in london), route to 'faq_agent'.\n"
     "If it's unclear, respond with 'clarify'.\n"
     "use memory provided below to check if the user input has some previous context and it should go to FAQ search agent or home inspection agent. For example if the"
     "user was talking to home_inspection_agent and askea followup question, without image, then query should be routed to home_inspection_agent"
     "memory: {memory}"
     "Return only one of: home_inspection_agent, faq_agent, or clarify."
     "Your ONLY job is to output one of the following strings exactly:\n"
     " - home_inspection_agent\n"
     " - faq_agent\n"
     " - clarify\n"
     "Do NOT explain your decision. Do NOT include any other text. Just return one of the above values."),
    ("human", "{input}")
])

def route_decision(state: AgentState) -> dict:
    input_text = state["input"]
    image_path = state.get("image_path")
    
    # if there's an image, go straight to home inspection
    if image_path:
        return {"route": "home_inspection_agent"}

    route = (router_prompt | router_llm).invoke({
    "input": input_text,
    "memory": state.get("memory", [])
        }).content.strip().lower()
    
    # Just in case, force valid outputs
    if route not in {"home_inspection_agent", "faq_agent", "clarify"}:
        route = "clarify"
    return {"route": route}

router_node = RunnableLambda(route_decision)

def home_inspection_agent_node(state: AgentState) -> AgentState:
    result = home_inspection_agent.invoke({
        "user_input": state["input"],
        "image_path": state.get("image_path"),
        "memory": state["memory"]
    })

    state["agent_response"] = result["output"]
    if state.get("memory") is None:
        state["memory"] = []
    state["memory"].append(f"User: {state['input']}")
    state["memory"].append(f"Agent: {result['output']}")
    
    return state

def faq_agent_node(state: AgentState) -> AgentState:
    result = faq_agent.invoke({
        "user_input": state["input"],
        "memory": state["memory"] ##remeber to text it with "user_input"
    })

    state["agent_response"] = result["output"]
    if state.get("memory") is None:
        state["memory"] = []
    state["memory"].append(f"User: {state['input']}")
    state["memory"].append(f"Agent: {result['output']}")
    
    return state

def clarify_node(state: AgentState) -> AgentState:
    state["agent_response"] = (
        "Could you please clarify your request? For example, are you describing a home issue or asking for general info?"
    )
    return state

builder = StateGraph(AgentState)

# Add router
builder.add_node("router", router_node)

# Add agent nodes
builder.add_node("home_inspection_agent", home_inspection_agent_node)
builder.add_node("faq_agent", faq_agent_node)
builder.add_node("clarify", clarify_node)

# Define routes
builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    path_map={
        "home_inspection_agent": "home_inspection_agent",
        "faq_agent": "faq_agent",
        "clarify": "clarify",
    }
)

# End after agent responds or clarification is sent
builder.add_edge("home_inspection_agent", END)
builder.add_edge("faq_agent", END)
builder.add_edge("clarify", END)

# Compile
graph = builder.compile()

# if __name__ == "__main__":
#     test_input_1 = {
#         "input": "what is the issue here?",
#         "image_path": "image_3.png",
#         "memory": [],
#     }

#     # Run the graph
#     result = graph.invoke(test_input_1)
#     print(f"\n✅ Routed to: {result['route']}")  

#     final_response = result['agent_response']

#     # Then stream this final_response:
#     print("AI: ", end="", flush=True)
#     for char in final_response:
#         print(char, end="", flush=True)
#         time.sleep(0.01)  # optional delay for natural effect
        