from langchain import hub

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import streamlit as st
from langchain_core.tools import tool


OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
TAVILY_API_KEY=st.secrets["TAVILY_API_KEY"]
#prompt = hub.pull("wfh/llm-compiler-joiner")

tool1 = TavilySearchResults(max_results=4) #increased number of results
print(type(tool1))
print(tool1.name)

@tool
def getEPAPI():
    """Use this to find the right API endpoint for Elastic Path."""
    return "GET https://api.elasticpath.dev/api"

tools = [tool1, getEPAPI]

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model = ChatOpenAI(model="gpt-3.5-turbo")  #reduce inference cost
abot = Agent(model, tools, system=prompt)

from IPython.display import Image, display
import os
image_data = Image(abot.graph.get_graph().draw_png())
filename = os.path.splitext(os.path.basename(__file__))[0]
with open(f"{filename}.png", "wb") as f:
    f.write(image_data.data)

messages = [HumanMessage(content="What is the Elastic Path API endpoint?")]
result = abot.graph.invoke({"messages": messages})
print(result)
print(result['messages'][-1].content)