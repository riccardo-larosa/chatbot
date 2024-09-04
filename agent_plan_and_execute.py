#https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_openai import ChatOpenAI
from openai import OpenAI

from langgraph.prebuilt import create_react_agent
from langsmith.wrappers import wrap_openai
import operator
from typing import Annotated, List, Tuple, TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from typing import Union, Literal
from langgraph.graph import StateGraph, START, END
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
TAVILY_API_KEY=st.secrets["TAVILY_API_KEY"]
LANGCHAIN_API_KEY=st.secrets["LANGCHAIN_API_KEY"]
print(LANGCHAIN_API_KEY)
LANGCHAIN_PROJECT="Plan-and-execute"
print(LANGCHAIN_PROJECT)
#LANGCHAIN_TRACING_V2=True
#print(LANGCHAIN_TRACING_V2)
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
print(LANGCHAIN_ENDPOINT)
tools = [TavilySearchResults(max_results=3)]
# Get the prompt to use - you can modify this!
prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-4-turbo-preview")
agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)

#result = agent_executor.invoke({"messages": [("user", "who is the winnner of the us open")]})
#print(result)

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)

planner.invoke(
    {
        "messages": [
            ("user", "what is the hometown of the current Australia open winner?")
        ]
    }
)

class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)
#from langchain_core.runnables import RunnableConfig
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    # Ensure agent_response["messages"][-1].content is a list
    agent_content = agent_response["messages"][-1].content
    if not isinstance(agent_content, list):
        agent_content = [agent_content]  # Convert to list if it's not
    return {
        "past_steps": [task] + agent_content,
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"
    
from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

from IPython.display import Image, display
import os
image_data = Image(app.get_graph().draw_png())
filename = os.path.splitext(os.path.basename(__file__))[0]
with open(f"{filename}.png", "wb") as f:
    f.write(image_data.data)

config = {"recursion_limit": 50}
inputs = {"input": "what is the hometown of the 2024 Australia open winner?"}
import asyncio
async def run_workflow():
    print("Running workflow")
    async for event in app.astream(inputs, config=config):
        print(f"Event:{event}")
        for k, v in event.items():
            if k != "__end__":
                print(v)

try: 
    asyncio.run(run_workflow())
finally:
    #wait_for_all_tracers()
    print("Tracers are done")