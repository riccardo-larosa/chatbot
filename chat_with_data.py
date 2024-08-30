import os
import yaml
import streamlit as st
import tiktoken
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi import planner
from langchain_openai import ChatOpenAI

ALLOW_DANGEROUS_REQUEST=True
OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
ACCESS_TOKEN="e0cd4841bcc7103b90e6a3d623d6a3bf8a5511b4"

def auth_headers():
    return {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# it would be good to have a way to search the description fields of the api spec
def search_api_spec(query: str) -> str:
    relevant_endpoints = []
    for endpoint, details in openapi_spec.endpoints:
        #todo: this should be a search not a string match
        if query.lower() in details["description"].lower():
            relevant_endpoints.append(f"{endpoint}: {details['description']}")
    return "\n".join(relevant_endpoints) if relevant_endpoints else "No relevant API found."


# Get API credentials.
headers = auth_headers()
print(headers) 
requests_wrapper = RequestsWrapper(headers=headers)

with open("./openapispecs/catalog/catalog_view.yaml") as f:
    raw_openapi_spec = yaml.load(f, Loader=yaml.Loader)
openapi_spec = reduce_openapi_spec(raw_openapi_spec, dereference=False)
for endpoint in openapi_spec.endpoints:
    print(endpoint[0])
    if endpoint[0] == "GET /catalog":
        print(endpoint)

enc = tiktoken.encoding_for_model("gpt-4")

def count_tokens(s):
    return len(enc.encode(s))

print(count_tokens(yaml.dump(raw_openapi_spec)))
print(count_tokens(yaml.dump(openapi_spec)))

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

agent = planner.create_openapi_agent(
    openapi_spec,
    requests_wrapper,
    llm,
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
    verbose=True,
)
user_query = (
    "show me a list of all shoes"
)
#print (agent)

response = agent.invoke(user_query)

print(response)