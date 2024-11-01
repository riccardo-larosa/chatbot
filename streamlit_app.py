import streamlit as st
from openai import OpenAI
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_nomic.embeddings import NomicEmbeddings
import requests


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def main():
    
    public_ip = requests.get('https://ifconfig.me').text.strip()

    # Print the public IP address
    print(f"Public IP Address: {public_ip}")

    # Show title and description.
    st.title("💬 Elastic Path Docs Chatbot")


    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MONGODB_ATLAS_CLUSTER_URI = st.secrets["MONGODB_ATLAS_CLUSTER_URI"]
    NOMIC_API_KEY = st.secrets["NOMIC_API_KEY"]
    #EMBEDDINGS_TYPE = st.secrets["EMBEDDINGS_TYPE"]
    COLLECTION_NAME = st.secrets["COLLECTION_NAME"]
    TOP_K = st.secrets["TOP_K"]
    PROMPT_BASE = ""

   

    PROMPT_TEMPLATE = """
    \n\n\033[33m--------------------------\033[0m\n\n
    You are knowledgeable about Elastic Path products. You can answer any questions about 
    Commerce Manager, 
    Product Experience Manager also known as PXM,
    Cart and Checkout,
    Promotions,
    Composer,
    Payments
    Subscriptions,
    Studio.
    {prompt_base}
    Answer the question based only on the following context:
    \n\033[33m--------------------------\033[0m\n
    {context}
    \n\033[33m--------------------------\033[0m\n
    Answer the question based on the above context: {question}
    \n\033[33m--------------------------\033[0m\n
    """

    with st.sidebar:
        #embedding_type = st.selectbox("Select a model", options=EMBEDDINGS_TYPE)
        collection = st.selectbox("Select a collection", options=COLLECTION_NAME)
        if collection == "Commerce Manager":
            #collection_name = "epdocs_openaiembeddings"
            collection_name = "epdocs_prod"
            st.write("The model you have selected will use the documentation for Commerce Manager to answer your questions.")
            PROMPT_BASE = """
            Build any of the relative links using https://elasticpath.dev as the root
            """
        else:
            collection_name = "openapi_spec"
            st.write("The model you have selected will use the OpenAPI specs for Commerce Extensions (for now) to answer your questions.")
            PROMPT_BASE = """
            Include the complete CURL commands in your response when you can.
            """
     
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create mongo connection
    #embeddings = OllamaEmbeddings(model="nomic-embed-text")
    #if "OpenAI" in embedding_type:
    #    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    #    collection_name = "epdocs_openaiembeddings"
    #else:
    #    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    #    collection_name = "epdocs_nomic-embed"

    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    db_name = "rag_db"        
    atlas_collection = client[db_name][collection_name]
    vector_search_index = "vector_index"
    # Create a MongoDBAtlasVectorSearch object
    db = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI,
        db_name + "." + collection_name,
        embeddings,
        index_name = vector_search_index
    )

    print(db, db_name, collection_name, vector_search_index)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        #st.session_state.messages = []
        welcome_msg = """
        Welcome to the Elastic Path Docs Chatbot! I can answer any questions about 
        Commerce Manager related to:
        Product Experience Manager also known as PXM,
        Cart and Checkout,
        Promotions,
        Composer,
        Payments and
        Subscriptions
        """
        st.session_state["messages"] = [ChatMessage(role="assistant", 
                                                    content=welcome_msg, 
                                                    avatar="https://www.elasticpath.com/favicons/favicon.ico")]    

    # Display the existing chat messages via `st.chat_message`.
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)
            

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input():
        

        # Store and display the current prompt.
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)
        
        results = db.similarity_search_with_score(prompt, k=TOP_K)
        print(results)

        # Generate prompt from the results
        context_text = "\n\n\033[32m--------------------------\033[0m\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt_context = prompt_template.format(prompt_base=PROMPT_BASE, context=context_text, question=prompt)
        print(prompt_context)
        sources =  [(doc.metadata.get("source", None), _score) for doc, _score in results]

        
        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant", avatar="https://www.elasticpath.com/favicons/favicon.ico"):
            stream_handler = StreamHandler(st.empty())
            model = ChatOpenAI(temperature=0.7, api_key=OPENAI_API_KEY, model="gpt-4o", streaming=True, callbacks=[stream_handler])
            response = model.invoke(prompt_context)
            with st.expander("See sources"):
                st.write("Answer based on the following sources:")
                st.write(sources)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
            print("\n\n\033[34m------- RESPONSE -------------------\033[0m\n\n" + response.content)
            print("\n\n\033[34m------- SOURCES -------------------\033[0m\n\n")
            print(sources)
        #st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
