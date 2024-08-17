import streamlit as st
from openai import OpenAI
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def main():
    
    # Show title and description.
    st.title("💬 Elastic Path Docs Chatbot")


    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MONGODB_ATLAS_CLUSTER_URI = st.secrets["MONGODB_ATLAS_CLUSTER_URI"]

    PROMPT_TEMPLATE = """
    You are knowledgeable about Elastic Path products. You can answer any questions about 
    Commerce Manager, 
    Product Experience Manager also known as PXM,
    Cart and Checkout,
    Promotions,
    Composer,
    Payments
    Subscriptions,
    Studio.
    Build any of the relative links using https://elasticpath.dev as the root
    Answer the question based only on the following context:
    {context}
    ---
    Answer the question based on the above context: {question}
    """

    # Create mongo connection
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    db_name = "langchain_db"
    collection_name = "epdocs"
    atlas_collection = client[db_name][collection_name]
    vector_search_index = "vector_index"
    # Create a MongoDBAtlasVectorSearch object
    db = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI,
        db_name + "." + collection_name,
        embeddings,
        index_name = vector_search_index
    )

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        #st.session_state.messages = []
        st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

    # Display the existing chat messages via `st.chat_message`.
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)
            

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input():
        

        # Store and display the current prompt.
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)
        
        results = db.similarity_search_with_score(prompt, k=5)

        # Generate prompt from the results
        context_text = "\n\n\033[32m--------------------------\033[0m\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt_context = prompt_template.format(context=context_text, question=prompt)
        print(prompt_context)
        sources =  [(doc.metadata.get("source", None), _score) for doc, _score in results]

        
        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            model = ChatOpenAI(temperature=0.7, api_key=OPENAI_API_KEY, model="gpt-4o", streaming=True, callbacks=[stream_handler])
            response = model.invoke(prompt_context)
            with st.expander("See sources"):
                st.write("Answer based on the following sources:")
                st.write(sources)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
        #st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
