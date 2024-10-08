import streamlit as st
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from datasets import load_dataset
import pandas as pd
from pymongo import MongoClient
from tqdm.auto import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.vectorstores import VectorStoreRetriever
from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate, RunConfig
import nest_asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity, answer_correctness
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from colorama import Fore, Back, Style
from colorama import init as colorama_init

LOAD_DB = False
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MONGODB_ATLAS_CLUSTER_URI = st.secrets["MONGODB_ATLAS_CLUSTER_URI"]
NOMIC_API_KEY = st.secrets["NOMIC_API_KEY"]
EVAL_EMBEDDING_MODELS = ["text-embedding-ada-002", "text-embedding-3-small"] # "text-embedding-3-large"
EVAL_LLM_MODELS = ["gpt-3.5-turbo", "gpt-4o-mini","gpt-4o"] # "gpt-3.5-turbo-1106"
DB_NAME = "ragas_evals"
colorama_init(autoreset=True)
openai_client = OpenAI()

def split_texts(text_splitter, texts: List[str]) -> List[str]:
    """
    Split large texts into chunks

    Args:
        texts (List[str]): List of reference texts

    Returns:
        List[str]: List of chunked texts
    """
    chunked_texts = []
    for text in texts:
        chunks = text_splitter.create_documents([text])
        chunked_texts.extend([chunk.page_content for chunk in chunks])
    return chunked_texts

def get_embeddings(openai_client, docs: List[str], model: str) -> List[List[float]]:
    """
    Generate embeddings using the OpenAI API.

    Args:
        docs (List[str]): List of texts to embed
        model (str, optional): Model name. Defaults to "text-embedding-3-large".

    Returns:
        List[float]: Array of embeddings
    """
    # replace newlines, which can negatively affect performance.
    docs = [doc.replace("\n", " ") for doc in docs]
    response = openai_client.embeddings.create(input=docs, model=model)
    response = [r.embedding for r in response.data]
    return response

def get_retriever(model: str, k: int) -> VectorStoreRetriever:
    """
    Given an embedding model and top k, get a vector store retriever object

    Args:
        model (str): Embedding model to use
        k (int): Number of results to retrieve

    Returns:
        VectorStoreRetriever: A vector store retriever object
    """
    embeddings = OpenAIEmbeddings(model=model)

    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=MONGODB_ATLAS_CLUSTER_URI,
        namespace=f"{DB_NAME}.{model}",
        embedding=embeddings,
        index_name="vector_index",
        text_key="text",
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    return retriever

def get_rag_chain(retriever: VectorStoreRetriever, model: str) -> RunnableSequence:
    """
    Create a basic RAG chain

    Args:
        retriever (VectorStoreRetriever): Vector store retriever object
        model (str): Chat completion model to use

    Returns:
        RunnableSequence: A RAG chain
    """
    # Generate context using the retriever, and pass the user question through
    retrieve = {
        "context": retriever
        | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough(),
    }
    template = """Answer the question based only on the following context: \
    {context}

    Question: {question}
    """
    # Defining the chat prompt
    prompt = ChatPromptTemplate.from_template(template)
    # Defining the model to be used for chat completion
    llm = ChatOpenAI(temperature=0, model=model)
    # Parse output as a string
    parse_output = StrOutputParser()

    # Naive RAG chain
    rag_chain = retrieve | prompt | llm | parse_output
    return rag_chain

def main():
    
    # Load the dataset
    #data = load_dataset("explodinggradients/ragas-wikiqa", split="train")
    def string_to_list(text):
        return text.split("\"")
    # I had to add a converter for the "context" field because it was being read as a string instead of a list
    data = load_dataset("csv", data_files="./test_data/ragas_test_data.csv", split="train", converters={"context": string_to_list})
        
    df = pd.DataFrame(data)
    
    # Split text by tokens using the tiktoken tokenizer
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", keep_separator=False, chunk_size=200, chunk_overlap=30
    )
    
    # Split the context field into chunks
    df["chunks"] = df["context"].apply(lambda x: split_texts(text_splitter, x))
    
    # Aggregate list of all chunks
    all_chunks = df["chunks"].tolist()
    docs = [item for chunk in all_chunks for item in chunk]

    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    db = client[DB_NAME]
    batch_size = 128
    
    if LOAD_DB:
        for model in EVAL_EMBEDDING_MODELS:
            embedded_docs = []
            print(f"Getting embeddings for the {model} model")
            for i in tqdm(range(0, len(docs), batch_size), colour="blue", desc=f"{Fore.BLUE}Processing documents{Style.RESET_ALL}"):
                end = min(len(docs), i + batch_size)
                batch = docs[i:end]
                # Generate embeddings for current batch
                batch_embeddings = get_embeddings(openai_client, batch, model)
                # Creating the documents to ingest into MongoDB for current batch
                batch_embedded_docs = [
                    {"text": batch[i], "embedding": batch_embeddings[i]}
                    for i in range(len(batch))
                ]
                embedded_docs.extend(batch_embedded_docs)
            print(f"Finished getting embeddings for the {model} model")

            # Bulk insert documents into a MongoDB collection
            print(f"Inserting embeddings for the {model} model")
            collection = db[model]
            collection.delete_many({})
            collection.insert_many(embedded_docs)
            print(f"Finished inserting embeddings for the {model} model")
    
    # Allow nested use of asyncio (used by RAGAS)
    nest_asyncio.apply()
    
    QUESTIONS = df["question"].to_list()
    GROUND_TRUTH = df["correct_answer"].tolist()
 
    # ----------------- Embeddings Evaluation -----------------
    # NOTE: Before you execute the following code, make sure to create the Atlas vector indexs in the collections
    print("\n\nEvaluating Embeddings models")
    for model in EVAL_EMBEDDING_MODELS:
        data = {"question": [], "ground_truth": [], "contexts": []}
        data["question"] = QUESTIONS
        data["ground_truth"] = GROUND_TRUTH

        retriever = get_retriever(model, 2)
        # Getting relevant documents for the evaluation dataset
        for question in tqdm(QUESTIONS,colour="blue", desc=f"{Fore.BLUE}Processing questions{Style.RESET_ALL}"):
            data["contexts"].append(
                [doc.page_content for doc in retriever.invoke(question)]
            )
        # I had to manually create the features for 'contexts' because the dataset was not being read correctly
        features = Features({
            'contexts': Sequence(Value('string')),
            'question': Value('string'),
            'ground_truth': Value('string'),
        })
        dataset = Dataset.from_dict(data, features=features)
        # RAGAS runtime settings to avoid hitting OpenAI rate limits
        run_config = RunConfig(max_workers=4, max_wait=180)
        result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall],
            run_config=run_config,
            raise_exceptions=False,
        )
        print(f"Result for the {Fore.GREEN}{model} model:{Style.RESET_ALL} {result}")

    # ----------------- LLM Evaluation -----------------
    print("\n\nEvaluating LLM models")
    for model in EVAL_LLM_MODELS:
        print(f"Getting results for the {Fore.GREEN}{model} model{Style.RESET_ALL}")
        data = {"question": [], "ground_truth": [], "contexts": [], "answer": []}
        data["question"] = QUESTIONS
        data["ground_truth"] = GROUND_TRUTH
        # Using the best embedding model from the retriever evaluation
        retriever = get_retriever("text-embedding-3-small", 2)
        rag_chain = get_rag_chain(retriever, model)
        for question in tqdm(QUESTIONS, colour="yellow", desc=f"{Fore.BLUE}Processing questions{Style.RESET_ALL}"):
            data["answer"].append(rag_chain.invoke(question))
            data["contexts"].append(
                [doc.page_content for doc in retriever.invoke(question)]
            )
        # RAGAS expects a Dataset object
        
        features = Features({
            'contexts': Sequence(Value('string')),
            'question': Value('string'),
            'ground_truth': Value('string'),
            'answer': Value('string'),
        })
        dataset = Dataset.from_dict(data, features=features)
        #dataset = Dataset.from_dict(data)
        # RAGAS runtime settings to avoid hitting OpenAI rate limits
        run_config = RunConfig(max_workers=4, max_wait=180)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, answer_similarity, answer_correctness],
            run_config=run_config,
            raise_exceptions=False,
        )
        print(f"Result for the {Fore.GREEN}{model} model{Style.RESET_ALL}: {result}")
        # Analyze the results with pandas DataFrame
        result_df = result.to_pandas()
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            result_df.set_index("question")[["answer_similarity", "answer_correctness"]], 
            annot=True, 
            cmap="flare" #"viridis"
        )
        plt.title(f"Results for {model}")
        plt.savefig(f"./test_data/{model}_results_{datetime.now()}.png")
        result["timestamp"] = datetime.now()
        collection = db["metrics"]
        collection.insert_one(result)
    
    return

if __name__ == "__main__":
    main()