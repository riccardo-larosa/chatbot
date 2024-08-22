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
from ragas.metrics import context_precision, context_recall
import nest_asyncio

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

def main():
    
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MONGODB_ATLAS_CLUSTER_URI = st.secrets["MONGODB_ATLAS_CLUSTER_URI"]
    NOMIC_API_KEY = st.secrets["NOMIC_API_KEY"]
    EMBEDDINGS_TYPE = st.secrets["EMBEDDINGS_TYPE"]

    openai_client = OpenAI()
    
    # Load the dataset
    #data = load_dataset("explodinggradients/ragas-wikiqa", split="train")
    def string_to_list(text):
        return text.split("\"")
    #data = pd.read_csv("./test_data/ragas_test_data.csv", converters={"context": string_to_list})
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

    print(len(docs))
    print(docs[10])

    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    DB_NAME = "ragas_evals"
    db = client[DB_NAME]

    batch_size = 128

    EVAL_EMBEDDING_MODELS = ["text-embedding-ada-002", "text-embedding-3-small"]
    """
    for model in EVAL_EMBEDDING_MODELS:
        embedded_docs = []
        print(f"Getting embeddings for the {model} model")
        for i in tqdm(range(0, len(docs), batch_size)):
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
    """
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
    
    QUESTIONS = df["question"].to_list()
    GROUND_TRUTH = df["correct_answer"].tolist()
    
    for model in EVAL_EMBEDDING_MODELS:
        data = {"question": [], "ground_truth": [], "contexts": []}
        data["question"] = QUESTIONS
        #print(data["question"])
        data["ground_truth"] = GROUND_TRUTH
        #print(data["ground_truth"])

        retriever = get_retriever(model, 2)
        # Getting relevant documents for the evaluation dataset
        for question in tqdm(QUESTIONS):
            data["contexts"].append(
                [doc.page_content for doc in retriever.get_relevant_documents(question)]
            )
        print(f"contexts: {data["contexts"]}")
        # RAGAS expects a Dataset object
        features = Features({
            'contexts': Sequence(Value('string')),
            'question': Value('string'),
            'ground_truth': Value('string'),
            # other features...
        })

        dataset = Dataset.from_dict(data, features=features)
        print(dataset[0])
        # RAGAS runtime settings to avoid hitting OpenAI rate limits
        run_config = RunConfig(max_workers=4, max_wait=180)
        result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall],
            run_config=run_config,
            raise_exceptions=False,
        )
        print(f"Result for the {model} model: {result}")
        
    return

if __name__ == "__main__":
    main()