import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import chromadb
chroma_client = chromadb.PersistentClient(path="../data")
embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct",
                                           model_kwargs={'device': 'cpu'},
                                           encode_kwargs={'normalize_embeddings': False})


def register_collection(collection_name):
    # append to new line collection_name as string to COLLECTIONS.txt file
    with open("COLLECTIONS.txt", "a") as f:
        f.write(collection_name + "\n")


def create_vector_db(docs, model_name, collection_name):
    vector_store = Chroma(
    collection_name="e5largecollection",
    embedding_function=embedding_function,
    persist_directory="..data/chroma_langchain_db",  # Where to save data locally, remove if not neccesary
        )

    # num_ids = collection.count()
    num_docs = len(docs)
    print("=======================")
    vector_store.add_documents(documents=docs)

    return vector_store


def load_local_db(collection_name):
    vector_store = Chroma(
    collection_name="e5largecollection",
    embedding_function=embedding_function,
    persist_directory="..data/chroma_langchain_db",  # Where to save data locally, remove if not neccesary
        )
    return vector_store


