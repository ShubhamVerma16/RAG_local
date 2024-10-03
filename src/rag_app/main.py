import argparse
import os
import math
from typing import Optional
from pydantic import Field
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_ollama.llms import OllamaLLM
from fastapi import FastAPI, Query, File, UploadFile
from flask import Flask, request
from werkzeug.utils import secure_filename

from load_data import load_split_pdf_file, load_split_html_file, initialize_splitter
# from load_llm import load_lamma_cpp
from vector_db import create_vector_db, load_local_db
from prompts import create_prompt
from utils import read_file
# from langchain_core.messages import (
#     HumanMessage,
#     SystemMessage,
# )
st.set_page_config(page_title="Chatbot")



def fake_output(x: float):
    return "Answer to this query is 42"

ml_models = {}
db_name = {}
text_splitter = initialize_splitter(chunk_size = 1000, chunk_overlap = 100)
vector_db_model_name = "test_db"
UPLOAD_FOLDER = 'file_uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/health", methods=["GET"])
def index():
    return {"message": "The server is up and running"}


@app.route("/upload", methods=["POST"])
def upload_file(collection_name : Optional[str] = "test_collection"):
    # try:
    if 'file' not in request.files:
        return "invalid Input"
    file = request.files['file']
    print("file: ", file)
    filename = secure_filename(file.filename)
    print("filename: ", filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # contents = file.read()
        # with open(f'../data/{file.filename}', 'wb') as f:
        #     f.write(contents)
    # except Exception:
    #     return {"message": "There was an error uploading the file"}
    # finally:
    #     file.file.close()
    
    if filename.endswith('.pdf'):
        data = load_split_pdf_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), text_splitter)
    elif filename.endswith('.html'):
        data = load_split_html_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), text_splitter)
    else:
        return {"message": "Only pdf and html files are supported"}
    
    db = create_vector_db(data, vector_db_model_name, collection_name)


    return {"message": f"Successfully uploaded {file.filename}", 
            "num_splits" : len(data)}


@app.route("/query", methods=["POST"])
def query(collection_name : Optional[str] = "test_collection"):
    data = request.get_json()
    print("data: ", data)
    query = data["query"]
    collection = load_local_db("test_collection")
    retriever = collection.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "score_threshold": 0.7}
    )

    context = ""
    results = retriever.invoke(query)
    for res in results:
        context += res.page_content
    template = """
    You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer the question in the same language asked:
    Q: what all is prohibited during the use of the car?
    A: According to the provided text, the following are prohibited during the use of the car: Smoking (smoking) Eating and drinking inside the vehicle Additionally, private rides are generally not allowed, except in certain circumstances , such as when traveling between home and work to attend an out-of-town meeting.
    Q: was ist bei der Nutzung des Autos alles verboten?
    A: Gemäß dem bereitgestellten Text ist während der Nutzung des Autos Folgendes verboten:
    Rauchen
    Der Verzehr von Speisen und Getränken im Fahrzeug
    Darüber hinaus sind private Fahrten grundsätzlich nicht gestattet, außer unter bestimmten Umständen, beispielsweise bei Fahrten zwischen Wohnung und Arbeit, um an einem auswärtigen Meeting teilzunehmen.


    context: {context}

    User question: {query}

    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = OllamaLLM(model="llama3.2")
    # llm = OpenAI()
    chain = prompt | llm | StrOutputParser()


    return chain.invoke({
    "context": context,
    "query": query
    })



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)