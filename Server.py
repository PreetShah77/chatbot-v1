import os
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from flask import Flask, request, jsonify
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
import re
import pyttsx3
from dataclasses import dataclass, asdict
from langchain.docstore.document import Document
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
sheet_id = "1hl3uC3BmTu7GNFF_ixkCuqn4apPNcVdOA53htNwZDIY"
sheet_name = "AllStudents"
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS

app = Flask(__name__)
CORS(app)  # Initialize CORS with your Flask app

@dataclass
class Student:
    name: str
    data: dict

def get_conversational_chain1():
    prompt_template = """
    Extract the named entity from the given prompt:
    Prompt: {prompt}
    Named Entity: {named_entity}
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["prompt", "named_entity"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def extract_student_name(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain1()
    response = chain({"prompt": user_question, "named_entity": ""}, return_only_outputs=True)
    print(response)
    return response
def get_student_data(first_name):
    student_name = first_name
    url = f"https://script.google.com/macros/s/AKfycbwP3XOlI33GcQzZ1m7DWzt-CuwRy3YB8BBwGU_0lFf7KD56kUY/exec?spreadsheet=a&action=getbyname&id={sheet_id}&sheet={sheet_name}&sheetuser={student_name}&sheetuserIndex=2"
    response = requests.get(url)
    data = response.json()

    if isinstance(data, dict) and 'records' in data:
        return [Student(name=record['Name of Student'], data=record) for record in data['records']]
    else:
        return None

def get_text_chunks(student_data):
    text = str(student_data)
    splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant that can provide information about students based on the data available. The data is represented as a dictionary where keys are the information fields and values are the corresponding details.

    Given the following context and query, provide the requested information if it's available in the data. If the requested information is not available, respond with "I'm sorry, but the requested information is not available in the student's data."

    Context:
    {context}

    Query:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(prompt):
    student_name = extract_student_name(prompt)
    print(student_name['text'])
    if student_name is None:
        return "No student name detected in the query."
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    student_data = get_student_data(student_name['text'])
    if not student_data:
        return "No student found with the given name."
    elif len(student_data) > 1:
        print(student_data)
        return handle_multiple_students(student_data, prompt)
    else:
        student_dict = student_data[0]
        context = str(student_dict)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_text(context)
        documents = [Document(page_content=chunk) for chunk in chunks]
        chain = get_conversational_chain()
        response = chain({"input_documents": documents, "question": prompt}, return_only_outputs=True)
        return response['output_text']
    
def handle_multiple_students(student_data, prompt):
    print("hello")
    options = []
    for i, student in enumerate(student_data, start=1):
        name = student.name
        options.append(f"Option {i}: {name}")

    response = "Please select the student you are looking for:\n"
    for option in options:
        response += option + "\n"
    print(response)

    response += "Enter the option number: "
    print(options)
    selected_index = int(input()) - 1
    selected_student = student_data[selected_index]
    print(asdict(selected_student))
    context = asdict(selected_student)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(str(context))
    chain = get_conversational_chain()
    response = chain({"context": context, "question": prompt}, return_only_outputs=True)
    return response['output_text']


@app.route('/', methods=["GET"])
def hii():
    return "hii" , 200
@app.route('/get_student_info', methods=['POST'])
def get_student_info():
    try:
        # Verify that the 'prompt' key is present in the JSON request data
        if 'prompt' not in request.json:
            raise ValueError('Missing "prompt" key in JSON data')

        prompt = request.json['prompt']
        response = user_input(prompt)
        return jsonify({'response': response})
    except Exception as e:
        error_message = f'Error processing request: {str(e)}'
        print(error_message)  # Log the error message to console or logs
        return jsonify({'error': error_message}), 500  # Return error response with status code 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))