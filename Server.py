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
import pytz
from datetime import datetime, timedelta
from dateutil import parser

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

# Add the necessary functions from the StudentInfoChatbot class
def convert_to_24_hours(time_str):
    # Convert time string to datetime object
    time_obj = datetime.strptime(time_str, "%I:%M %p")
    # Convert to 24-hour format string
    return time_obj.strftime("%H:%M")

def get_current_schedule(data, test_datetime=None):
    if data is None:
        return None, None, None, None
    # Get current Indian standard time
    ist = pytz.timezone('Asia/Kolkata')
    if test_datetime:
        now = test_datetime  # datetime.strptime(test_datetime, "%d-%m-%Y %I:%M %p").astimezone(ist)
    else:
        now = datetime.now(ist)
    today = now.strftime("%A")
    current_time = now.strftime("%I:%M %p")
    # Convert current_time to 24-hour format
    current_time_24h = convert_to_24_hours(current_time)

    # Find the schedule for today
    today_schedule = [record for record in data['records'] if record['Day/Time'] == today]

    if not today_schedule:
        return None, None, None, f"{current_time} ({today})"

    # Find the current class based on current time
    current_class = None
    for record in today_schedule:
        for key in record:
            if key != 'Day/Time':
                start_time, end_time = map(convert_to_24_hours, key.split(" to "))
                if start_time <= current_time_24h <= end_time:
                    current_class = record[key]
                    return fetch_last_value(current_class, f"{key} ({today})")
        if current_class:
            break

    return None, None, None, f"{current_time} ({today})"

def fetch_last_value(input_string, duration):
    # Check if input_string contains a comma
    if ',' not in input_string:
        return '', '', '', duration  # Return empty string if comma is not found

    # Split the input string by comma and remove any leading/trailing spaces
    values = [value.strip() for value in input_string.split(',')]
    if len(values) < 3:
        return "", "", "", duration
    # Return the last value
    return values[-1], values[0], values[1], duration

def extract_time(user_input):
    patterns = [
        r"(after|in)\s+(\d+)\s+(hour|hours)",
        r"(after|in)\s+an?\s+hour",
        r"(before)\s+(\d+)\s+(hour|hours)",
        r"(before)\s+an?\s+hour",
        r"now"
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            if "hour" in pattern:
                hours = int(match.group(2)) if match.group(2) else 1
                direction = match.group(1)
                if direction == "before":
                    return f"{hours} hours ago"
                else:
                    return f"{hours} hours from now"
            elif "now" in pattern:
                return "now"
    return None

def get_student_location_info(student_info, user_input, response):
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if user_input:
        # Extract time information from user input
        time_str = extract_time(user_input)
        print("time_str=", time_str)
        if time_str:
            try:
                if "ago" in time_str:
                    hours = int(time_str.split(" ")[0])
                    print("hours=", hours)
                    target_time = now - timedelta(hours=hours)
                elif "from now" in time_str:
                    hours = int(time_str.split(" ")[0])
                    print("hours=", hours)
                    target_time = now + timedelta(hours=hours)
                else:
                    target_time = parser.parse(time_str)

                    if target_time < now:
                        target_time = target_time.replace(year=now.year, month=now.month, day=now.day)
            except Exception as e:
                print(e)
                target_time = now
        else:
            target_time = now
    else:
        target_time = now
    # print("target_time=", target_time)
    student_location, subject, teacher, time_duration = get_current_schedule(get_student_tt_info(student_info),
                                                                             target_time)
    if time_duration is None:
        response['message'] = f"The college schedule for {student_info['Name of Student']} does not provided by authority."
    else:
        if student_location is None:
            response['message'] = f"The college schedule for {student_info['Name of Student']} does not include any activities at the specified time {time_duration}."
        elif student_location == "":
            response['message'] = f"{student_info['Name of Student']} is free at the specified time from {time_duration}."
            response['student_name'] = student_info['Name of Student']
        elif student_location == "-":
            response['message'] = f"{student_info['Name of Student']} is attending subject {subject} at the specified time from {time_duration}."
            response['student_name'] = student_info['Name of Student']
        else:
            response['message'] = f"The location information for {student_info['Name of Student']} at the specified time from {time_duration} is {student_location} for subject {subject}."
            response['student_name'] = student_info['Name of Student']
            response['student_location'] = student_location
            if teacher:
                response['message'] = f"The location information for {student_info['Name of Student']} at the specified time from {time_duration} is {student_location} for subject {subject} with teacher {teacher}."

    return response

def is_student_available(student_info):
    response = {}
    response['is_student_available'] = True
    response['student_info'] = student_info
    response['student_location'] = get_student_location_info(student_info, None, {})['message']
    return response


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

def get_student_tt_info(student_info):
    if 'Class' not in student_info:
        return None
    api_url = f"https://script.google.com/macros/s/AKfycbwP3XOlI33GcQzZ1m7DWzt-CuwRy3YB8BBwGU_0lFf7KD56kUY/exec?spreadsheet=a&action=get&id={sheet_id}&sheet={student_info['Class']}&sheetuser=all&sheetuserIndex=0"
    response = requests.get(api_url)
    data = response.json()
    return data

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
    # Check for greeting intents
    greeting_patterns = r'^(hi|hello|hey)\b'
    if re.search(greeting_patterns, prompt, re.IGNORECASE):
        return "Hello! What can I do for you?"

    # Extract student name
    student_name = extract_student_name(prompt)
    print(student_name['text'])
    if student_name is None:
        return "No student name detected in the query."

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    student_data = get_student_data(student_name['text'])
    if not student_data:
        return "No student found with the given name."
    # elif len(student_data) > 1:
    #     return handle_multiple_students(student_data, prompt)
    else:
        student_dict = student_data[0]
        context = str(student_dict)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_text(context)
        documents = [Document(page_content=chunk) for chunk in chunks]
        chain = get_conversational_chain()

        # Check if the query is related to the student's location
        if any(keyword in prompt.lower() for keyword in ['location', 'where']):
            student_info = student_dict.data
            availability_info = is_student_available(student_info)
            if availability_info['is_student_available']:
                location_response = availability_info['student_location']
                return location_response
            else:
                return "I'm sorry, but the requested location information is not available for this student."

        # If not a location query, proceed with the regular question answering
        response = chain({"input_documents": documents, "question": prompt}, return_only_outputs=True)
        return response['output_text']

def handle_multiple_students(student_data, prompt):
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
    return "hii", 200

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
