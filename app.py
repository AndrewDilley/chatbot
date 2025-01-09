"""
app.py (chatbot10)

Description:
------------
This Flask-based application implements a chatbot that uses OpenAI's GPT model
to provide intelligent responses based on user queries. The chatbot references
specific policy documents to generate relevant answers and includes the names
of the referenced policies in the response. The document data is processed
using FAISS for efficient similarity-based retrieval.

Key Features:
-------------
1. Loads and processes multiple policy documents (Word format) into a FAISS index.
2. Embeds document text using OpenAI's embedding model (text-embedding-ada-002).
3. Retrieves relevant document text based on user input using FAISS.
4. Generates intelligent responses using OpenAI's GPT model.
5. Includes the name of the relevant policy as a reference in chatbot responses.

Recent Changes:
---------------
1. Use of WW procedures and policies documents.

Setup:
------
1. Ensure OpenAI API key is configured in the `.env` file.
2. Place relevant documents in the `documents` folder.
3. Run the app locally or deploy using Docker.

Dependencies:
-------------
- Flask: Web framework for the chatbot.
- OpenAI: For embeddings and chat completions.
- FAISS: For similarity-based document retrieval.
- python-docx: To extract text from Word documents.
- dotenv: To manage environment variables.

Author:
-------
Andrew Dilley

"""

from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from docx import Document
import faiss
import numpy as np
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Path to documents
# Use a default path for local development, and override it if running in Docker

if os.getenv("DOCKER_ENV") == "true":
    DOCUMENTS_PATH = "/app/documents"  # Path inside Docker
else:
    DOCUMENTS_PATH = "C:/Users/andrew.dilley/development/chatbot10/documents"  # Path for local development

SHAREPOINT_LINKS = {
    "Alcohol and Drugs in the Workplace Procedure.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/508",
    "Consequence Of Employee Misconduct.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/286",
    "Contractor Management Procedure.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/417",
    "Cyber Security Incident Response Plan Framework.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/885",
    "Flexible Working Arrangements Procedure.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/640",
    "Gifts Benefits and Hospitality Policy - BOARD.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/822",
    "Hazard Reporting Procedure.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/293",
    "Incident Reporting and Response Procedure.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/665",
    "Information Technology Security Procedure.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/815",
    "Mobile Phone Procedure.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/896",
    "Motor Vehicle Operational Procedure.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/240",
    "Personal Protective Equipment and Field Uniform.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/230",
    "Vehicle Logbook Procedure.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/1321",
    "Physical Security Policy.docx": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/1355",
    "Use of text based Generative Artificial Intelligence (AI).DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/1373",
    "Vehicle Safety System Alarm Procedure.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/883",
    "Vehicle Safety System Manual.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/1317",
    "Zero Harm Policy.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/722"

    # Add other file names and links here
}

FILES = list(SHAREPOINT_LINKS.keys())

# Initialize FAISS index and text map
dimension = 1536  # Dimensionality of OpenAI's text-embedding-ada-002
index = faiss.IndexFlatL2(dimension)
text_map = []

# Function to extract text from Word documents
def extract_text_from_word(file_path):
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

# Function to generate embeddings using OpenAI
def generate_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Function to load and process documents


def load_and_process_documents():
    global index, text_map
    index.reset()  # Clear FAISS index
    text_map = []  # Clear text map

    for file in FILES:
        file_path = os.path.join(DOCUMENTS_PATH, file)
        text = extract_text_from_word(file_path)
        embedding = generate_embeddings(text)
        index.add(np.array([embedding]).astype('float32'))
        text_map.append((text, file))  # Store text with file name


# Load and process documents when the app starts
load_and_process_documents()

# Function to search relevant text in FAISS index


def search_relevant_text(query):
    query_embedding = np.array([generate_embeddings(query)]).astype('float32')
    distances, indices = index.search(query_embedding, k=1)

    matched_text, file_name = text_map[indices[0][0]]
    return matched_text, file_name


# Generate chatbot response based on relevant text

def generate_response(user_input):
    try:
        relevant_text, file_name = search_relevant_text(user_input)

        # Build the prompt
        prompt = f"Use the following document text to answer the question:\n\n{relevant_text}\n\nQuestion: {user_input}"

        # Generate a response from OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = completion.choices[0].message.content

        # Add hyperlink to the reference
        sharepoint_link = SHAREPOINT_LINKS.get(file_name, "#")
        formatted_answer = (
            f"{answer}<br><br>"
            f"<span style='color:purple; font-weight:bold;'>Reference:</span> "
            f"<a href='{sharepoint_link}' target='_blank'>{file_name}</a>"
        )
        return formatted_answer
    except Exception as e:
        return f"Error: {str(e)}"




# Route for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    response = generate_response(user_input)
    return jsonify({"response": response})

# Default route
@app.route('/')
def home():
    return render_template('index.html')  # Render the chatbot interface

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    # Check if the app is running in Docker
    if os.getenv("DOCKER_ENV") == "true":
        # Running inside Docker
        app.run(host='0.0.0.0', port=80, debug=False)
    else:
        # Running locally (development environment)
        app.run(host='127.0.0.1', port=5000, debug=True)
