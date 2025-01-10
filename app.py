from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from docx import Document
from PyPDF2 import PdfReader  # Add PyPDF2 for PDF support
import faiss
import numpy as np
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Path to documents
if os.getenv("DOCKER_ENV") == "true":
    DOCUMENTS_PATH = "/app/documents"
else:
    DOCUMENTS_PATH = "C:/Users/andrew.dilley/development/chatbot11/documents"

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
    "Wannon Water Enterprise Agreement 2020.PDF": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/908",
    "Zero Harm Policy.DOCX": "https://wannonwater.sharepoint.com/sites/cdms/SitePages/Homepage.aspx#/PublishedDocumentView/722"
    # Add other files here
}

FILES = list(SHAREPOINT_LINKS.keys())

# Initialize FAISS index and text map
dimension = 1536
index = faiss.IndexFlatL2(dimension)
text_map = []

def split_text_into_chunks(text, chunk_size=8000):
    """
    Splits text into smaller chunks to fit within the token limit.
    """
    chunks = []
    while len(text) > chunk_size:
        split_index = text[:chunk_size].rfind(" ")  # Find the last space to avoid cutting words
        chunks.append(text[:split_index])
        text = text[split_index:].strip()
    chunks.append(text)
    return chunks


# Function to extract text from Word documents
def extract_text_from_word(file_path):
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

# Function to extract text from PDF documents
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to generate embeddings using OpenAI

def generate_embeddings(text):
    """
    Generates embeddings for the given text.
    Splits the text into chunks if it exceeds the token limit.
    """
    embeddings = []
    text_chunks = split_text_into_chunks(text)
    for chunk in text_chunks:
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embeddings.append(response.data[0].embedding)
    # Optionally, return a combined embedding (e.g., average of embeddings)
    return np.mean(embeddings, axis=0)  # Average embedding for simplicity


# Function to load and process documents
def load_and_process_documents():
    global index, text_map
    index.reset()
    text_map = []

    for file in FILES:
        file_path = os.path.join(DOCUMENTS_PATH, file)
        if file.lower().endswith(".docx"):
            text = extract_text_from_word(file_path)
        elif file.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            continue

        embedding = generate_embeddings(text)
        index.add(np.array([embedding]).astype('float32'))
        text_map.append((text, file))

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
        prompt = f"Use the following document text to answer the question:\n\n{relevant_text}\n\nQuestion: {user_input}"

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = completion.choices[0].message.content

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
    return render_template('index.html')

if __name__ == '__main__':
    if os.getenv("DOCKER_ENV") == "true":
        app.run(host='0.0.0.0', port=80, debug=False)
    else:
        app.run(host='127.0.0.1', port=5000, debug=True)
