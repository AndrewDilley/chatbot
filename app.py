"""
app.py (chatbot5)

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
1. Updated the text mapping to include document file names for referencing.
2. Enhanced `search_relevant_text` to return both text and file name.
3. Modified `generate_response` to include the policy name in the chatbot's output.
4. Improved error handling to provide more user-friendly feedback.
5. Added explanatory comments to functions for better maintainability.

How to Use:
-----------
1. Ensure OpenAI API key is set in the environment variables.
2. Place policy documents in the `documents` folder specified by `DOCUMENTS_PATH`.
3. Start the application using `python app.py`.
4. Interact with the chatbot through the web interface or API endpoint `/chat`.

Dependencies:
-------------
- Flask: Web framework for the chatbot.
- OpenAI: For embeddings and chat completions.
- FAISS: For similarity-based document retrieval.
- python-docx: To extract text from Word documents.
- dotenv: To manage environment variables.

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
DOCUMENTS_PATH = "C:\\Users\\andrew.dilley\\development\\chatbot5\\documents"
FILES = ["Acceptable.docx", "ai.docx", "data.docx", "Remote.docx", "security.docx"]

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

# def load_and_process_documents():
#     global index, text_map
#     index.reset()  # Clear FAISS index
#     text_map = []  # Clear text map

#     texts = [extract_text_from_word(os.path.join(DOCUMENTS_PATH, file)) for file in FILES]

#     for i, text in enumerate(texts):
#         embedding = generate_embeddings(text)
#         index.add(np.array([embedding]).astype('float32'))
#         text_map.append(text)



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

""" def search_relevant_text(query):
    query_embedding = np.array([generate_embeddings(query)]).astype('float32')
    distances, indices = index.search(query_embedding, k=1)
 
    matched_text = text_map[indices[0][0]]
    return matched_text
 """

def search_relevant_text(query):
    query_embedding = np.array([generate_embeddings(query)]).astype('float32')
    distances, indices = index.search(query_embedding, k=1)

    matched_text, file_name = text_map[indices[0][0]]
    return matched_text, file_name



# Generate chatbot response based on relevant text

""" def generate_response(user_input):
    try:
        relevant_text = search_relevant_text(user_input)
        prompt = f"Use the following document text to answer the question:\n\n{relevant_text}\n\nQuestion: {user_input}"

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = completion.choices[0].message.content
        return answer
    except Exception as e:
        return f"Error: {str(e)}"
 """

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
        # Add policy name reference to the response
        return f"{answer}\n\nReference: {file_name}"
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

if __name__ == '__main__':
    app.run(debug=True)
