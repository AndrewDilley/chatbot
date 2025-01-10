from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from docx import Document
from PyPDF2 import PdfReader
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import json
import faiss

app = Flask(__name__)

# Load environment variables
load_dotenv()

from config import DOCUMENTS_PATH, PREPROCESSED_PATH, SHAREPOINT_LINKS


# Load the FAISS index
index_file_path = os.path.join(PREPROCESSED_PATH, "faiss_index.bin")
index = faiss.read_index(index_file_path)

# Load text map
with open(os.path.join(PREPROCESSED_PATH, "text_map.json"), "r") as f:
    text_map = json.load(f)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


FILES = list(SHAREPOINT_LINKS.keys())


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
    return np.mean(embeddings, axis=0)  # Average embedding for simplicity


def search_relevant_text(query, similarity_threshold=0.5):

    query_embedding = np.array([generate_embeddings(query)]).astype('float32')
    distances, indices = index.search(query_embedding, k=1)

    # If the best match is below the similarity threshold, return "N/A"
    if distances[0][0] > similarity_threshold:
        return None, "N/A"

    matched_text, file_name = text_map[indices[0][0]]
    return matched_text, file_name


# Generate chatbot response based on relevant text
def generate_response(user_input):
    try:

        generic_responses = {"hi", "hello", "hey", "greetings"}
        if user_input.lower() in generic_responses:
            return "Hello! How can I assist you today?"

        relevant_text, file_name = search_relevant_text(user_input)
        
        print(f"DEBUG: Matched file_name: {file_name}")

        # If no relevant document text is found
        if file_name == "N/A":
            # Handle general queries directly
            prompt = f"You are a helpful assistant. Please answer this question directly:\n\nQuestion: {user_input}"
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content  # No reference for general queries

        # Generate a response for document-related queries
        prompt = f"Use the following document text to answer the question:\n\n{relevant_text}\n\nQuestion: {user_input}"
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = completion.choices[0].message.content

        # Add document reference only if relevant
        sharepoint_link = SHAREPOINT_LINKS.get(file_name, "#")
        if file_name and file_name != "N/A":
            formatted_answer = (
                f"{answer}<br><br>"
                f"<span style='color:purple; font-weight:bold;'>Reference:</span> "
                f"<a href='{sharepoint_link}' target='_blank'>{file_name}</a>"
            )
            return formatted_answer

        return answer

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
