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
DOCUMENTS_PATH = "C:\\Users\\andrew.dilley\\development\\chatbot3\\documents"
FILES = ["Acceptable.docx", "ai.docx", "data.docx", "Remote.docx", "security.docx"]

# Load and process documents
def extract_text_from_word(file_path):
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

texts = [extract_text_from_word(os.path.join(DOCUMENTS_PATH, file)) for file in FILES]

# Generate embeddings and create FAISS index
dimension = 1536  # Dimensionality of OpenAI's text-embedding-ada-002
index = faiss.IndexFlatL2(dimension)
text_map = []

def generate_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    # Correctly access the embedding using dot notation
    return response.data[0].embedding

for i, text in enumerate(texts):
    embedding = generate_embeddings(text)
    index.add(np.array([embedding]).astype('float32'))
    text_map.append(text)

# Search for relevant text in FAISS index
def search_relevant_text(query):
    query_embedding = np.array([generate_embeddings(query)]).astype('float32')
    distances, indices = index.search(query_embedding, k=1)
    return text_map[indices[0][0]]

# Generate chatbot response based on relevant text
def generate_response(user_input):
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
        return completion.choices[0].message.content
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
