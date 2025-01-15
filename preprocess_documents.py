import os
import json
import faiss
import numpy as np
from docx import Document
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from config import DOCUMENTS_PATH, PREPROCESSED_PATH, SHAREPOINT_LINKS

FILES = list(SHAREPOINT_LINKS.keys())


dimension = 1536  # Embedding vector size
index = faiss.IndexFlatL2(dimension)  # FAISS index
text_map = []  # To map indices to text and filenames

os.makedirs(PREPROCESSED_PATH, exist_ok=True)

def extract_text_from_word(file_path):
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text_into_chunks(text, chunk_size=8000):
    chunks = []
    while len(text) > chunk_size:
        split_index = text[:chunk_size].rfind(" ")
        chunks.append(text[:split_index])
        text = text[split_index:].strip()
    chunks.append(text)
    return chunks



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


def preprocess_documents():
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

    
    # Save index
    index_file_path = os.path.join(PREPROCESSED_PATH, "faiss_index.bin")
    faiss.write_index(index, index_file_path)

    # Save text map
    with open(os.path.join(PREPROCESSED_PATH, "text_map.json"), "w") as f:
        json.dump(text_map, f)


if __name__ == "__main__":
    preprocess_documents()
