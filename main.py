from openai import OpenAI
import PyPDF2
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import transformers
import torch

load_dotenv()

api_key = os.getenv('api_key')
client = OpenAI(api_key=api_key) 

def set_open_params(model = 'gpt-3.5-turbo-0125', temperature = 0.6, max_tokens = 256, top_p = 1, frequency_penalty = 0, presence_penalty =0,):
    openai_params = {}
    openai_params['model'] = model
    openai_params['temperature'] = temperature
    openai_params['max_tokens'] = max_tokens
    openai_params['top_p'] = top_p
    openai_params['frequency_penalty'] = frequency_penalty
    openai_params['presence_penalty'] = presence_penalty
    return openai_params

def get_completion(params, messages):
    response = client.chat.completions.create(
        model = params['model'],
         messages = messages,
        temperature = params['temperature'],
        max_tokens = params['max_tokens'],
        top_p = params['top_p'],
        frequency_penalty = params['frequency_penalty'],
        presence_penalty = params['presence_penalty'],
    )
    return response.choices[0].message.content


def create_chunks(string, chunk_size = 1000, overlap = 200):
    chunks  = []
    start_index = 0
    end_index = chunk_size

    while start_index < len(string):
        chunk = string[start_index:end_index]
        chunks.append(chunk)
        start_index += chunk_size - overlap
        end_index = min(start_index + chunk_size, len(string))
    
    return chunks


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   embedding_result = client.embeddings.create(input = text, model=model, encoding_format="float").data[0].embedding
   return embedding_result


def search_result(embeddings, query_embedding, n=10):
    """
    Find the indices of the top n embeddings most similar to the query embedding.
    """   
    # Normalize each query embedding and compute cosine similarity
    similarities = [cosine_similarity(query_emb / np.linalg.norm(query_emb), emb) for query_emb, emb in zip(query_embedding, embeddings)]
    
    # Get indices of top n most similar embeddings
    sorted_indices = np.argsort(similarities)[::-1][:n]
    return sorted_indices

print("Loading the PDF file")
pdf_text = ''
with open(r'C:\Users\coditas\Downloads\c4611_sample_explain.pdf', 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)

    num_of_pages = len(pdf_reader.pages)

    for page_num in range(num_of_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()

        pdf_text += text

print("Creating Chunks......")
chunks = create_chunks(pdf_text)

print("Creating Embeddings......")
embeddings = []
for chunk in chunks:
    embedding = get_embedding(chunk)
    embeddings.append(embedding)

print("RAG model created sucessfully.")
while True:

    user_prompt = input("Enter your query(press 0 to exit):- \n")
    if user_prompt == '0':
        print("Quitting...")
        break
    query_embedding = get_embedding(user_prompt)

    user_prompt_embed_np = np.array(query_embedding).reshape(1,-1)
    similarity_matrix = cosine_similarity(embeddings, user_prompt_embed_np)

    chunk_idx = np.argmax(similarity_matrix)

    context = chunks[chunk_idx]

    messages = [
    {
        "role": "system",
        "content": f"""Provide answer to query of the user from the given context:
        {context}.
         If you don't have sufficient information reply with 'I don't know'."""
    },
    {
        "role": "user",
        "content": user_prompt
    }
    ]

    params = set_open_params()
    response = get_completion(messages= messages, params= params)

    print(response)