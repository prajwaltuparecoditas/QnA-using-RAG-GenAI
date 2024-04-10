import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rag_tools import set_open_params, get_completion,create_chunks, get_embedding
url = 'https://pypi.org/project/youtube-transcript-api/'

def extract_text(url):  
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        return text
    else:
        print("Failed to retrieve the webpage.")
        return None
    
url = 'https://pypi.org/project/youtube-transcript-api/'
print("Extracting ........")
webpage_text = extract_text(url)
print("Creating Chunks......")
text_chunks = create_chunks(webpage_text)
print("Creating Embeddings....")
embeddings = []
for chunk in text_chunks:
    embedding = get_embedding(chunk)
    embeddings.append(embedding)

while True:

    user_prompt = input("Enter your query(press 0 to exit):- \n")
    if user_prompt == '0':
        print("Quitting...")
        break
    query_embedding = get_embedding(user_prompt)

    user_prompt_embed_np = np.array(query_embedding).reshape(1,-1)
    similarity_matrix = cosine_similarity(embeddings, user_prompt_embed_np)

    chunk_idx = np.argmax(similarity_matrix)

    context = text_chunks[chunk_idx]

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

