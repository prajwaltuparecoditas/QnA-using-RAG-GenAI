from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rag_tools import set_open_params, get_completion,create_chunks, get_embedding

def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en','ja','hi',])
        text = ' '.join([t['text'] for t in transcript])
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# video_id = 'b2luyGaFHcQ'
video_id = 'gthlEJO_qnc'
transcript_text = get_youtube_transcript(video_id)

text_chunks = create_chunks(transcript_text)
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

