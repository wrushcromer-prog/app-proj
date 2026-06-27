import os
import streamlit as st
import chromadb
from openai import OpenAI

client = OpenAI(api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://inference.baseten.co/v1")

st.title("Rush's AI Handbook Assistant v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="policies")

question = st.text_input("Ask a question about the handbook:")

if question:
    results = collection.query(
        query_texts=[question],
        n_results=1
    )

    relevant_chunks = results["documents"][0]

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Pro",
        messages=[
            {
                "role": "system",
                "content": "Answer only using the provided handbook context. If the answer is not contained in the context, say you don't know. If the user mentions her name is Katie or says this is Katie or anything to that affect, start your response with a playful greeting like 'Omgggg hey home girl!' and then answer normally. If the user introduces himself as Reed, say 'Gihhherrrrt! Welcome, Mr. Granbarkebirkebeiner!'"
            },
            {
                "role": "user",
                "content": f"""
Relevant handbook sections:

{chr(10).join(relevant_chunks)}

Question:
{question}
"""
            }
        ]
    )

    st.write(response.choices[0].message.content)