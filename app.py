import os
import streamlit as st
import chromadb
from openai import OpenAI

# -----------------------------
# Configure Baseten client
# -----------------------------
client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://inference.baseten.co/v1"
)

# -----------------------------
# Streamlit page
# -----------------------------
st.title("Rush's AI Handbook Assistant v2")

# -----------------------------
# Load Chroma database
# -----------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="policies")

# -----------------------------
# Initialize chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display previous messages
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat input
# -----------------------------
question = st.chat_input("Ask a question about the handbook:")

if question:

    # Display user's message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------
    # Build retrieval query using recent conversation
    # -----------------------------
    recent_chat = "\n".join(
        [
            f'{message["role"]}: {message["content"]}'
            for message in st.session_state.messages[-6:]
        ]
    )

    search_query = f"""
Conversation so far:
{recent_chat}

Latest question:
{question}
"""

    # -----------------------------
    # Retrieve relevant handbook chunks
    # -----------------------------
    results = collection.query(
        query_texts=[search_query],
        n_results=3
    )

    relevant_chunks = results["documents"][0]

    handbook_context = "\n\n".join(relevant_chunks)

    # -----------------------------
    # Build prompt
    # -----------------------------
    messages = [
        {
            "role": "system",
            "content": f"""
You are an employee handbook assistant.

Only answer using the handbook context provided below.

If the answer is not contained in the handbook, say you don't know.

If the user introduces herself as Katie, begin your response with:
"Omgggg hey home girl!"

If the user introduces himself as Reed, begin your response with:
"Gihhherrrrt! Welcome, Mr. Granbarkebirkebeiner!"

Handbook Context:

{handbook_context}
"""
        }
    ]

    messages.extend(st.session_state.messages)

    # -----------------------------
    # Generate response
    # -----------------------------
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Pro",
        messages=messages
    )

    answer = response.choices[0].message.content

    # -----------------------------
    # Save assistant response
    # -----------------------------
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)