import os
import requests
from retriever import get_relevant_texts

# LMStudio configuration
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("LM_MODEL_NAME", "Mistral-7B-Instruct-v0.2")

def generate_answer(prompt: str) -> str:
    """
    Sends the prompt to the locally running LMStudio model and returns the response.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2
    }
    
    response = requests.post(f"{LMSTUDIO_URL}/v1/generate", json=payload)
    
    if response.status_code != 200:
        raise RuntimeError(f"LMStudio request failed: {response.text}")
    
    data = response.json()
    return data['results'][0]['text']

def main():
    user_query = input("Enter your question: ")
    
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = get_relevant_texts(user_query, top_k=5)
    
    if not retrieved_chunks:
        print("No relevant context found!")
        return
    
    # Step 2: Combine context and query into a single prompt
    context_str = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""
Based on the following context about food safety and nutrition, please provide a clear and concise answer to the question.

Context:
{context_str}

Question: {user_query}
"""
    # Step 3: Generate answer from LLM
    answer = generate_answer(prompt)
    
    # Step 4: Display results
    print("\n=== Retrieved Context ===")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"{i}. {chunk[:300]}...")  # print first 300 chars
    
    print("\n=== Generated Answer ===")
    print(answer)

if __name__ == "__main__":
    main()