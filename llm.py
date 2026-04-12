import os
import requests
from dotenv import load_dotenv

load_dotenv()

class GroqLLM:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env!")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"  
    
    def generate(self, query, context):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {"role": "system", "content": "Answer concisely using ONLY the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=20)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error: {str(e)}"