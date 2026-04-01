import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

headers = {"Authorization": f"Bearer {api_key}"}
try:
    response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
    models = response.json().get("data", [])

    print("Available Free models on OpenRouter:")
    for m in models:
        if "free" in m["id"].lower():
            print("-", m["id"])
except Exception as e:
    print("Error fetching models:", e)
