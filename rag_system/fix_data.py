import os
import shutil
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

try:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": "nvidia/llama-nemotron-embed-vl-1b-v2:free", "input": ["Hello world"]}
    response = requests.post("https://openrouter.ai/api/v1/embeddings", headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    print("Dimension of Nemotron embeddings:", len(result['data'][0]['embedding']))
except Exception as e:
    print("API Error:", e)

# Delete the data directory which holds the previous faiss index
data_dir = "./data"
if os.path.exists(data_dir):
    try:
        shutil.rmtree(data_dir)
        print("Successfully wiped old FAISS database.")
    except Exception as e:
        print("Failed to delete ./data:", e)
else:
    print("./data directory does not exist.")
