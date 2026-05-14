import sys
import json
sys.path.append("d:/Shl")
from backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

messages = [
    {"role": "user", "content": "What is the difference between OPQ32r and GSA?"}
]

print("TESTING COMPARE...")
response = client.post("/chat", json={"messages": messages})
res_json = response.json()
print(json.dumps(res_json, indent=2))
