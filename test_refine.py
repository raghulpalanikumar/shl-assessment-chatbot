import sys
import json
sys.path.append("d:/Shl")
from backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

messages = [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
]

print("TESTING REFINEMENT...")
response = client.post("/chat", json={"messages": messages})
res_json = response.json()
print(json.dumps(res_json, indent=2))
