import sys
import json
sys.path.append("d:/Shl")
from backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

messages = [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"},
    {"role": "assistant", "content": "Here are 5 assessments that fit a mid-level Java dev with stakeholder needs."},
    {"role": "user", "content": "That works. Thanks."}
]

print("TESTING CLOSURE...")
response = client.post("/chat", json={"messages": messages})
res_json = response.json()
print(json.dumps(res_json, indent=2))
