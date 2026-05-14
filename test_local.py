import sys
sys.path.append("d:/Shl")
from backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
response = client.post("/chat", json={"messages": [{"role": "user", "content": "How much should I pay a Java developer?"}]})
print("RESPONSE:", response.json())
