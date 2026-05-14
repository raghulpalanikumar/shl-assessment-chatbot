import urllib.request
import json

url = "http://localhost:8000/chat"
data = {
    "messages": [
        {"role": "user", "content": "I am hiring a Java developer"}
    ]
}

req = urllib.request.Request(url, json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode('utf-8'))
except Exception as e:
    print(e)
