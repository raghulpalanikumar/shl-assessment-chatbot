import asyncio
import os
import sys

sys.path.append("d:/Shl/backend")

from app import call_gemini, load_catalog, retrieve_catalog, build_prompt, Message

async def test():
    catalog = load_catalog()
    query = "Hiring a Java developer who works with stakeholders"
    messages = [Message(role="user", content=query)]
    retrieved = retrieve_catalog(query, catalog, limit=28)
    prompt = build_prompt(messages, retrieved)
    print("----- PROMPT -----")
    print(prompt)
    print("----- END PROMPT -----")
    res = await call_gemini(prompt)
    print("----- GEMINI RAW RESULT -----")
    print(res)

asyncio.run(test())
