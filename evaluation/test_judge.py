import os, json, re
from openai import OpenAI

def extract_json(text):
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
        else:
            json_str = text
    return json_str

client = OpenAI()
resp = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[{"role": "user", "content": "What is 2+2? Gold: 4. Generated: The answer is 4.\n\nReturn JSON with key \"label\" set to CORRECT or WRONG."}],
    response_format={"type": "json_object"},
    temperature=0.0,
    extra_body={"thinking": {"type": "disabled"}},
)
content = resp.choices[0].message.content
print("Raw content:", repr(content))
print("Extracted:", repr(extract_json(content)))
try:
    print("Parsed:", json.loads(extract_json(content)))
except Exception as e:
    print("Parse error:", e)
