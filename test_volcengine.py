import requests, os

key = '[REDACTED]'
headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

# Try different model names
model_names = ['deepseek-v3-241226', 'deepseek-v3', 'DeepSeek-V3']

for model in model_names:
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': 'Say OK'}],
        'max_tokens': 5,
        'temperature': 0.1
    }
    try:
        r = requests.post('https://ark.cn-beijing.volces.com/api/v3/chat/completions', 
                         headers=headers, json=payload, timeout=15)
        print(f'Model: {model} | Status: {r.status_code}')
        if r.status_code == 200:
            data = r.json()
            print(f'  Response: {data["choices"][0]["message"]["content"]}')
            print(f'  Model used: {data.get("model", "?")}')
            break
        else:
            print(f'  Error: {r.text[:300]}')
    except Exception as e:
        print(f'Model: {model} | Error: {e}')
