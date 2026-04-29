import requests

key = '[REDACTED]'
headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
payload = {
    'model': 'deepseek-v3-2-251201',
    'messages': [{'role': 'user', 'content': '请用中文回复：API连通测试成功'}],
    'max_tokens': 20,
    'temperature': 0.1
}

r = requests.post('https://ark.cn-beijing.volces.com/api/v3/chat/completions',
                 headers=headers, json=payload, timeout=30)
print(f'Status: {r.status_code}')
if r.status_code == 200:
    data = r.json()
    print(f'Model: {data.get("model", "?")}')
    print(f'Response: {data["choices"][0]["message"]["content"]}')
    print(f'Usage: {data.get("usage", {})}')
else:
    print(f'Error: {r.text[:500]}')
