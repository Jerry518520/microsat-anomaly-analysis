import requests, os
key = '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'
headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
payload = {'model': 'qwen/qwen3.5-397b-a17b', 'messages': [{'role':'user','content':'Say OK'}], 'max_tokens': 5, 'temperature': 0.1}
try:
    r = requests.post('https://integrate.api.nvidia.com/v1/chat/completions', headers=headers, json=payload, timeout=30)
    print(f'Status: {r.status_code}')
    if r.status_code == 200:
        print(f'Response: {r.json()["choices"][0]["message"]["content"]}')
    else:
        print(f'Error: {r.text[:300]}')
except Exception as e:
    print(f'Error: {e}')
