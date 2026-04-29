import requests

key = '[REDACTED]'
headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

# 火山引擎Ark用endpoint ID作为model参数
# 常见格式: ep-xxxxxxxx-xxxxx
# 也可能是用户自己创建的接入点
model_names = [
    'ep-20250429100000-deepseek',  # 示例格式
    '[REDACTED-KEY-PREFIX]',  # 用API key的前缀试试
]

# Also try listing models
try:
    r = requests.get('https://ark.cn-beijing.volces.com/api/v3/models',
                    headers=headers, timeout=10)
    print(f'List models status: {r.status_code}')
    if r.status_code == 200:
        print(r.text[:500])
    else:
        print(f'List error: {r.text[:300]}')
except Exception as e:
    print(f'List error: {e}')

print()
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
            print(f'  SUCCESS: {r.json()["choices"][0]["message"]["content"]}')
            break
        else:
            print(f'  Error: {r.text[:200]}')
    except Exception as e:
        print(f'Model: {model} | Error: {e}')
