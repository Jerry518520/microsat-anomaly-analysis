import requests, json

key = 'ark-e1423552-067c-4537-80a2-ad6c764c3bbe-52da8'
headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

r = requests.get('https://ark.cn-beijing.volces.com/api/v3/models',
                headers=headers, timeout=10)
data = r.json()

# Filter for DeepSeek models
print('=== All available models ===')
for m in data['data']:
    mid = m['id']
    name = m.get('name', '')
    status = m.get('status', '')
    # Show all models, highlight deepseek
    marker = ' <<< ' if 'deep' in mid.lower() or 'deep' in name.lower() else ''
    if marker or 'Active' in status:
        print(f'{mid} | {name} | {status}{marker}')

print(f'\nTotal models: {len(data["data"])}')

# Also specifically search deepseek
print('\n=== DeepSeek models ===')
for m in data['data']:
    if 'deep' in m['id'].lower() or 'deep' in m.get('name','').lower():
        print(json.dumps(m, indent=2, ensure_ascii=False))
