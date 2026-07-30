import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'interactive-project-handover.html')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def R(p):
    with open(os.path.join(ROOT, p), 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def E(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

# Read project source files for code translations
src_main = R('src/api/main.py')
src_pipeline = R('src/integration/anomaly_rag_pipeline.py')
src_explain = R('src/api/routes/explanation.py')
src_dash = R('src/api/routes/dashboard.py')
src_rag = R('src/rag/pipeline.py')
src_ts = R('frontend/src/App.tsx')

# Truncate to key sections for display
def snippet(code, start_marker, end_marker, max_lines=20):
    lines = code.split('\n')
    start = 0
    for i, l in enumerate(lines):
        if start_marker in l:
            start = i
            break
    end = min(start + max_lines, len(lines))
    if end_marker:
        for i in range(start, end):
            if end_marker in lines[i]:
                end = i + 1
                break
    return '\n'.join(lines[start:end])

# CSS (no Chinese)
CSS = R('docs/interactive-project-handover.html').split('<style>')[1].split('</style>')[0] if os.path.exists(OUT) else ''