import json, sys, ast

path = r'F:\微小卫星项目\microsat-anomaly-analysis\notebooks\微小卫星遥测异常检测与RAG解释系统_完整Pipeline.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find syntax errors and print problematic lines
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        try:
            ast.parse(src)
        except SyntaxError as e:
            lines = src.split('\n')
            line_no = e.lineno - 1
            sys.stdout.buffer.write(f"\n=== Cell {i}: SyntaxError at line {e.lineno} ===\n".encode('utf-8'))
            start = max(0, line_no - 2)
            end = min(len(lines), line_no + 3)
            for j in range(start, end):
                marker = ">>> " if j == line_no else "    "
                sys.stdout.buffer.write(f"{marker}{j+1}: {repr(lines[j])}\n".encode('utf-8'))
