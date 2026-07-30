import json, sqlite3, os
db = os.path.join(os.path.dirname(__file__), '..', '.codegraph', 'codegraph.db')
conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute("SELECT path,language,size,node_count FROM files ORDER BY path")
files = [{"path":r[0],"lang":r[1],"size":r[2],"nodes":r[3]} for r in cur.fetchall()]
cur.execute("SELECT kind,name,qualified_name,file_path,start_line,signature FROM nodes WHERE kind IN ('function','method','class') ORDER BY file_path,start_line")
funcs = [{"kind":r[0],"name":r[1],"qname":r[2],"file":r[3],"line":r[4],"sig":r[5]} for r in cur.fetchall()]
cur.execute("SELECT source,target,kind FROM edges WHERE kind IN ('calls','imports','instantiates')")
edges = [{"src":r[0],"tgt":r[1],"kind":r[2]} for r in cur.fetchall()]
out = {"files":files,"functions":funcs[:200],"edges":edges}
out_path = os.path.join(os.path.dirname(__file__), '..', 'docs', '_codegraph_data.json')
with open(out_path,"w",encoding="utf-8") as f:
    f.write(json.dumps(out,ensure_ascii=False,indent=1))
print("OK",len(files),"files",len(funcs),"funcs",len(edges),"edges")
