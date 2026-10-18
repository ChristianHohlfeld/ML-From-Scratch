from pathlib import Path
root=Path(__file__).resolve().parents[1]
template=(root/'web/template.html').read_text()
js=(root/'web/engine.js').read_text()
assert '/*__ENGINE__*/' in template
output=template.replace('/*__ENGINE__*/',js)
path=root/'dist/index.html';path.parent.mkdir(exist_ok=True);path.write_text(output)
print(f'Built {path} ({len(output)} bytes; no external JS/CSS)')
