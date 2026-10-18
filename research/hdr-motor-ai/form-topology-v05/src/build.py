from pathlib import Path
root=Path(__file__).resolve().parent.parent
html=(root/'web'/'index.html').read_text()
for name in ('meshes.js','bench.js'):
    html=html.replace(f'<script src="{name}"></script>', '<script>\n'+(root/'web'/name).read_text()+'\n</script>')
assert '<script src=' not in html
(root/'dist').mkdir(exist_ok=True)
out=root/'dist'/'index.html'
out.write_text(html)
print('Built',out,out.stat().st_size)
