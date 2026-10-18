from pathlib import Path
root=Path(__file__).resolve().parents[1]
html=(root/'web/index.html').read_text()
js=(root/'web/engine.js').read_text()
assert '<script src="engine.js"></script>' in html
html=html.replace('<script src="engine.js"></script>',f'<script>\n{js}\n</script>')
(root/'dist').mkdir(exist_ok=True)
(root/'dist/gear_lab.html').write_text(html)
print('Built dist/gear_lab.html:',len(html),'characters')
