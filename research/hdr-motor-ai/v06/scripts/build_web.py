"""Build a single offline HTML with the same engine tested in Node."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
page=(root/'web/index.html').read_text()
js=(root/'web/engine.js').read_text()
assert '<script src="engine.js"></script>' in page
html=page.replace('<script src="engine.js"></script>','<script>\n'+js+'\n</script>')
assert 'src="engine.js"' not in html
(root/'dist').mkdir(exist_ok=True)
(root/'dist/gear_lab.html').write_text(html)
print('built',root/'dist/gear_lab.html',len(html),'bytes')
