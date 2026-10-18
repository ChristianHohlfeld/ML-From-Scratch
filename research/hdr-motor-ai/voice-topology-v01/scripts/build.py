from pathlib import Path
import json
from scipy.signal import butter
root=Path(__file__).resolve().parents[1]
sos=[butter(3,[lo,hi],btype='bandpass',fs=8000,output='sos').tolist() for lo,hi in [(100,350),(350,650),(650,1050),(1050,1600),(1600,2400),(2400,3800)]]
js=(root/'web/engine.js').read_text().replace('__SOS__',json.dumps(sos,separators=(',',':')))
(root/'dist/engine.js').write_text(js)
html=(root/'web/template.html').read_text().replace('/*HDR_ENGINE*/',js)
(root/'dist/index.html').write_text(html)
print('built',len(html),'bytes')
