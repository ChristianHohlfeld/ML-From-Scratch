from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
file=root.parent/'HDR_WaveForm_v04_complete.zip'
with zipfile.ZipFile(file,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if not p.is_file() or any(q in p.parts for q in ('__pycache__','.pytest_cache')) or p.suffix in ('.pyc','.nbc','.nbi'):continue
        z.write(p,arcname=str(Path('HDR_WaveForm_v04')/p.relative_to(root)))
print(file,file.stat().st_size)
