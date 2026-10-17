"""Build a single offline HTML file with inlined tested physics engine."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    shell = (ROOT / 'web' / 'index.template.html').read_text(encoding='utf-8')
    engine = (ROOT / 'web' / 'engine.js').read_text(encoding='utf-8')
    # Inline implementation exactly: Node smoke tests exercise this JS, not a reimplementation.
    html = shell.replace('/* INLINE_ENGINE */', engine)
    preset = ROOT / 'results' / 'example_gears.json'
    learned = 'globalThis.HDR_LEARNED_PRESET = ' + preset.read_text(encoding='utf-8') + ';' if preset.exists() else 'globalThis.HDR_LEARNED_PRESET = null;'
    return html.replace('/* LEARNED_PRESET */', learned)


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    output=ROOT/'dist'/'motor_lab.html'
    data=build()
    if args.check:
        assert output.exists() and output.read_text(encoding='utf-8') == data, 'dist file stale; run build_web.py'
        print('Offline HTML up-to-date')
    else:
        output.parent.mkdir(exist_ok=True)
        output.write_text(data, encoding='utf-8')
        print(output)
if __name__=='__main__':main()
