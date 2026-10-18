from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1] / 'web'

def test_browser_live():
    html = (WEB/'index.html').read_text()
    for name in ('meshes.js','bench.js'):
        html=html.replace(f'<script src="{name}"></script>', '<script>'+ (WEB/name).read_text()+'</script>')
    with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
            for width,height,name in [(1320,900,'desktop'),(390,844,'mobile')]:
                page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
                errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.set_content(html)
                page.wait_for_timeout(600)
                assert page.locator('#nodes').inner_text()=='96'
                assert page.locator('.result').count()==4
                assert '88.9 %' in page.locator('.comparison').inner_text()
                page.select_option('#shape','torus');page.wait_for_timeout(100)
                assert page.locator('#loops').inner_text()=='2'
                page.select_option('#shape','two_spheres');page.wait_for_timeout(100)
                assert page.locator('#components').inner_text()=='2'
                page.select_option('#shape','dumbbell');page.wait_for_timeout(100)
                assert page.locator('#loops').inner_text()=='0'
                page.select_option('#coupling','shuffled')
                page.locator('#pulse').click()
                page.locator('#scale').fill('1.5')
                assert page.locator('#scaleValue').inner_text()=='1.5 ×'
                page.locator('#play').click(); assert 'Weiter' in page.locator('#play').inner_text()
                assert not errors, errors
                path=WEB.parent/'results'/f'{name}.png'
                page.screenshot(path=str(path),full_page=True)
                page.close()
            browser.close()
