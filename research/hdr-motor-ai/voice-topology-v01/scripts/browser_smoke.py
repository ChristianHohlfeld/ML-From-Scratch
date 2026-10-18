from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
HTML=(ROOT/'dist/index.html').read_text()
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
    for kind in ('desktop','mobile'):
        viewport={'width':1365,'height':900} if kind=='desktop' else {'width':390,'height':844}
        ctx=browser.new_context(viewport=viewport,is_mobile=kind=='mobile',has_touch=kind=='mobile')
        page=ctx.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.set_content(HTML,wait_until='load',timeout=25000)
        page.wait_for_function("document.getElementById('geometry').textContent !== '—'",timeout=30000)
        page.locator('#rememberA').click();page.locator('#rememberB').click();page.locator('#recognize').click()
        assert 'Best fit:' in page.locator('#prediction').inner_text()
        page.locator('#pickB').click();page.locator('[data-v="u"]').click()
        assert 'Shape B ready' in page.locator('#status').inner_text()
        page.locator('#rotate').fill('180')
        if kind=='mobile': assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth+2'),'mobile overflow'
        assert not errors,errors
        page.screenshot(path=str(ROOT/'results'/f'{kind}.png'),full_page=True)
        ctx.close();print(kind,'PASS')
    browser.close()
