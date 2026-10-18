from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
html=(root/'dist/index.html').read_text()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    for mode,w,h in [('desktop',1380,850),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.set_content(html,wait_until='load')
        page.wait_for_function("document.getElementById('cfl').textContent !== '—'",timeout=30000)
        assert float(page.locator('#movement').inner_text())>0
        assert page.locator('#status').inner_text().startswith('Physical solution computed')
        page.screenshot(path=str(root/'results'/f'{mode}.png'),full_page=True)
        if mode=='desktop':
            before=float(page.locator('#loss').inner_text());page.locator('#fit').click()
            page.wait_for_function("!document.getElementById('fit').disabled",timeout=60000)
            after=float(page.locator('#loss').inner_text());print('Fit',before,after,flush=True)
            assert after<before
            page.locator('#wallEnabled').uncheck();assert float(page.locator('#movement').inner_text())==0
            page.locator('#wallEnabled').check();assert float(page.locator('#movement').inner_text())>0
            bb=page.locator('#scene').bounding_box()
            page.mouse.move(bb['x']+100,bb['y']+180)
            page.mouse.down();page.mouse.move(bb['x']+210,bb['y']+140,steps=5);page.mouse.up()
            assert not errors,errors
            page.locator('#rod').check();assert page.locator('#betti').inner_text()=='1 / 1'
        else: assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        print(mode,'OK',page.locator('#status').inner_text(),flush=True)
        assert not errors,errors
        page.close()
    browser.close()
