from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
from threading import Thread
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
srv=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(root/'dist')))
Thread(target=srv.serve_forever,daemon=True).start()
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True, executable_path='/usr/bin/chromium',args=['--no-sandbox','--no-proxy-server','--proxy-bypass-list=*'])
    for w,h,tag in [(1440,900,'desktop'),(390,844,'mobile')]:
        page=browser.new_page(viewport={'width':w,'height':h}, device_scale_factor=1)
        errors=[]
        page.on('pageerror',lambda e: errors.append(str(e)))
        page.set_content((root/'dist/gear_lab.html').read_text(), wait_until='load')
        page.wait_for_timeout(500)
        assert page.locator('#mGears').inner_text()=='0'
        page.locator('#learn').click()
        assert page.locator('#mGears').inner_text()=='1'
        page.locator('#auto').click()
        n=int(page.locator('#mGears').inner_text());assert n>=1
        old=page.evaluate('window.__hdrLab.phase')
        page.wait_for_timeout(400)
        new=page.evaluate('window.__hdrLab.phase')
        assert new>old
        page.locator('#play').click()
        page.screenshot(path=str(root/'results'/f'{tag}.png'),full_page=True)
        assert not errors,errors
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
        print(tag,'gears',n,'phase',old,new,'error',page.locator('#mError').inner_text())
        page.close()
    browser.close()

srv.shutdown()
