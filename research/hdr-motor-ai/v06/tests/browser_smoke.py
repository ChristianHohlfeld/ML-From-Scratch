"""Offline desktop/mobile browser regression. No external web resources."""
from pathlib import Path
from shutil import which
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from threading import Thread
from functools import partial
from playwright.sync_api import sync_playwright

pagefile=Path(__file__).resolve().parents[1]/'dist/gear_lab.html'
server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(pagefile.parent)))
thread=Thread(target=server.serve_forever,daemon=True)
url=f"http://127.0.0.1:{server.server_address[1]}/gear_lab.html"
if __name__=='__main__':
  thread.start()
  with sync_playwright() as p:
    for name,w,h,device_scale_factor in [('desktop',1440,900,1),('mobile',390,844,2)]:
      browser=p.chromium.launch(headless=True,**({'executable_path':which('chromium')} if which('chromium') else {}),args=['--no-sandbox'])
      page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=device_scale_factor,accept_downloads=True)
      errors=[]
      page.on('pageerror',lambda err:errors.append(str(err)))
      page.set_content(pagefile.read_text(),wait_until="load")
      page.wait_for_timeout(350)
      assert page.locator('#nStages').inner_text()=='0'
      page.get_by_role('button',name='Alles lernen').click()
      page.wait_for_timeout(400)
      assert page.locator('#nStages').inner_text()=='5', f"wrong stage count: {page.locator('#nStages').inner_text()}"
      assert page.locator('#nMixed').inner_text()=='3'
      assert float(page.locator('#err').inner_text())<1e-10
      page.locator('#motorA').fill('65'); page.locator('#motorB').fill('-95')
      page.wait_for_timeout(350)
      assert page.locator('#coord').inner_text().startswith('V₀ =')
      assert not errors, errors
      assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+1'), 'horizontal mobile overflow'
      (pagefile.parent.parent/'results').mkdir(exist_ok=True)
      page.screenshot(path=str(pagefile.parent.parent/'results'/f'{name}.png'),full_page=True)
      # Ablation on known interaction should not reproduce full compound task.
      page.locator('#reset').click();page.locator('#allow').uncheck();page.locator('#auto').click()
      assert float(page.locator('#err').inner_text())>.1
      page.locator('#task').select_option('additive');page.locator('#auto').click()
      assert float(page.locator('#err').inner_text())<1e-8
      with page.expect_download() as d:page.locator('#export').click()
      assert d.value.suggested_filename.endswith('.json')
      print(name,'ok','viewport',w,'x',h,'errors',errors)
      browser.close()

  server.shutdown()
