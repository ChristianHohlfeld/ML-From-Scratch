from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
path=root/'dist/index.html'
root.joinpath('results').mkdir(exist_ok=True)
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-web-security'])
 for mode,w,h in [('desktop',1400,900),('mobile',390,844)]:
  page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
  errors=[];page.on('pageerror',lambda err: errors.append(str(err)))
  page.set_content(path.read_text(),wait_until='load',timeout=30000)
  page.wait_for_function("document.getElementById('cfl').textContent !== '—'",timeout=30000)
  print(mode,'ready',page.locator('#status').inner_text(),'loss',page.locator('#loss').inner_text(),flush=True)
  page.screenshot(path=str(root/'results'/f'{mode}.png'),full_page=True)
  assert not errors,errors
  if mode=='desktop':
   before=float(page.locator('#loss').inner_text())
   page.locator('#fit').click()
   page.wait_for_function("!document.getElementById('fit').disabled",timeout=60000)
   after=float(page.locator('#loss').inner_text())
   print('Fit loss',before,'->',after,'status',page.locator('#status').inner_text(),flush=True)
   assert after<before,(before,after)
   page.locator('#rod').check()
   page.wait_for_timeout(150)
   assert page.locator('#betti').inner_text()=='1 / 1'
   page.locator('[data-target="annular"]').click()
   assert page.locator('.preset.active').get_attribute('data-target')=='annular'
  else:
   assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'), 'mobile horizontal scroll'
  assert not errors,errors
  page.close()
 browser.close()
