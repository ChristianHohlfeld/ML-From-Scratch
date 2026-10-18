"""Run real Chromium UI smoke test locally (Playwright)."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

HTML=Path(__file__).resolve().parents[1]/'dist'/'physical_lab.html'
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
  for width,height in [(1280,850),(390,844)]:
   page=await browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
   errors=[]
   page.on('pageerror',lambda err:errors.append(str(err)))
   await page.set_content(HTML.read_text(encoding="utf-8"),wait_until="load")
   before=await page.locator('#pos').inner_text()
   await page.locator('#a').fill('20');await page.locator('#b').fill('30')
   after=await page.locator('#pos').inner_text()
   assert before!=after,('no effect',before)
   assert float((await page.locator('#mix').inner_text()).split()[0])>0
   assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),width
   assert not errors,errors
   await page.screenshot(path=str(HTML.parents[1]/'results'/('desktop.png' if width>500 else 'mobile.png')))
   await page.close()
  await browser.close()
if __name__=='__main__':asyncio.run(main())
