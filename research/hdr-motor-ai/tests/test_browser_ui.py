"""Optional real Chromium tests; run locally with `pytest tests/test_browser_ui.py`.

GitHub CI remains dependency-light: if playwright/Chromium is unavailable,
other deterministic Python + Node tests still run. The browser test is optional.
"""
from pathlib import Path
import shutil

import pytest

playwright = pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def test_desktop_and_mobile_interactions():
    chromium = shutil.which('chromium') or shutil.which('google-chrome')
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, executable_path=chromium) if chromium else p.chromium.launch(headless=True)
        except Exception as e:
            pytest.skip(f'Chromium not available: {e}')
        try:
            for width,height,label in ((1280,800,'desktop'),(390,844,'mobile')):
                page = browser.new_page(viewport={'width': width,'height':height},device_scale_factor=1)
                errors=[]
                page.on('pageerror',lambda e: errors.append(str(e)))
                page.set_content((ROOT/'dist'/'motor_lab.html').read_text(encoding='utf-8'), wait_until='load')
                page.wait_for_function('window.HDRLab && window.HDRLab.state.time > 0.1')
                assert page.locator('#motors input[type="range"]').count() == 6
                assert page.locator('#network line[data-edge]').count() == 28
                page.locator('#play').click()
                page.evaluate('window.HDRLab.pause()')
                before=page.evaluate('window.HDRLab.state.time')
                page.evaluate('window.HDRLab.step(20)')
                after=page.evaluate('window.HDRLab.state.time')
                assert after > before
                page.locator('#add').click()
                assert page.locator('#motors input[type="range"]').count() == 7
                page.locator('#remove').click()
                assert page.locator('#motors input[type="range"]').count() == 6
                page.locator('#network line[data-edge]').first.click(force=True)
                assert page.locator('#stiff').is_enabled()
                page.locator('#stiff').evaluate("el => { el.value='1.65'; el.dispatchEvent(new Event('input', {bubbles:true})); }")
                assert page.evaluate('window.HDRLab.model.edges[window.HDRLab.selected].k[0]') == pytest.approx(1.65)
                # Train once and verify local couplers change while the running rotor state is preserved.
                previous=page.evaluate('window.HDRLab.model.edges[26].k[0]')
                page.locator('#teach').evaluate("el => { el.value='-0.75'; el.dispatchEvent(new Event('input', {bubbles:true})); }")
                page.locator('#train_local').click()
                assert 'Gelernte Verbindungen:' in page.locator('#teachinfo').inner_text()
                assert page.evaluate('window.HDRLab.model.edges[26].k[0]') != pytest.approx(previous)
                assert page.locator('#status').inner_text().startswith('Lokale Lernphase beendet')
                (ROOT/'results'/'v04').mkdir(exist_ok=True)
                page.screenshot(path=str(ROOT/'results'/'v04'/f'v04_{label}.png'),full_page=True)
                assert not errors, str(errors)
                page.close()
        finally:
            browser.close()
