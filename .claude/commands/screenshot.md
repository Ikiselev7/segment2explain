Take a screenshot of the running React app at http://localhost:5173 using Selenium headless Chrome.

Save the screenshot to `tests/fixtures/screenshots/` with a descriptive filename.

Use this Python snippet via `uv run python -c "..."`:

```python
import time, os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

opts = Options()
opts.add_argument('--headless')
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-dev-shm-usage')
opts.add_argument('--window-size=1920,1200')

driver = webdriver.Chrome(options=opts)
try:
    driver.get('http://localhost:5173')
    time.sleep(3)
    path = 'tests/fixtures/screenshots/<NAME>.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    driver.save_screenshot(path)
    print(f'Saved {path}')
finally:
    driver.quit()
```

After taking the screenshot, read the PNG file with the Read tool so you can see and describe what's on screen.

If the user provides arguments, use them as the screenshot filename. Otherwise use a timestamped name.
