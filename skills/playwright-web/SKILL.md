---
name: playwright-web
description: Browser automation with Playwright for web scraping, testing, and interaction. Navigate pages, fill forms, capture screenshots, extract data.
author: jarvis
version: 1.0.0
tags: [playwright, browser, automation, scraping, testing]
metadata:
  jarvis:
    requires:
      bins: [npx]
      env: []
      config: []
    always: false
    user-invocable: true
    emoji: "🎭"
---

# Playwright Web Automation

Browser automation using Playwright for web scraping, testing, and complex web interactions.

## Setup

```bash
# Install Playwright
npm install -D @playwright/test
npx playwright install

# Or just run directly (auto-installs)
npx playwright --help
```

## Core Commands

### Navigation
```bash
# Open and navigate
npx playwright open https://example.com

# Screenshot
npx playwright screenshot https://example.com screenshot.png

# Generate PDF
npx playwright pdf https://example.com page.pdf
```

### Code Generation
```bash
# Record actions and generate code
npx playwright codegen https://example.com

# Generate to file
npx playwright codegen --target python https://example.com -o test.py
```

### Testing
```bash
# Run tests
npx playwright test

# Run specific test
npx playwright test example.spec.ts

# Debug mode
npx playwright test --debug

# UI mode
npx playwright test --ui
```

## Programmatic Usage

### Basic Script
```javascript
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('https://example.com');
  
  // Interact with page
  await page.fill('#username', 'user@example.com');
  await page.fill('#password', 'secret');
  await page.click('button[type="submit"]');
  
  // Wait for navigation
  await page.waitForURL('**/dashboard');
  
  // Extract data
  const title = await page.textContent('h1');
  const items = await page.$$eval('.item', els => els.map(e => e.textContent));
  
  await browser.close();
})();
```

### Python Version
```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('https://example.com')
    
    # Interact
    page.fill('#username', 'user@example.com')
    page.click('button[type="submit"]')
    
    # Extract
    title = page.text_content('h1')
    
    browser.close()
```

## Common Patterns

### Form Handling
```javascript
// Fill form fields
await page.fill('input[name="email"]', 'test@example.com');
await page.check('input[type="checkbox"]');  // Check
await page.select_option('select#country', 'US');  // Dropdown

// Handle file upload
await page.setInputFiles('input[type="file"]', 'path/to/file.pdf');

// Wait for form submission
await Promise.all([
  page.waitForNavigation(),
  page.click('button[type="submit"]')
]);
```

### Waiting Strategies
```javascript
// Wait for selector
await page.waitForSelector('.loaded');

// Wait for element state
await page.waitForSelector('button', { state: 'visible' });
await page.waitForSelector('#loading', { state: 'hidden' });

// Wait for function
await page.waitForFunction(() => document.querySelectorAll('.item').length > 5);

// Wait for network idle
await page.waitForLoadState('networkidle');
```

### Extracting Data
```javascript
// Single element
const text = await page.textContent('.title');

// Multiple elements
const links = await page.$$eval('a', els => 
  els.map(el => ({ text: el.textContent, href: el.href }))
);

// Table data
const rows = await page.$$eval('table tr', rows => 
  rows.map(row => Array.from(row.querySelectorAll('td'), td => td.textContent))
);

// Attributes
const src = await page.getAttribute('img.banner', 'src');
```

### Handling Dialogs & Popups
```javascript
// Accept dialog
page.on('dialog', async dialog => {
  console.log(dialog.message());
  await dialog.accept();
});

// Handle new page/tab
const [newPage] = await Promise.all([
  context.waitForEvent('page'),
  page.click('a[target="_blank"]')
]);
```

### Authentication
```javascript
// Basic auth
await page.goto('https://user:pass@example.com');

// Login flow (save state)
const context = await browser.newContext({
  storageState: 'auth.json'  // Uses saved cookies/storage
});
await page.login();  // After login
await context.storageState({ path: 'auth.json' });  // Save for reuse
```

### Screenshots & PDFs
```javascript
// Full page screenshot
await page.screenshot({ path: 'full.png', fullPage: true });

// Element screenshot
await page.locator('.chart').screenshot({ path: 'chart.png' });

// PDF generation
await page.pdf({ path: 'page.pdf', format: 'A4' });
```

## Browser Selection

```javascript
// Chromium (default - best compatibility)
const browser = await chromium.launch();

// Firefox
const browser = await firefox.launch();

// WebKit (Safari-like)
const browser = await webkit.launch();
```

## Best Practices

1. **Use locators over selectors**: `page.getByRole('button')` vs `page.click('button')`
2. **Wait strategically**: Prefer auto-waiting locators over explicit waits
3. **Handle async properly**: Use `Promise.all` for concurrent operations
4. **Isolate tests**: Each test should be independent
5. **Use context isolation**: Separate cookie/storage per test
6. **Mock network**: Use `page.route()` for reliable testing

## Anti-Detection Tips

```javascript
const browser = await chromium.launch({
  args: ['--disable-blink-features=AutomationControlled']
});

const context = await browser.newContext({
  userAgent: 'Mozilla/5.0...',
  viewport: { width: 1920, height: 1080 },
  locale: 'en-US',
  timezoneId: 'America/New_York'
});
```