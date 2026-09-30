import os
import re
import pandas as pd
from playwright.sync_api import sync_playwright

URL_FILE = "data/visual_urls.txt"
OUTPUT_FILE = "data/visual_raw_features.csv"

os.makedirs("data", exist_ok=True)


def safe_count(page, selector):
    try:
        return page.locator(selector).count()
    except Exception:
        return 0


def get_numeric(value):
    try:
        return float(value)
    except Exception:
        return 0


def analyze_desktop(page):
    return page.evaluate("""
    () => {
        const elements = [...document.querySelectorAll('*')];

        const visible = elements.filter(el => {
            const r = el.getBoundingClientRect();
            const s = getComputedStyle(el);
            return r.width > 0 &&
                   r.height > 0 &&
                   s.display !== 'none' &&
                   s.visibility !== 'hidden';
        });

        const images = [...document.images];

        const headings = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')];

        const fonts = new Set();
        const colors = new Set();

        visible.forEach(el => {
            const s = getComputedStyle(el);

            if (s.fontFamily) {
                fonts.add(s.fontFamily);
            }

            if (s.color) {
                colors.add(s.color);
            }

            if (s.backgroundColor &&
                s.backgroundColor !== 'rgba(0, 0, 0, 0)') {
                colors.add(s.backgroundColor);
            }
        });

        const largeImages = images.filter(img => {
            return img.naturalWidth >= 600 ||
                   img.naturalHeight >= 400;
        });

        const imagesWithAlt = images.filter(img => {
            return img.alt && img.alt.trim().length > 0;
        });

        const wordCount = document.body
            ? document.body.innerText.trim().split(/\\s+/).filter(Boolean).length
            : 0;

        return {
            links: document.querySelectorAll('a').length,

            buttons:
                document.querySelectorAll('button, input[type="button"], input[type="submit"], [role="button"]').length,

            cta_count:
                [...document.querySelectorAll(
                    'a, button, input[type="button"], input[type="submit"], [role="button"]'
                )].filter(el => {
                    const text = (
                        el.innerText ||
                        el.value ||
                        el.getAttribute('aria-label') ||
                        ''
                    ).toLowerCase().trim();

                    return /book|buy|shop|contact|get started|sign up|signup|register|learn more|apply|call|schedule|appointment|subscribe|request|quote|start/.test(text);
                }).length,

            images: images.length,

            images_with_alt: imagesWithAlt.length,

            large_images: largeImages.length,

            headings: headings.length,

            h1_count: document.querySelectorAll('h1').length,

            h2_count: document.querySelectorAll('h2').length,

            inputs: document.querySelectorAll('input, textarea, select').length,

            nav_elements:
                document.querySelectorAll('nav, [role="navigation"]').length,

            sections:
                document.querySelectorAll(
                    'section, main > div, article, header, footer'
                ).length,

            font_count: fonts.size,

            color_count: colors.size,

            word_count: wordCount,

            page_height: Math.max(
                document.body ? document.body.scrollHeight : 0,
                document.documentElement.scrollHeight
            ),

            viewport_width: window.innerWidth,

            scroll_width: Math.max(
                document.body ? document.body.scrollWidth : 0,
                document.documentElement.scrollWidth
            ),

            visible_elements: visible.length
        };
    }
    """)


def analyze_mobile(page):
    return page.evaluate("""
    () => {
        const images = [...document.images];

        const overflowingImages = images.filter(img => {
            const r = img.getBoundingClientRect();
            return r.right > window.innerWidth + 2 ||
                   r.left < -2;
        });

        const smallButtons = [
            ...document.querySelectorAll(
                'button, input[type="button"], input[type="submit"], [role="button"]'
            )
        ].filter(el => {
            const r = el.getBoundingClientRect();

            return r.width > 0 &&
                   r.height > 0 &&
                   (r.width < 44 || r.height < 44);
        });

        const visibleButtons = [
            ...document.querySelectorAll(
                'button, input[type="button"], input[type="submit"], [role="button"]'
            )
        ].filter(el => {
            const r = el.getBoundingClientRect();

            return r.width > 0 && r.height > 0;
        });

        return {
            viewport_width: window.innerWidth,

            scroll_width: Math.max(
                document.body ? document.body.scrollWidth : 0,
                document.documentElement.scrollWidth
            ),

            horizontal_overflow:
                Math.max(
                    document.body ? document.body.scrollWidth : 0,
                    document.documentElement.scrollWidth
                ) > window.innerWidth + 2,

            mobile_buttons: visibleButtons.length,

            small_buttons: smallButtons.length,

            mobile_images: images.length,

            overflowing_images: overflowingImages.length
        };
    }
    """)


def calculate_scores(d):
    # Layout
    layout = 100

    if d["horizontal_overflow"]:
        layout -= 25

    if d["page_height"] > 15000:
        layout -= 10

    if d["visible_elements"] < 80:
        layout -= 10

    layout = max(40, min(100, layout))

    # Typography
    typography = 100

    if d["h1_count"] == 0:
        typography -= 15

    if d["headings"] < 3:
        typography -= 10

    if d["font_count"] <= 1:
        typography -= 5

    typography = max(40, min(100, typography))

    # Color
    color = 100

    if d["color_count"] < 4:
        color -= 15

    color = max(40, min(100, color))

    # Hierarchy
    hierarchy = 100

    if d["h1_count"] == 0:
        hierarchy -= 15

    if d["h2_count"] == 0:
        hierarchy -= 10

    if d["sections"] < 3:
        hierarchy -= 10

    hierarchy = max(40, min(100, hierarchy))

    # Navigation
    navigation = 100

    if d["links"] < 5:
        navigation -= 20

    if d["nav_elements"] == 0:
        navigation -= 15

    navigation = max(40, min(100, navigation))

    # CTA
    cta = min(100, 25 + d["cta_count"] * 10)

    if d["buttons"] == 0:
        cta -= 10

    cta = max(20, min(100, cta))

    # Image quality
    if d["images"] == 0:
        image_quality = 40
    else:
        alt_ratio = d["images_with_alt"] / d["images"]

        image_quality = 45 + min(
            35,
            d["large_images"] * 5
        )

        image_quality += min(
            20,
            alt_ratio * 20
        )

    image_quality = max(30, min(100, round(image_quality)))

    # Whitespace / structure
    whitespace = 95

    if d["visible_elements"] > 1200:
        whitespace -= 10

    if d["page_height"] > 20000:
        whitespace -= 5

    whitespace = max(60, min(100, whitespace))

    # Mobile
    mobile = 100

    if d["horizontal_overflow"]:
        mobile -= 25

    mobile -= min(
        20,
        d["overflowing_images"] * 5
    )

    if d["mobile_buttons"] > 0:
        small_ratio = d["small_buttons"] / d["mobile_buttons"]

        if small_ratio > 0.5:
            mobile -= 10

    mobile = max(40, min(100, mobile))

    # Overall visual score
    visual_score = round(
        layout * 0.15 +
        typography * 0.10 +
        color * 0.10 +
        hierarchy * 0.10 +
        navigation * 0.10 +
        cta * 0.15 +
        image_quality * 0.10 +
        whitespace * 0.05 +
        mobile * 0.15
    )

    return {
        "layout_score": layout,
        "typography_score": typography,
        "color_score": color,
        "hierarchy_score": hierarchy,
        "navigation_score": navigation,
        "cta_score": cta,
        "image_quality_score": image_quality,
        "whitespace_score": whitespace,
        "mobile_score": mobile,
        "visual_score": visual_score
    }


def main():
    if not os.path.exists(URL_FILE):
        print(f"ERROR: {URL_FILE} not found.")
        return

    with open(URL_FILE, "r", encoding="utf-8") as f:
        urls = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        desktop = browser.new_context(
            viewport={"width": 1440, "height": 1000}
        )

        mobile = browser.new_context(
            viewport={"width": 390, "height": 844},
            is_mobile=True,
            device_scale_factor=1
        )

        desktop_page = desktop.new_page()
        mobile_page = mobile.new_page()

        for index, url in enumerate(urls, start=1):

            print(f"\n[{index}/{len(urls)}] {url}")

            try:
                desktop_page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=30000
                )

                desktop_page.wait_for_timeout(1500)

                desktop_data = analyze_desktop(desktop_page)

            except Exception as e:
                print(f"Desktop error: {e}")
                desktop_data = {
                    "links": 0,
                    "buttons": 0,
                    "cta_count": 0,
                    "images": 0,
                    "images_with_alt": 0,
                    "large_images": 0,
                    "headings": 0,
                    "h1_count": 0,
                    "h2_count": 0,
                    "inputs": 0,
                    "nav_elements": 0,
                    "sections": 0,
                    "font_count": 0,
                    "color_count": 0,
                    "word_count": 0,
                    "page_height": 0,
                    "viewport_width": 1440,
                    "scroll_width": 1440,
                    "visible_elements": 0
                }

            try:
                mobile_page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=30000
                )

                mobile_page.wait_for_timeout(1500)

                mobile_data = analyze_mobile(mobile_page)

            except Exception as e:
                print(f"Mobile error: {e}")
                mobile_data = {
                    "viewport_width": 390,
                    "scroll_width": 390,
                    "horizontal_overflow": False,
                    "mobile_buttons": 0,
                    "small_buttons": 0,
                    "mobile_images": 0,
                    "overflowing_images": 0
                }

            data = {
                "image": f"website_{index:03d}",
                "url": url,
                **desktop_data,
                **{
                    f"mobile_{k}": v
                    for k, v in mobile_data.items()
                    if k != "viewport_width"
                }
            }

            scores = calculate_scores({
                **desktop_data,
                **mobile_data
            })

            data.update(scores)

            rows.append(data)

            print(
                f"Desktop elements: {desktop_data['visible_elements']} | "
                f"Mobile overflow: {mobile_data['horizontal_overflow']} | "
                f"Visual Score: {scores['visual_score']}/100"
            )

        browser.close()

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========================================")
    print("RAW VISUAL DATASET CREATED")
    print("========================================")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()