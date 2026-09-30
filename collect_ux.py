import requests
from bs4 import BeautifulSoup
import re
import time
import csv


# ==========================================
# CONFIGURATION
# ==========================================

URLS = [
    "https://example.com",
]

OUTPUT_FILE = "ux_dataset.csv"


# ==========================================
# DOWNLOAD WEBSITE
# ==========================================

def get_html(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/153.0.0.0 Safari/537.36"
        )
    }

    print(f"Analyzing: {url}")

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        return response.text

    except Exception as e:

        print("ERROR:", e)

        return None


# ==========================================
# TEXT HELPERS
# ==========================================

def clean_text(text):

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def has_any_text(text, keywords):

    text = text.lower()

    return any(
        keyword.lower() in text
        for keyword in keywords
    )


# ==========================================
# EXTRACT UX FEATURES
# ==========================================

def extract_ux_features(html, url):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # ======================================
    # BASIC ELEMENTS
    # ======================================

    headings = soup.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6"
        ]
    )

    h1_tags = soup.find_all("h1")

    h2_tags = soup.find_all("h2")

    links = soup.find_all("a")

    buttons = soup.find_all("button")

    inputs = soup.find_all("input")

    images = soup.find_all("img")

    nav_elements = soup.find_all("nav")

    sections = soup.find_all(
        [
            "section",
            "article",
            "main"
        ]
    )

    # ======================================
    # PAGE TEXT
    # ======================================

    page_text = clean_text(
        soup.get_text(
            " ",
            strip=True
        )
    )

    word_count = len(
        page_text.split()
    )

    # ======================================
    # TITLE
    # ======================================

    title_tag = soup.find("title")

    has_title = bool(
        title_tag
        and clean_text(
            title_tag.get_text()
        )
    )

    # ======================================
    # META DESCRIPTION
    # ======================================

    meta_description = soup.find(
        "meta",
        attrs={
            "name": re.compile(
                "^description$",
                re.I
            )
        }
    )

    has_meta_description = bool(
        meta_description
        and meta_description.get("content")
    )

    # ======================================
    # H1
    # ======================================

    has_h1 = len(h1_tags) > 0

    # ======================================
    # NAVIGATION
    # ======================================

    has_navigation = (
        len(nav_elements) > 0
        or len(links) >= 5
    )

    # ======================================
    # CTA DETECTION
    # ======================================

    cta_keywords = [

        "contact",

        "get started",

        "learn more",

        "sign up",

        "signup",

        "register",

        "buy now",

        "shop now",

        "book now",

        "schedule",

        "request",

        "call now",

        "free consultation",

        "consultation",

        "get quote",

        "quote",

        "start",

        "apply"
    ]

    cta_count = 0

    for element in (
        buttons + links
    ):

        text = clean_text(
            element.get_text(
                " ",
                strip=True
            )
        )

        if text and has_any_text(
            text,
            cta_keywords
        ):

            cta_count += 1

    has_clear_cta = cta_count > 0

    # ======================================
    # BOOKING / APPOINTMENT
    # ======================================

    booking_keywords = [

        "book",

        "booking",

        "appointment",

        "schedule",

        "reserve",

        "reservation"
    ]

    has_booking_or_appointment = (
        has_any_text(
            page_text,
            booking_keywords
        )
    )

    # ======================================
    # PHONE
    # ======================================

    phone_pattern = r"""
        (\+?\d[\d\s().-]{7,}\d)
    """

    has_phone = bool(
        re.search(
            phone_pattern,
            page_text,
            re.VERBOSE
        )
    )

    # ======================================
    # EMAIL
    # ======================================

    has_email = bool(
        re.search(
            r"[\w\.-]+@[\w\.-]+\.\w+",
            page_text
        )
    )

    # ======================================
    # ADDRESS
    # ======================================

    address_keywords = [

        "address",

        "street",

        "road",

        "avenue",

        "ave",

        "boulevard",

        "blvd",

        "location"
    ]

    has_address = has_any_text(
        page_text,
        address_keywords
    )

    # ======================================
    # IMAGE ALT TEXT
    # ======================================

    images_with_alt = 0

    for image in images:

        alt = image.get("alt")

        if alt and alt.strip():

            images_with_alt += 1

    if images:

        image_alt_ratio = (
            images_with_alt /
            len(images)
        )

    else:

        image_alt_ratio = 1.0

    # ======================================
    # HEADING HIERARCHY
    # ======================================

    heading_hierarchy_score = 0

    if len(h1_tags) == 1:

        heading_hierarchy_score += 50

    elif len(h1_tags) > 1:

        heading_hierarchy_score += 20

    if len(h2_tags) > 0:

        heading_hierarchy_score += 30

    if len(headings) >= 3:

        heading_hierarchy_score += 20

    heading_hierarchy_score = min(
        heading_hierarchy_score,
        100
    )

    # ======================================
    # INFORMATION ARCHITECTURE
    # ======================================

    information_architecture_score = 0

    if has_navigation:

        information_architecture_score += 40

    if len(headings) >= 3:

        information_architecture_score += 30

    if len(sections) >= 2:

        information_architecture_score += 30

    information_architecture_score = min(
        information_architecture_score,
        100
    )

    # ======================================
    # CTA PRESENCE
    # ======================================

    cta_presence_score = min(
        cta_count * 25,
        100
    )

    # ======================================
    # CONVERSION UX SIGNAL
    # ======================================

    conversion_signals = 0

    if has_clear_cta:

        conversion_signals += 1

    if has_booking_or_appointment:

        conversion_signals += 1

    if has_phone:

        conversion_signals += 1

    if has_email:

        conversion_signals += 1

    conversion_ux_signal = (
        conversion_signals / 4
    ) * 100

    # ======================================
    # CONTENT DENSITY
    # ======================================

    content_density_signal = min(
        word_count / 10,
        100
    )

    # ======================================
    # MOBILE CONTENT DENSITY
    #
    # Real mobile rendering will be added
    # later through Playwright.
    # ======================================

    mobile_content_density_signal = (
        content_density_signal
    )

    # ======================================
    # VISUAL CLUTTER
    # ======================================

    visual_clutter_score = min(
        (
            len(links)
            + len(buttons)
            + len(images)
            + len(inputs)
        ) / 2,
        100
    )

    mobile_visual_clutter_score = (
        visual_clutter_score
    )

    # ======================================
    # VISUAL BALANCE
    #
    # Temporary conservative value.
    # Screenshot collector will replace this.
    # ======================================

    visual_balance_score = 70

    mobile_visual_balance_score = 70

    # ======================================
    # MOBILE INTERACTION
    # ======================================

    mobile_interaction_signal = min(
        (
            len(buttons)
            + len(inputs)
        ) * 10,
        100
    )

    # ======================================
    # MOBILE PENALTIES
    #
    # Browser-based measurements will be
    # added later.
    # ======================================

    mobile_small_button_penalty = 0

    mobile_image_overflow_penalty = 0

    # ======================================
    # CONTACTABILITY
    # ======================================

    contactability_signals = 0

    if has_phone:

        contactability_signals += 1

    if has_email:

        contactability_signals += 1

    if has_address:

        contactability_signals += 1

    contactability_ux_signal = (
        contactability_signals / 3
    ) * 100

    # ======================================
    # RETURN FEATURES
    # ======================================

    features = {

        # ----------------------------------
        # URL
        # ----------------------------------

        "url": url,

        # ----------------------------------
        # 14 UX MODEL FEATURES
        # ----------------------------------

        "heading_hierarchy_score":
            heading_hierarchy_score,

        "information_architecture_score":
            information_architecture_score,

        "cta_presence_score":
            cta_presence_score,

        "conversion_ux_signal":
            conversion_ux_signal,

        "content_density_signal":
            content_density_signal,

        "mobile_content_density_signal":
            mobile_content_density_signal,

        "visual_clutter_score":
            visual_clutter_score,

        "mobile_visual_clutter_score":
            mobile_visual_clutter_score,

        "visual_balance_score":
            visual_balance_score,

        "mobile_visual_balance_score":
            mobile_visual_balance_score,

        "mobile_interaction_signal":
            mobile_interaction_signal,

        "mobile_small_button_penalty":
            mobile_small_button_penalty,

        "mobile_image_overflow_penalty":
            mobile_image_overflow_penalty,

        "contactability_ux_signal":
            contactability_ux_signal,

        # ----------------------------------
        # RULE-BASED AUDIT FEATURES
        # ----------------------------------

        "has_title":
            int(has_title),

        "has_meta_description":
            int(has_meta_description),

        "has_h1":
            int(has_h1),

        "has_navigation":
            int(has_navigation),

        "has_phone":
            int(has_phone),

        "has_email":
            int(has_email),

        "has_address":
            int(has_address),

        "has_clear_cta":
            int(has_clear_cta),

        "has_booking_or_appointment":
            int(has_booking_or_appointment),

        # ----------------------------------
        # ADDITIONAL INFORMATION
        # ----------------------------------

        "has_images":
            int(len(images) > 0),

        "images":
            len(images),

        "images_with_alt":
            images_with_alt,

        "image_alt_ratio":
            round(
                image_alt_ratio,
                3
            ),

        "word_count":
            word_count,

        "headings":
            len(headings),

        "links":
            len(links),

        "buttons":
            len(buttons),

        "inputs":
            len(inputs),

        "navigation_elements":
            len(nav_elements),

        "sections":
            len(sections)
    }

    return features


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    all_rows = []

    for url in URLS:

        html = get_html(url)

        if html:

            features = extract_ux_features(
                html,
                url
            )

            all_rows.append(
                features
            )

            print()
            print("SUCCESS")
            print("-----------------------------------")

            for key, value in features.items():

                print(
                    f"{key}: {value}"
                )

        time.sleep(1)

    # ======================================
    # SAVE CSV
    # ======================================

    if all_rows:

        fieldnames = all_rows[0].keys()

        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                all_rows
            )

        print()
        print(
            "==================================="
        )

        print(
            "UX dataset saved successfully!"
        )

        print(
            "File:",
            OUTPUT_FILE
        )

        print(
            "==================================="
        )

    else:

        print(
            "No data collected."
        )