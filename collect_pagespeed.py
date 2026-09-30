import requests
import csv
import time
import os

from website_list import URLS


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("PAGESPEED_API_KEY", "")

OUTPUT_FILE = "pagespeed_dataset.csv"

# Faster settings
TIMEOUT = 30
MAX_ATTEMPTS = 2
RETRY_WAIT = 2
DELAY_BETWEEN_URLS = 1

# Use mobile PageSpeed analysis
STRATEGY = "mobile"


# ============================================================
# DATASET COLUMNS
# ============================================================

FIELDNAMES = [
    "url",
    "first_contentful_paint",
    "largest_contentful_paint",
    "speed_index",
    "interactive",
    "total_blocking_time",
    "cumulative_layout_shift",
    "server_response_time",
    "bootup_time",
    "total_resources",
    "total_transfer_size",
    "script_resources",
    "image_resources",
    "stylesheet_resources",
    "font_resources",
]


# ============================================================
# LOAD ALREADY COLLECTED WEBSITES
# ============================================================

def load_existing_urls():

    existing_urls = set()

    if not os.path.exists(OUTPUT_FILE):
        return existing_urls

    try:
        with open(
            OUTPUT_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                url = row.get("url")

                if url:
                    existing_urls.add(url.strip())

        print(
            f"Found existing dataset: "
            f"{len(existing_urls)} websites already collected."
        )

    except Exception as e:

        print(f"Could not read existing dataset: {e}")

    return existing_urls


# ============================================================
# SAVE ONE RESULT
# ============================================================

def save_result(data):

    file_exists = os.path.exists(OUTPUT_FILE)

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


# ============================================================
# GET NUMERIC AUDIT VALUE
# ============================================================

def numeric_value(audits, audit_id):

    try:

        audit = audits.get(audit_id, {})

        numeric = audit.get("numericValue")

        if numeric is None:
            return 0

        return float(numeric)

    except Exception:
        return 0


# ============================================================
# RESOURCE SUMMARY
# ============================================================

def resource_summary(audits):

    result = {
        "total_resources": 0,
        "total_transfer_size": 0,
        "script_resources": 0,
        "image_resources": 0,
        "stylesheet_resources": 0,
        "font_resources": 0,
    }

    try:

        audit = audits.get("resource-summary", {})

        items = audit.get("details", {}).get("items", [])

        for item in items:

            resource_type = str(
                item.get("resourceType", "")
            ).lower()

            request_count = item.get(
                "requestCount",
                0
            )

            transfer_size = item.get(
                "transferSize",
                0
            )

            result["total_resources"] += int(
                request_count or 0
            )

            result["total_transfer_size"] += int(
                transfer_size or 0
            )

            if resource_type == "script":
                result["script_resources"] += int(
                    request_count or 0
                )

            elif resource_type == "image":
                result["image_resources"] += int(
                    request_count or 0
                )

            elif resource_type == "stylesheet":
                result["stylesheet_resources"] += int(
                    request_count or 0
                )

            elif resource_type == "font":
                result["font_resources"] += int(
                    request_count or 0
                )

    except Exception as e:

        print(
            f"Resource summary warning: {e}"
        )

    return result


# ============================================================
# CALL GOOGLE PAGESPEED API
# ============================================================

def get_pagespeed_data(url):

    api_url = (
        "https://www.googleapis.com/"
        "pagespeedonline/v5/runPagespeed"
    )

    params = {
        "url": url,
        "key": API_KEY,
        "strategy": STRATEGY,
        "category": "performance",
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:

            response = requests.get(
                api_url,
                params=params,
                timeout=TIMEOUT
            )

            if response.status_code == 200:

                return response.json()

            print(
                f"Attempt {attempt}/{MAX_ATTEMPTS} "
                f"HTTP {response.status_code}"
            )

            # API rate limit
            if response.status_code == 429:

                print(
                    "Rate limit reached. "
                    "Waiting before retry..."
                )

                time.sleep(5)

            else:

                time.sleep(RETRY_WAIT)

        except requests.exceptions.Timeout:

            print(
                f"Attempt {attempt}/{MAX_ATTEMPTS} timed out"
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT)

        except requests.exceptions.RequestException as e:

            print(
                f"Attempt {attempt}/{MAX_ATTEMPTS} "
                f"request error: {e}"
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT)

        except Exception as e:

            print(
                f"Attempt {attempt}/{MAX_ATTEMPTS} "
                f"unexpected error: {e}"
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT)

    return None


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(url, data):

    lighthouse = data.get(
        "lighthouseResult",
        {}
    )

    audits = lighthouse.get(
        "audits",
        {}
    )

    resources = resource_summary(audits)

    return {

        "url": url,

        "first_contentful_paint":
            numeric_value(
                audits,
                "first-contentful-paint"
            ),

        "largest_contentful_paint":
            numeric_value(
                audits,
                "largest-contentful-paint"
            ),

        "speed_index":
            numeric_value(
                audits,
                "speed-index"
            ),

        "interactive":
            numeric_value(
                audits,
                "interactive"
            ),

        "total_blocking_time":
            numeric_value(
                audits,
                "total-blocking-time"
            ),

        "cumulative_layout_shift":
            numeric_value(
                audits,
                "cumulative-layout-shift"
            ),

        "server_response_time":
            numeric_value(
                audits,
                "server-response-time"
            ),

        "bootup_time":
            numeric_value(
                audits,
                "bootup-time"
            ),

        "total_resources":
            resources["total_resources"],

        "total_transfer_size":
            resources["total_transfer_size"],

        "script_resources":
            resources["script_resources"],

        "image_resources":
            resources["image_resources"],

        "stylesheet_resources":
            resources["stylesheet_resources"],

        "font_resources":
            resources["font_resources"],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if API_KEY == "PASTE_YOUR_NEW_API_KEY_HERE":

        print()
        print("ERROR: PageSpeed API key is not configured.")
        print()
        print(
            "Open collect_pagespeed.py and replace:"
        )
        print(
            "PASTE_YOUR_NEW_API_KEY_HERE"
        )
        print(
            "with your new PageSpeed API key."
        )
        print()

        return

    print("=" * 60)
    print("GOOGLE PAGESPEED DATA COLLECTION")
    print("=" * 60)

    print(
        f"Total URLs in website_list.py: {len(URLS)}"
    )

    print(
        f"Timeout per attempt: {TIMEOUT} seconds"
    )

    print(
        f"Maximum attempts: {MAX_ATTEMPTS}"
    )

    print(
        f"Strategy: {STRATEGY}"
    )

    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Load existing results
    # --------------------------------------------------------

    existing_urls = load_existing_urls()

    remaining_urls = [
        url for url in URLS
        if url not in existing_urls
    ]

    print(
        f"Already collected: {len(existing_urls)}"
    )

    print(
        f"Remaining websites: {len(remaining_urls)}"
    )

    print()

    if not remaining_urls:

        print(
            "All websites have already been collected."
        )

        return

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    successful = 0
    failed = 0

    total_remaining = len(remaining_urls)

    # --------------------------------------------------------
    # Process websites
    # --------------------------------------------------------

    for index, url in enumerate(
        remaining_urls,
        start=1
    ):

        print(
            f"[{index}/{total_remaining}] "
            f"Starting analysis"
        )

        print(
            f"Analyzing: {url}"
        )

        data = get_pagespeed_data(url)

        if data is None:

            failed += 1

            print(
                "SKIPPED - Could not collect data"
            )

            print()

            continue

        try:

            features = extract_features(
                url,
                data
            )

            save_result(features)

            successful += 1

            print("SUCCESS")

            print("Collected data:")

            print(features)

        except Exception as e:

            failed += 1

            print(
                f"FAILED - Could not extract data: {e}"
            )

        print()

        # Small delay to avoid hammering the API
        time.sleep(DELAY_BETWEEN_URLS)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)

    print(
        f"New successful websites: {successful}"
    )

    print(
        f"New failed/skipped websites: {failed}"
    )

    print(
        f"Previously collected websites: "
        f"{len(existing_urls)}"
    )

    print(
        f"Total successful dataset size: "
        f"{len(existing_urls) + successful}"
    )

    print(
        f"Dataset saved to: {OUTPUT_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()

