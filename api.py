from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError
)

from extract_advanced_screenshot_features import extract_features
import final_audit
import collect_ux
from extract_visual_features import analyze_desktop, analyze_mobile


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Website Audit ML API",
    version="4.2.0"
)


# ============================================================
# PATHS
# ============================================================

MODELS_DIR = "models"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_feature_names(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return [
            line.strip()
            for line in f
            if line.strip()
        ]


def create_feature_array(
    features,
    feature_names
):

    model_features = {}

    for feature in feature_names:

        value = features.get(
            feature,
            0
        )

        try:

            value = float(value)

        except (
            TypeError,
            ValueError
        ):

            value = 0.0

        model_features[feature] = value

    return pd.DataFrame(
        [model_features],
        columns=feature_names
    )


# ============================================================
# PAGE SPEED MODEL
# ============================================================

print(
    "Loading PageSpeed ML model..."
)

pagespeed_model = joblib.load(
    f"{MODELS_DIR}/pagespeed_website_score_model.joblib"
)

pagespeed_feature_names = load_feature_names(
    f"{MODELS_DIR}/pagespeed_feature_names.txt"
)

print(
    "PageSpeed model loaded successfully."
)

print(
    "PageSpeed features required:",
    len(pagespeed_feature_names)
)


# ============================================================
# CLEAN UX MODEL
# ============================================================

print(
    "Loading Clean UX ML model..."
)

ux_model = joblib.load(
    f"{MODELS_DIR}/clean_ux_quality_model.joblib"
)

ux_feature_names = load_feature_names(
    f"{MODELS_DIR}/clean_ux_feature_names.txt"
)

print(
    "Clean UX model loaded successfully."
)

print(
    "Clean UX features required:",
    len(ux_feature_names)
)


# ============================================================
# VISUAL RAW MODEL
# ============================================================

print(
    "Loading Visual Raw ML model..."
)

visual_model = joblib.load(
    f"{MODELS_DIR}/visual_raw_quality_model.joblib"
)

visual_feature_names = load_feature_names(
    f"{MODELS_DIR}/visual_raw_feature_names.txt"
)

print(
    "Visual Raw model loaded successfully."
)

print(
    "Visual Raw features required:",
    len(visual_feature_names)
)


# ============================================================
# ADVANCED SCREENSHOT MODEL
# ============================================================

print(
    "Loading Advanced Screenshot ML model..."
)

advanced_screenshot_bundle = joblib.load(
    f"{MODELS_DIR}/advanced_screenshot_quality_model_v3.joblib"
)


if isinstance(
    advanced_screenshot_bundle,
    dict
):

    advanced_screenshot_model = (
        advanced_screenshot_bundle["model"]
    )

    advanced_screenshot_label_encoder = (
        advanced_screenshot_bundle["label_encoder"]
    )

else:

    raise TypeError(
        "Advanced screenshot model must be "
        "a dictionary containing 'model' "
        "and 'label_encoder'."
    )


advanced_screenshot_feature_names = (
    load_feature_names(
        f"{MODELS_DIR}/advanced_screenshot_feature_names_v3.txt"
    )
)


print(
    "Advanced screenshot model loaded successfully."
)

print(
    "Advanced screenshot model:",
    type(
        advanced_screenshot_model
    ).__name__
)

print(
    "Advanced screenshot label encoder:",
    type(
        advanced_screenshot_label_encoder
    ).__name__
)

print(
    "Advanced screenshot features required:",
    len(
        advanced_screenshot_feature_names
    )
)


# ============================================================
# REQUEST MODELS
# ============================================================

class WebsiteData(BaseModel):

    features: dict[str, float]


class AuditRequest(BaseModel):

    features: dict[str, float]

    website: str | None = None


class VisualAuditRequest(BaseModel):

    website: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "status": "online",

        "service": "Website Audit ML API",

        "version": "4.2.0",

        "models": {

            "pagespeed": True,

            "clean_ux": True,

            "visual_raw": True,

            "advanced_screenshot": True

        }

    }


# ============================================================
# PAGE SPEED PREDICTION
# ============================================================

@app.post("/predict")
def predict_pagespeed(
    data: WebsiteData
):

    X = create_feature_array(
        data.features,
        pagespeed_feature_names
    )

    prediction = float(
        pagespeed_model.predict(X)[0]
    )

    prediction = max(
        0.0,
        min(
            100.0,
            prediction
        )
    )

    if prediction >= 90:

        quality = "GOOD"

    elif prediction >= 50:

        quality = "AVERAGE"

    else:

        quality = "POOR"

    return {

        "success": True,

        "score": round(
            prediction,
            2
        ),

        "quality": quality,

        "category": quality

    }


# ============================================================
# CLEAN UX PREDICTION
# ============================================================

@app.post("/predict-ux")
def predict_ux(
    data: WebsiteData
):

    X = create_feature_array(
        data.features,
        ux_feature_names
    )

    prediction = ux_model.predict(
        X
    )[0]

    quality = str(
        prediction
    ).upper()

    if quality == "GOOD":

        score = 90

    elif quality == "AVERAGE":

        score = 60

    else:

        score = 30

    result = {

        "success": True,

        "quality": quality,

        "score": score

    }

    if hasattr(
        ux_model,
        "predict_proba"
    ):

        probabilities = (
            ux_model.predict_proba(X)[0]
        )

        result["confidence"] = round(

            float(
                np.max(
                    probabilities
                )
            ) * 100,

            2

        )

        result["probabilities"] = {

            str(label).upper():

            round(
                float(probability) * 100,
                2
            )

            for label, probability

            in zip(
                ux_model.classes_,
                probabilities
            )

        }

    return result


# ============================================================
# VISUAL RAW PREDICTION
# ============================================================

@app.post("/predict-visual")
def predict_visual(
    data: WebsiteData
):

    X = create_feature_array(
        data.features,
        visual_feature_names
    )

    prediction = visual_model.predict(
        X
    )[0]

    quality = str(
        prediction
    ).upper()

    if quality == "GOOD":

        score = 90

    elif quality == "AVERAGE":

        score = 60

    else:

        score = 30

    result = {

        "success": True,

        "quality": quality,

        "score": score

    }

    if hasattr(
        visual_model,
        "predict_proba"
    ):

        probabilities = (
            visual_model.predict_proba(X)[0]
        )

        result["confidence"] = round(

            float(
                np.max(
                    probabilities
                )
            ) * 100,

            2

        )

        result["probabilities"] = {

            str(label).upper():

            round(
                float(probability) * 100,
                2
            )

            for label, probability

            in zip(
                visual_model.classes_,
                probabilities
            )

        }

    return result


# ============================================================
# ADVANCED SCREENSHOT PREDICTION
# ============================================================

@app.post("/predict-screenshot")
def predict_screenshot(
    data: WebsiteData
):

    X = create_feature_array(
        data.features,
        advanced_screenshot_feature_names
    )

    encoded_prediction = (
        advanced_screenshot_model.predict(X)[0]
    )

    try:

        prediction = (
            advanced_screenshot_label_encoder
            .inverse_transform(
                [encoded_prediction]
            )[0]
        )

    except Exception:

        prediction = encoded_prediction

    quality = str(
        prediction
    ).upper()

    if quality == "GOOD":

        score = 90

    elif quality == "AVERAGE":

        score = 60

    else:

        score = 30

    result = {

        "success": True,

        "quality": quality,

        "score": score,

        "features_used": len(
            advanced_screenshot_feature_names
        ),

        "input_type": (
            "combined desktop + mobile "
            "screenshot features"
        )

    }

    if hasattr(
        advanced_screenshot_model,
        "predict_proba"
    ):

        probabilities = (
            advanced_screenshot_model
            .predict_proba(X)[0]
        )

        result["confidence"] = round(

            float(
                np.max(
                    probabilities
                )
            ) * 100,

            2

        )

        if hasattr(
            advanced_screenshot_label_encoder,
            "classes_"
        ):

            probability_labels = (
                advanced_screenshot_label_encoder
                .classes_
            )

        elif hasattr(
            advanced_screenshot_model,
            "classes_"
        ):

            probability_labels = (
                advanced_screenshot_model
                .classes_
            )

        else:

            probability_labels = [

                str(i)

                for i in range(
                    len(probabilities)
                )

            ]

        result["probabilities"] = {

            str(label).upper():

            round(
                float(probability) * 100,
                2
            )

            for label, probability

            in zip(
                probability_labels,
                probabilities
            )

        }

    return result


# ============================================================
# AUTOMATIC VISUAL WEBSITE AUDIT
# ============================================================

@app.post("/visual-audit")
def visual_audit(
    data: VisualAuditRequest
):

    website = data.website.strip()

    # --------------------------------------------------------
    # VALIDATE URL
    # --------------------------------------------------------

    if not website.startswith(
        ("http://", "https://")
    ):

        return {

            "success": False,

            "website": website,

            "error": (
                "Website URL must start with "
                "http:// or https://"
            )

        }

    # --------------------------------------------------------
    # CREATE TEMPORARY DIRECTORY
    # --------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="visual_audit_"
        )
    )

    desktop_path = (
        temp_dir / "desktop.png"
    )

    mobile_path = (
        temp_dir / "mobile.png"
    )

    browser = None

    try:

        print()
        print("=" * 70)
        print("AUTOMATIC VISUAL WEBSITE AUDIT")
        print("=" * 70)

        print(
            f"Website: {website}"
        )

        # ====================================================
        # PLAYWRIGHT
        # ====================================================

        with sync_playwright() as p:

            browser = p.chromium.launch(
                headless=True
            )

            # =================================================
            # DESKTOP
            # =================================================

            print()
            print(
                "Capturing desktop screenshot..."
            )

            desktop_context = browser.new_context(
                viewport={
                    "width": 1440,
                    "height": 1000
                }
            )

            desktop_page = (
                desktop_context.new_page()
            )

            desktop_page.set_default_navigation_timeout(
                25000
            )

            try:

                desktop_page.goto(
                    website,
                    wait_until="domcontentloaded",
                    timeout=25000
                )

                print(
                    "Desktop page loaded."
                )

            except PlaywrightTimeoutError:

                print(
                    "Warning: Desktop navigation "
                    "timed out. Continuing..."
                )

            except Exception as e:

                print(
                    "Warning: Desktop navigation "
                    f"issue: {e}"
                )

            # -------------------------------------------------
            # SHORT WAIT
            # -------------------------------------------------

            desktop_page.wait_for_timeout(
                1500
            )

            # -------------------------------------------------
            # LIMITED SCROLL
            # -------------------------------------------------

            try:

                desktop_page.evaluate(
                    """
                    async () => {

                        const maxScrolls = 8;
                        const distance = 600;

                        for (
                            let i = 0;
                            i < maxScrolls;
                            i++
                        ) {

                            window.scrollBy(
                                0,
                                distance
                            );

                            await new Promise(
                                resolve =>
                                setTimeout(
                                    resolve,
                                    100
                                )
                            );

                        }

                        window.scrollTo(
                            0,
                            0
                        );

                    }
                    """
                )

            except Exception as e:

                print(
                    "Desktop scroll warning: "
                    f"{e}"
                )

            desktop_page.wait_for_timeout(
                500
            )

            # -------------------------------------------------
            # SCREENSHOT
            # -------------------------------------------------

            desktop_page.screenshot(
                path=str(desktop_path),
                full_page=True
            )

            print(
                "Desktop screenshot saved: "
                f"{desktop_path}"
            )

            desktop_page.close()

            desktop_context.close()

            # =================================================
            # MOBILE
            # =================================================

            print()
            print(
                "Capturing mobile screenshot..."
            )

            mobile_context = browser.new_context(
                viewport={
                    "width": 390,
                    "height": 844
                },

                is_mobile=True,

                has_touch=True,

                device_scale_factor=1
            )

            mobile_page = (
                mobile_context.new_page()
            )

            mobile_page.set_default_navigation_timeout(
                25000
            )

            try:

                mobile_page.goto(
                    website,
                    wait_until="domcontentloaded",
                    timeout=25000
                )

                print(
                    "Mobile page loaded."
                )

            except PlaywrightTimeoutError:

                print(
                    "Warning: Mobile navigation "
                    "timed out. Continuing..."
                )

            except Exception as e:

                print(
                    "Warning: Mobile navigation "
                    f"issue: {e}"
                )

            # -------------------------------------------------
            # SHORT WAIT
            # -------------------------------------------------

            mobile_page.wait_for_timeout(
                1500
            )

            # -------------------------------------------------
            # LIMITED SCROLL
            # -------------------------------------------------

            try:

                mobile_page.evaluate(
                    """
                    async () => {

                        const maxScrolls = 8;
                        const distance = 600;

                        for (
                            let i = 0;
                            i < maxScrolls;
                            i++
                        ) {

                            window.scrollBy(
                                0,
                                distance
                            );

                            await new Promise(
                                resolve =>
                                setTimeout(
                                    resolve,
                                    100
                                )
                            );

                        }

                        window.scrollTo(
                            0,
                            0
                        );

                    }
                    """
                )

            except Exception as e:

                print(
                    "Mobile scroll warning: "
                    f"{e}"
                )

            mobile_page.wait_for_timeout(
                500
            )

            # -------------------------------------------------
            # SCREENSHOT
            # -------------------------------------------------

            mobile_page.screenshot(
                path=str(mobile_path),
                full_page=True
            )

            print(
                "Mobile screenshot saved: "
                f"{mobile_path}"
            )

            mobile_page.close()

            mobile_context.close()

            browser.close()

            browser = None

        # ====================================================
        # VERIFY SCREENSHOTS
        # ====================================================

        if not desktop_path.exists():

            raise RuntimeError(
                "Desktop screenshot was not created."
            )

        if not mobile_path.exists():

            raise RuntimeError(
                "Mobile screenshot was not created."
            )

        # ====================================================
        # EXTRACT DESKTOP FEATURES
        # ====================================================

        print()
        print(
            "Extracting desktop features..."
        )

        desktop_features = extract_features(
            desktop_path
        )

        print(
            "Desktop features: "
            f"{len(desktop_features)}"
        )

        # ====================================================
        # EXTRACT MOBILE FEATURES
        # ====================================================

        print(
            "Extracting mobile features..."
        )

        mobile_features = extract_features(
            mobile_path
        )

        print(
            "Mobile features: "
            f"{len(mobile_features)}"
        )

        # ====================================================
        # COMBINE FEATURES
        # ====================================================

        combined_features = {}

        for name, value in desktop_features.items():

            combined_features[
                f"desktop_{name}"
            ] = value

        for name, value in mobile_features.items():

            combined_features[
                f"mobile_{name}"
            ] = value

        print(
            "Combined features: "
            f"{len(combined_features)}"
        )

        # ====================================================
        # CREATE MODEL INPUT
        # ====================================================

        X = create_feature_array(
            combined_features,
            advanced_screenshot_feature_names
        )

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        encoded_prediction = (
            advanced_screenshot_model
            .predict(X)[0]
        )

        # ====================================================
        # DECODE LABEL
        # ====================================================

        try:

            prediction = (
                advanced_screenshot_label_encoder
                .inverse_transform(
                    [encoded_prediction]
                )[0]
            )

        except Exception:

            prediction = encoded_prediction

        quality = str(
            prediction
        ).upper()

        # ====================================================
        # VISUAL SCORE
        # ====================================================

        if quality == "GOOD":

            score = 90

        elif quality == "AVERAGE":

            score = 60

        else:

            score = 30

        result = {

            "success": True,

            "website": website,

            "visual_score": score,

            "visual_quality": quality,

            "features_used": len(
                advanced_screenshot_feature_names
            ),

            "input_type": (
                "desktop + mobile screenshots"
            )

        }

        # ====================================================
        # CONFIDENCE + PROBABILITIES
        # ====================================================

        if hasattr(
            advanced_screenshot_model,
            "predict_proba"
        ):

            probabilities = (
                advanced_screenshot_model
                .predict_proba(X)[0]
            )

            result["confidence"] = round(

                float(
                    np.max(
                        probabilities
                    )
                ) * 100,

                2

            )

            if hasattr(
                advanced_screenshot_label_encoder,
                "classes_"
            ):

                probability_labels = (
                    advanced_screenshot_label_encoder
                    .classes_
                )

            elif hasattr(
                advanced_screenshot_model,
                "classes_"
            ):

                probability_labels = (
                    advanced_screenshot_model
                    .classes_
                )

            else:

                probability_labels = [

                    str(i)

                    for i in range(
                        len(probabilities)
                    )

                ]

            result["probabilities"] = {

                str(label).upper():

                round(
                    float(probability) * 100,
                    2
                )

                for label, probability

                in zip(
                    probability_labels,
                    probabilities
                )

            }

        # ====================================================
        # COMPLETE
        # ====================================================

        print()
        print("=" * 70)
        print("VISUAL AUDIT COMPLETE")
        print("=" * 70)

        print(
            f"Visual Score    : "
            f"{result['visual_score']}/100"
        )

        print(
            f"Visual Quality  : "
            f"{result['visual_quality']}"
        )

        print(
            f"Features Used   : "
            f"{result['features_used']}"
        )

        print(
            f"Input Type      : "
            f"{result['input_type']}"
        )

        if "confidence" in result:

            print(
                f"Confidence      : "
                f"{result['confidence']}%"
            )

        print("=" * 70)

        return result

    except Exception as e:

        print()
        print(
            "=" * 70
        )

        print(
            "VISUAL AUDIT ERROR:"
        )

        print(
            str(e)
        )

        print(
            "=" * 70
        )

        return {

            "success": False,

            "website": website,

            "error": str(e)

        }

    finally:

        if browser is not None:

            try:

                browser.close()

            except Exception:

                pass


# ============================================================
# COMPLETE WEBSITE AUDIT
# ============================================================

@app.post("/audit")
def audit_website(
    data: AuditRequest
):

    print()
    print("=" * 70)
    print("COMPLETE WEBSITE AUDIT")
    print("=" * 70)

    features = data.features or {}

    print(
        f"Website: {data.website}"
    )

    # ========================================================
    # FEATURE COLLECTION
    # ========================================================

    ux_features = {}
    visual_features = {}
    feature_collection_errors = []

    if data.website:

        # ====================================================
        # 1A. COLLECT REAL UX FEATURES FROM HTML
        # ====================================================

        print()
        print("Collecting real UX features from website HTML...")

        try:

            html = collect_ux.get_html(
                data.website
            )

            if html:

                ux_features = collect_ux.extract_ux_features(
                    html,
                    data.website
                )

                print(
                    f"UX features collected: "
                    f"{len(ux_features)}"
                )

            else:

                feature_collection_errors.append(
                    "Could not retrieve website HTML for UX extraction."
                )

                print(
                    "WARNING: Website HTML could not be retrieved."
                )

        except Exception as e:

            feature_collection_errors.append(
                f"UX feature extraction failed: {str(e)}"
            )

            print(
                f"UX extraction error: {e}"
            )

        # ====================================================
        # 1B. COLLECT REAL VISUAL RAW FEATURES
        # ====================================================

        print()
        print(
            "Collecting real Visual Raw features with Playwright..."
        )

        try:

            with sync_playwright() as p:

                browser = p.chromium.launch(
                    headless=True
                )

                desktop = browser.new_context(
                    viewport={
                        "width": 1440,
                        "height": 1000
                    }
                )

                mobile = browser.new_context(
                    viewport={
                        "width": 390,
                        "height": 844
                    },
                    is_mobile=True,
                    device_scale_factor=1
                )

                desktop_page = desktop.new_page()
                mobile_page = mobile.new_page()

                # --------------------------------------------
                # Desktop
                # --------------------------------------------

                try:

                    desktop_page.goto(
                        data.website,
                        wait_until="domcontentloaded",
                        timeout=30000
                    )

                    desktop_page.wait_for_timeout(
                        1500
                    )

                    desktop_data = analyze_desktop(
                        desktop_page
                    )

                except Exception as e:

                    desktop_data = {}

                    feature_collection_errors.append(
                        f"Desktop visual extraction failed: {str(e)}"
                    )

                    print(
                        f"Desktop extraction error: {e}"
                    )

                # --------------------------------------------
                # Mobile
                # --------------------------------------------

                try:

                    mobile_page.goto(
                        data.website,
                        wait_until="domcontentloaded",
                        timeout=30000
                    )

                    mobile_page.wait_for_timeout(
                        1500
                    )

                    mobile_data = analyze_mobile(
                        mobile_page
                    )

                except Exception as e:

                    mobile_data = {}

                    feature_collection_errors.append(
                        f"Mobile visual extraction failed: {str(e)}"
                    )

                    print(
                        f"Mobile extraction error: {e}"
                    )

                # --------------------------------------------
                # Combine desktop + mobile
                # --------------------------------------------

                visual_features = {
                    **desktop_data,
                    **{
                        f"mobile_{key}": value
                        for key, value in mobile_data.items()
                        if key != "viewport_width"
                    }
                }

                desktop.close()
                mobile.close()
                browser.close()

                print(
                    f"Visual Raw features collected: "
                    f"{len(visual_features)}"
                )

        except Exception as e:

            feature_collection_errors.append(
                f"Visual feature collection failed: {str(e)}"
            )

            print(
                f"Visual feature collection error: {e}"
            )

    else:

        print(
            "No website supplied. "
            "Using supplied features only."
        )

    # ========================================================
    # 2. PAGE SPEED
    # ========================================================

    print()
    print("2. Running PageSpeed ML...")

    pagespeed_features = {}
    pagespeed_collection_error = None

    if data.website:
        print("Collecting real PageSpeed metrics...")

        try:
            import collect_pagespeed

            pagespeed_data = collect_pagespeed.get_pagespeed_data(
                data.website
            )

            if pagespeed_data:
                pagespeed_result = collect_pagespeed.extract_features(
                    data.website,
                    pagespeed_data
                )

                # Keep only the numeric ML features.
                # The "url" field is metadata and is not a model feature.
                pagespeed_features = {
                    key: value
                    for key, value in pagespeed_result.items()
                    if key != "url"
                }

                print(
                    f"PageSpeed features collected: "
                    f"{len(pagespeed_features)}"
                )

            else:
                pagespeed_collection_error = (
                    "PageSpeed API returned no data."
                )

                print(
                    "WARNING: PageSpeed API returned no data."
                )

        except Exception as e:
            pagespeed_collection_error = str(e)

            feature_collection_errors.append(
                f"PageSpeed feature collection failed: {str(e)}"
            )

            print(
                f"PageSpeed collection error: {e}"
            )

    # --------------------------------------------------------
    # Run PageSpeed ML using REAL metrics when available.
    # --------------------------------------------------------

    if pagespeed_features:
        performance = final_audit.predict_pagespeed(
            pagespeed_features
        )

        performance["features_used"] = len(
            [
                feature
                for feature in pagespeed_feature_names
                if feature in pagespeed_features
            ]
        )

        performance["missing_features"] = [
            feature
            for feature in pagespeed_feature_names
            if feature not in pagespeed_features
        ]

        performance["input_type"] = (
            "real Google PageSpeed metrics"
        )

        if pagespeed_collection_error:
            performance["collection_error"] = (
                pagespeed_collection_error
            )

    else:
        # ----------------------------------------------------
        # Fallback for requests that do not contain a website
        # or when the PageSpeed API could not be reached.
        # ----------------------------------------------------

        performance = final_audit.predict_pagespeed(
            features
        )

        performance["features_used"] = len(
            [
                feature
                for feature in pagespeed_feature_names
                if feature in features
            ]
        )

        performance["missing_features"] = [
            feature
            for feature in pagespeed_feature_names
            if feature not in features
        ]

        performance["input_type"] = (
            "supplied features fallback"
        )

        if pagespeed_collection_error:
            performance["collection_error"] = (
                pagespeed_collection_error
            )

    print(
        f"Performance: "
        f"{performance['score']}/100 "
        f"({performance['quality']})"
    )

    print(
        f"PageSpeed model features used: "
        f"{performance['features_used']}/"
        f"{len(pagespeed_feature_names)}"
    )

    # ========================================================
    # 3. CLEAN UX
    # ========================================================

    print()
    print("3. Running Clean UX ML...")

    if ux_features:

        ux = final_audit.predict_ux(
            ux_features
        )

        ux["features_used"] = len(
            [
                feature
                for feature in ux_feature_names
                if feature in ux_features
            ]
        )

        ux["missing_features"] = [
            feature
            for feature in ux_feature_names
            if feature not in ux_features
        ]

        ux["input_type"] = "real website HTML"

    else:

        ux = final_audit.predict_ux(
            features
        )

        ux["features_used"] = 0

        ux["missing_features"] = list(
            ux_feature_names
        )

        ux["input_type"] = "supplied features fallback"

    print(
        f"UX: "
        f"{ux['score']}/100 "
        f"({ux['quality']})"
    )

    print(
        f"UX model features used: "
        f"{ux['features_used']}/{len(ux_feature_names)}"
    )

    # ========================================================
    # 4. VISUAL RAW
    # ========================================================

    print()
    print("4. Running Visual Raw ML...")

    if visual_features:

        visual_raw = final_audit.predict_visual(
            visual_features
        )

        visual_raw["features_used"] = len(
            [
                feature
                for feature in visual_feature_names
                if feature in visual_features
            ]
        )

        visual_raw["missing_features"] = [
            feature
            for feature in visual_feature_names
            if feature not in visual_features
        ]

        visual_raw["input_type"] = (
            "real website desktop + mobile DOM features"
        )

    else:

        visual_raw = final_audit.predict_visual(
            features
        )

        visual_raw["features_used"] = 0

        visual_raw["missing_features"] = list(
            visual_feature_names
        )

        visual_raw["input_type"] = (
            "supplied features fallback"
        )

    print(
        f"Visual Raw: "
        f"{visual_raw['score']}/100 "
        f"({visual_raw['quality']})"
    )

    print(
        f"Visual Raw model features used: "
        f"{visual_raw['features_used']}/"
        f"{len(visual_feature_names)}"
    )

    # ========================================================
    # 5. ADVANCED SCREENSHOT / VISUAL AUDIT
    # ========================================================

    print()
    print(
        "5. Running Desktop + Mobile Visual ML..."
    )

    if data.website:

        visual_result = visual_audit(
            VisualAuditRequest(
                website=data.website
            )
        )

        if visual_result.get("success"):

            advanced_screenshot = {

                "score": visual_result[
                    "visual_score"
                ],

                "quality": visual_result[
                    "visual_quality"
                ],

                "features_used": visual_result.get(
                    "features_used",
                    64
                ),

                "input_type": visual_result.get(
                    "input_type",
                    "desktop + mobile screenshots"
                ),

                "confidence": visual_result.get(
                    "confidence"
                ),

                "probabilities": visual_result.get(
                    "probabilities"
                )

            }

        else:

            print(
                "Visual screenshot audit failed."
            )

            advanced_screenshot = {

                "score": 60,

                "quality": "AVERAGE",

                "features_used": 0,

                "input_type": (
                    "desktop + mobile screenshots"
                ),

                "error": visual_result.get(
                    "error",
                    "Visual audit failed"
                )

            }

    else:

        print(
            "No website supplied. "
            "Using feature-based screenshot model."
        )

        advanced_screenshot = (
            final_audit.predict_advanced_screenshot(
                features
            )
        )

    print(
        f"Advanced Visual: "
        f"{advanced_screenshot['score']}/100 "
        f"({advanced_screenshot['quality']})"
    )

    # ========================================================
    # 6. RULE-BASED AUDIT
    # ========================================================

    print()
    print(
        "6. Running website rules..."
    )

    audit_features = {
        **features,
        **ux_features
    }

    rules = final_audit.run_audit_rules(
        audit_features
    )

    print(
        f"Rule issues: "
        f"{rules['issue_count']}"
    )

    # ========================================================
    # 7. FINAL SCORE
    # ========================================================

    print()
    print(
        "7. Calculating final score..."
    )

    final = final_audit.calculate_final_score(

        performance_score=(
            performance["score"]
        ),

        ux_quality=(
            ux["quality"]
        ),

        visual_score=(
            advanced_screenshot["score"]
        ),

        issue_count=(
            rules["issue_count"]
        )

    )

    print(
        f"FINAL SCORE: "
        f"{final['final_score']}/100 "
        f"({final['status']})"
    )

    print("=" * 70)
    print("COMPLETE WEBSITE AUDIT FINISHED")
    print("=" * 70)

    # ========================================================
    # COMPLETE RESPONSE
    # ========================================================

    return {

        "success": True,

        "performance": performance,

        "ux": ux,

        "visual_raw": visual_raw,

        "advanced_screenshot": (
            advanced_screenshot
        ),

        "rules": rules,

        "final": final,

        "feature_collection": {

            "ux_features_count": len(
                ux_features
            ),

            "ux_model_features_required": len(
                ux_feature_names
            ),

            "ux_model_features_used": len(
                [
                    feature
                    for feature in ux_feature_names
                    if feature in ux_features
                ]
            ),

            "ux_missing_features": [
                feature
                for feature in ux_feature_names
                if feature not in ux_features
            ],

            "visual_features_count": len(
                visual_features
            ),

            "visual_model_features_required": len(
                visual_feature_names
            ),

            "visual_model_features_used": len(
                [
                    feature
                    for feature in visual_feature_names
                    if feature in visual_features
                ]
            ),

            "visual_missing_features": [
                feature
                for feature in visual_feature_names
                if feature not in visual_features
            ],

            "errors": feature_collection_errors

        }

    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )


