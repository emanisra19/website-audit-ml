# ============================================================
# FINAL WEBSITE AUDIT SYSTEM
# ============================================================
#
# Components:
#
# 1. PageSpeed ML
# 2. Clean UX ML
# 3. Visual Quality ML
# 4. Rule-Based Audit
# 5. Advanced Screenshot Visual ML
# 6. Final Combined Website Score
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import joblib
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"

DATA_DIR = BASE_DIR / "data"


# ============================================================
# HELPER â€” LOAD FEATURE NAMES
# ============================================================

def load_feature_names(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

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


# ============================================================
# HELPER â€” LOAD MODEL
# ============================================================

def load_model(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return joblib.load(path)


# ============================================================
# PAGE SPEED MODEL
# ============================================================

PAGESPEED_MODEL_PATH = (
    MODELS_DIR /
    "pagespeed_website_score_model.joblib"
)

PAGESPEED_FEATURES_PATH = (
    MODELS_DIR /
    "pagespeed_feature_names.txt"
)


pagespeed_model = load_model(
    PAGESPEED_MODEL_PATH
)

pagespeed_feature_names = load_feature_names(
    PAGESPEED_FEATURES_PATH
)


print(
    "PageSpeed model loaded: "
    f"{type(pagespeed_model).__name__}"
)

print(
    "PageSpeed features loaded: "
    f"{len(pagespeed_feature_names)}"
)


# ============================================================
# PAGE SPEED PREDICTION
# ============================================================

def predict_pagespeed(features):

    model_features = {}

    for feature in pagespeed_feature_names:

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


    X = pd.DataFrame(
        [model_features],
        columns=pagespeed_feature_names
    )


    score = float(
        pagespeed_model.predict(X)[0]
    )


    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )


    if score >= 90:

        quality = "GOOD"

    elif score >= 50:

        quality = "AVERAGE"

    else:

        quality = "POOR"


    return {

        "score": round(
            score,
            2
        ),

        "quality": quality,

        "category": quality

    }


# ============================================================
# CLEAN UX MODEL
# ============================================================

UX_MODEL_PATH = (
    MODELS_DIR /
    "clean_ux_quality_model.joblib"
)

UX_FEATURES_PATH = (
    MODELS_DIR /
    "clean_ux_feature_names.txt"
)


ux_model = load_model(
    UX_MODEL_PATH
)

ux_feature_names = load_feature_names(
    UX_FEATURES_PATH
)


print(
    "Clean UX model loaded: "
    f"{type(ux_model).__name__}"
)

print(
    "Clean UX features loaded: "
    f"{len(ux_feature_names)}"
)


# ============================================================
# GENERIC MODEL INPUT BUILDER
# ============================================================

def build_model_dataframe(
    features,
    feature_names
):

    model_features = {}

    missing = []


    for feature in feature_names:

        if feature in features:

            value = features[feature]

        else:

            value = 0

            missing.append(feature)


        try:

            value = float(value)

        except (
            TypeError,
            ValueError
        ):

            value = 0.0


        model_features[feature] = value


    X = pd.DataFrame(
        [model_features],
        columns=feature_names
    )


    return X, missing


# ============================================================
# QUALITY LABEL NORMALIZATION
# ============================================================

def normalize_quality(label):

    label = str(
        label
    ).strip().upper()


    if "GOOD" in label:

        return "GOOD"


    if "AVERAGE" in label:

        return "AVERAGE"


    if "POOR" in label:

        return "POOR"


    return label


# ============================================================
# QUALITY TO SCORE
# ============================================================

def quality_to_score(quality):

    quality = normalize_quality(
        quality
    )


    if quality == "GOOD":

        return 90


    if quality == "AVERAGE":

        return 60


    if quality == "POOR":

        return 30


    return 0


# ============================================================
# CLEAN UX PREDICTION
# ============================================================

def predict_ux(features):

    X, missing = build_model_dataframe(
        features,
        ux_feature_names
    )


    prediction = ux_model.predict(X)[0]


    quality = normalize_quality(
        prediction
    )


    result = {

        "quality": quality,

        "score": quality_to_score(
            quality
        )

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
# VISUAL RAW MODEL
# ============================================================

VISUAL_MODEL_PATH = (
    MODELS_DIR /
    "visual_raw_quality_model.joblib"
)

VISUAL_FEATURES_PATH = (
    MODELS_DIR /
    "visual_raw_feature_names.txt"
)


visual_model = load_model(
    VISUAL_MODEL_PATH
)

visual_feature_names = load_feature_names(
    VISUAL_FEATURES_PATH
)


print(
    "Visual raw model loaded: "
    f"{type(visual_model).__name__}"
)

print(
    "Visual raw features loaded: "
    f"{len(visual_feature_names)}"
)


# ============================================================
# VISUAL PREDICTION
# ============================================================

def predict_visual(features):

    X, missing = build_model_dataframe(
        features,
        visual_feature_names
    )


    prediction = visual_model.predict(X)[0]


    quality = normalize_quality(
        prediction
    )


    result = {

        "quality": quality,

        "score": quality_to_score(
            quality
        )

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
# ADVANCED SCREENSHOT MODEL
# ============================================================
#
# IMPORTANT:
#
# advanced_screenshot_quality_model_v3.joblib
# is NOT the sklearn model directly.
#
# It is a dictionary:
#
# {
#     "model": trained sklearn model,
#     "label_encoder": trained label encoder
# }
#
# ============================================================

ADVANCED_SCREENSHOT_MODEL_PATH = (
    MODELS_DIR /
    "advanced_screenshot_quality_model_v3.joblib"
)

ADVANCED_SCREENSHOT_FEATURES_PATH = (
    MODELS_DIR /
    "advanced_screenshot_feature_names_v3.txt"
)


advanced_screenshot_bundle = load_model(
    ADVANCED_SCREENSHOT_MODEL_PATH
)


# ============================================================
# EXTRACT MODEL + LABEL ENCODER
# ============================================================

if isinstance(
    advanced_screenshot_bundle,
    dict
):

    if "model" not in advanced_screenshot_bundle:

        raise KeyError(
            "Advanced screenshot model bundle "
            "does not contain 'model'."
        )


    if "label_encoder" not in advanced_screenshot_bundle:

        raise KeyError(
            "Advanced screenshot model bundle "
            "does not contain 'label_encoder'."
        )


    advanced_screenshot_model = (
        advanced_screenshot_bundle["model"]
    )


    advanced_screenshot_label_encoder = (
        advanced_screenshot_bundle["label_encoder"]
    )

else:

    raise TypeError(
        "Expected advanced screenshot model "
        "to be a dictionary containing "
        "'model' and 'label_encoder'."
    )


advanced_screenshot_feature_names = (
    load_feature_names(
        ADVANCED_SCREENSHOT_FEATURES_PATH
    )
)


print(
    "Advanced screenshot model loaded: "
    f"{type(advanced_screenshot_model).__name__}"
)

print(
    "Advanced screenshot label encoder loaded: "
    f"{type(advanced_screenshot_label_encoder).__name__}"
)

print(
    "Advanced screenshot features loaded: "
    f"{len(advanced_screenshot_feature_names)}"
)


# ============================================================
# ADVANCED SCREENSHOT PREDICTION
# ============================================================

def predict_advanced_screenshot(
    screenshot_features
):

    X, missing = build_model_dataframe(
        screenshot_features,
        advanced_screenshot_feature_names
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    encoded_prediction = (
        advanced_screenshot_model
        .predict(X)[0]
    )


    # --------------------------------------------------------
    # DECODE LABEL
    # --------------------------------------------------------

    try:

        prediction = (
            advanced_screenshot_label_encoder
            .inverse_transform(
                [encoded_prediction]
            )[0]
        )

    except Exception:

        # Safety fallback in case the model
        # already returns a text label.

        prediction = encoded_prediction


    quality = normalize_quality(
        prediction
    )


    result = {

        "quality": quality,

        "score": quality_to_score(
            quality
        ),

        "missing_features": missing

    }


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # DECODE PROBABILITY LABELS
        # ----------------------------------------------------

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
# RULE-BASED AUDIT
# ============================================================

def run_audit_rules(features):

    issues = []


    # --------------------------------------------------------
    # SEO
    # --------------------------------------------------------

    if not features.get(
        "has_title",
        0
    ):

        issues.append(
            "Missing page title"
        )


    if not features.get(
        "has_meta_description",
        0
    ):

        issues.append(
            "Missing meta description"
        )


    if not features.get(
        "has_h1",
        0
    ):

        issues.append(
            "Missing H1 heading"
        )


    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    if not features.get(
        "has_navigation",
        0
    ):

        issues.append(
            "Missing navigation"
        )


    # --------------------------------------------------------
    # CONTACT
    # --------------------------------------------------------

    if not features.get(
        "has_phone",
        0
    ):

        issues.append(
            "Missing phone number"
        )


    if not features.get(
        "has_email",
        0
    ):

        issues.append(
            "Missing email"
        )


    if not features.get(
        "has_address",
        0
    ):

        issues.append(
            "Missing address"
        )


    # --------------------------------------------------------
    # CTA
    # --------------------------------------------------------

    if not features.get(
        "has_clear_cta",
        0
    ):

        issues.append(
            "Missing clear CTA"
        )


    # --------------------------------------------------------
    # BOOKING
    # --------------------------------------------------------

    if not features.get(
        "has_booking_or_appointment",
        0
    ):

        issues.append(
            "No booking or appointment option"
        )


    # --------------------------------------------------------
    # IMAGES
    # --------------------------------------------------------

    if features.get(
        "has_images",
        0
    ):

        image_alt_ratio = float(
            features.get(
                "image_alt_ratio",
                0
            )
        )


        if image_alt_ratio < 0.80:

            issues.append(
                "Poor image alt-text coverage"
            )


    # --------------------------------------------------------
    # MOBILE
    # --------------------------------------------------------

    if features.get(
        "mobile_horizontal_overflow",
        0
    ):

        issues.append(
            "Mobile horizontal overflow"
        )


    if features.get(
        "mobile_small_buttons",
        0
    ):

        issues.append(
            "Small mobile buttons"
        )


    if features.get(
        "mobile_overflowing_images",
        0
    ):

        issues.append(
            "Mobile overflowing images"
        )


    return {

        "issue_count": len(
            issues
        ),

        "issues": issues

    }


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_final_score(
    performance_score,
    ux_quality,
    visual_score,
    issue_count
):

    ux_score = quality_to_score(
        ux_quality
    )


    # --------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------
    #
    # Performance = 40%
    # UX          = 25%
    # Visual      = 25%
    # Rules       = 10%
    #
    # --------------------------------------------------------

    rule_score = max(
        0,
        100 - (
            issue_count * 10
        )
    )


    final_score = (

        performance_score * 0.40

        +

        ux_score * 0.25

        +

        visual_score * 0.25

        +

        rule_score * 0.10

    )


    final_score = max(
        0,
        min(
            100,
            final_score
        )
    )


    if final_score >= 80:

        status = "GOOD"

    elif final_score >= 50:

        status = "AVERAGE"

    else:

        status = "POOR"


    return {

        "final_score": round(
            final_score,
            2
        ),

        "status": status

    }


# ============================================================
# PRINT SCENARIO RESULT
# ============================================================

def print_scenario_result(
    title,
    performance,
    ux,
    visual,
    rules,
    final
):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)


    print()

    print(
        f"PageSpeed Score : "
        f"{performance['score']}/100"
    )

    print(
        f"PageSpeed       : "
        f"{performance['quality']}"
    )


    print()

    print(
        f"UX Score        : "
        f"{ux['score']}/100"
    )

    print(
        f"UX Quality      : "
        f"{ux['quality']}"
    )


    if "confidence" in ux:

        print(
            f"UX Confidence   : "
            f"{ux['confidence']:.2f}%"
        )


    print()

    print(
        f"Visual Score    : "
        f"{visual['score']}/100"
    )

    print(
        f"Visual Quality  : "
        f"{visual['quality']}"
    )


    if "confidence" in visual:

        print(
            f"Visual Confidence: "
            f"{visual['confidence']:.2f}%"
        )


    print()

    print(
        f"Rule Issues     : "
        f"{rules['issue_count']}"
    )


    if rules["issues"]:

        print()

        print(
            "Issues:"
        )

        for issue in rules["issues"]:

            print(
                f"  - {issue}"
            )


    print()

    print(
        f"FINAL SCORE     : "
        f"{final['final_score']}/100"
    )

    print(
        f"FINAL STATUS    : "
        f"{final['status']}"
    )


# ============================================================
# SCENARIO TESTING
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 70)

    print(
        "FINAL WEBSITE AUDIT SCENARIO TEST"
    )

    print("=" * 70)


    # ========================================================
    # SCENARIO 1 â€” GOOD WEBSITE
    # ========================================================

    good_features = {

        # ----------------------------------------------------
        # PAGE SPEED
        # ----------------------------------------------------

        "first_contentful_paint": 1000,
        "largest_contentful_paint": 1800,
        "speed_index": 2000,
        "interactive": 2500,
        "total_blocking_time": 100,
        "cumulative_layout_shift": 0.03,
        "server_response_time": 200,
        "bootup_time": 400,
        "total_resources": 50,
        "total_transfer_size": 900000,
        "script_resources": 12,
        "image_resources": 20,
        "stylesheet_resources": 6,
        "font_resources": 3,

        # ----------------------------------------------------
        # CLEAN UX
        # ----------------------------------------------------

        "heading_hierarchy_score": 0.95,
        "information_architecture_score": 0.95,
        "cta_presence_score": 1.0,
        "conversion_ux_signal": 0.95,
        "content_density_signal": 0.85,
        "mobile_content_density_signal": 0.85,
        "visual_clutter_score": 0.90,
        "mobile_visual_clutter_score": 0.90,
        "visual_balance_score": 0.90,
        "mobile_visual_balance_score": 0.90,
        "mobile_interaction_signal": 1.0,
        "mobile_small_button_penalty": 1.0,
        "mobile_image_overflow_penalty": 1.0,
        "contactability_ux_signal": 0.90,

        # ----------------------------------------------------
        # RULES
        # ----------------------------------------------------

        "has_title": 1,
        "has_meta_description": 1,
        "has_h1": 1,
        "has_navigation": 1,
        "has_phone": 1,
        "has_email": 1,
        "has_address": 1,
        "has_clear_cta": 1,
        "has_booking_or_appointment": 1,
        "has_images": 1,
        "image_alt_ratio": 0.95,

        "mobile_horizontal_overflow": 0,
        "mobile_small_buttons": 0,
        "mobile_overflowing_images": 0,

        # ----------------------------------------------------
        # VISUAL
        # ----------------------------------------------------

        "links": 20,
        "buttons": 8,
        "cta_count": 3,
        "images": 20,
        "images_with_alt": 19,
        "large_images": 2,
        "headings": 8,
        "h1_count": 1,
        "h2_count": 4,
        "inputs": 3,
        "nav_elements": 1,
        "sections": 8,
        "font_count": 3,
        "color_count": 8,
        "word_count": 700,
        "page_height": 5000,
        "viewport_width": 390,
        "scroll_width": 390,
        "visible_elements": 120,
        "mobile_scroll_width": 390,
        "mobile_mobile_buttons": 8,
        "mobile_mobile_images": 20

    }


    performance_good = predict_pagespeed(
        good_features
    )

    ux_good = predict_ux(
        good_features
    )

    visual_good = predict_visual(
        good_features
    )

    rules_good = run_audit_rules(
        good_features
    )

    final_good = calculate_final_score(

        performance_good["score"],

        ux_good["quality"],

        visual_good["score"],

        rules_good["issue_count"]

    )


    print_scenario_result(

        "SCENARIO 1 â€” GOOD WEBSITE",

        performance_good,

        ux_good,

        visual_good,

        rules_good,

        final_good

    )


    # ========================================================
    # SCENARIO 2 â€” AVERAGE WEBSITE
    # ========================================================

    average_features = {

        # ----------------------------------------------------
        # PAGE SPEED
        # ----------------------------------------------------

        "first_contentful_paint": 3500,
        "largest_contentful_paint": 5200,
        "speed_index": 5000,
        "interactive": 6500,
        "total_blocking_time": 500,
        "cumulative_layout_shift": 0.15,
        "server_response_time": 700,
        "bootup_time": 1200,
        "total_resources": 120,
        "total_transfer_size": 2500000,
        "script_resources": 35,
        "image_resources": 45,
        "stylesheet_resources": 12,
        "font_resources": 6,

        # ----------------------------------------------------
        # CLEAN UX
        # ----------------------------------------------------

        "heading_hierarchy_score": 0.75,
        "information_architecture_score": 0.75,
        "cta_presence_score": 0.80,
        "conversion_ux_signal": 0.70,
        "content_density_signal": 0.70,
        "mobile_content_density_signal": 0.65,
        "visual_clutter_score": 0.70,
        "mobile_visual_clutter_score": 0.65,
        "visual_balance_score": 0.70,
        "mobile_visual_balance_score": 0.65,
        "mobile_interaction_signal": 0.70,
        "mobile_small_button_penalty": 0.60,
        "mobile_image_overflow_penalty": 0.70,
        "contactability_ux_signal": 0.75,

        # ----------------------------------------------------
        # RULES
        # ----------------------------------------------------

        "has_title": 1,
        "has_meta_description": 1,
        "has_h1": 1,
        "has_navigation": 1,
        "has_phone": 1,
        "has_email": 1,
        "has_address": 1,
        "has_clear_cta": 1,
        "has_booking_or_appointment": 0,
        "has_images": 1,
        "image_alt_ratio": 0.70,

        "mobile_horizontal_overflow": 0,
        "mobile_small_buttons": 2,
        "mobile_overflowing_images": 1,

        # ----------------------------------------------------
        # VISUAL
        # ----------------------------------------------------

        "links": 45,
        "buttons": 15,
        "cta_count": 2,
        "images": 50,
        "images_with_alt": 35,
        "large_images": 12,
        "headings": 14,
        "h1_count": 1,
        "h2_count": 8,
        "inputs": 5,
        "nav_elements": 1,
        "sections": 15,
        "font_count": 6,
        "color_count": 15,
        "word_count": 1500,
        "page_height": 9000,
        "viewport_width": 390,
        "scroll_width": 390,
        "visible_elements": 250,
        "mobile_scroll_width": 390,
        "mobile_mobile_buttons": 15,
        "mobile_mobile_images": 50

    }


    performance_average = predict_pagespeed(
        average_features
    )

    ux_average = predict_ux(
        average_features
    )

    visual_average = predict_visual(
        average_features
    )

    rules_average = run_audit_rules(
        average_features
    )

    final_average = calculate_final_score(

        performance_average["score"],

        ux_average["quality"],

        visual_average["score"],

        rules_average["issue_count"]

    )


    print_scenario_result(

        "SCENARIO 2 â€” AVERAGE WEBSITE",

        performance_average,

        ux_average,

        visual_average,

        rules_average,

        final_average

    )


    # ========================================================
    # SCENARIO 3 â€” POOR WEBSITE
    # ========================================================

    poor_features = {

        # ----------------------------------------------------
        # PAGE SPEED
        # ----------------------------------------------------

        "first_contentful_paint": 6500,
        "largest_contentful_paint": 9000,
        "speed_index": 8500,
        "interactive": 11000,
        "total_blocking_time": 1500,
        "cumulative_layout_shift": 0.45,
        "server_response_time": 1800,
        "bootup_time": 3000,
        "total_resources": 300,
        "total_transfer_size": 7000000,
        "script_resources": 100,
        "image_resources": 120,
        "stylesheet_resources": 30,
        "font_resources": 15,

        # ----------------------------------------------------
        # CLEAN UX
        # ----------------------------------------------------

        "heading_hierarchy_score": 0.30,
        "information_architecture_score": 0.25,
        "cta_presence_score": 0.25,
        "conversion_ux_signal": 0.20,
        "content_density_signal": 0.30,
        "mobile_content_density_signal": 0.25,
        "visual_clutter_score": 0.25,
        "mobile_visual_clutter_score": 0.20,
        "visual_balance_score": 0.30,
        "mobile_visual_balance_score": 0.20,
        "mobile_interaction_signal": 0.25,
        "mobile_small_button_penalty": 0.20,
        "mobile_image_overflow_penalty": 0.20,
        "contactability_ux_signal": 0.30,

        # ----------------------------------------------------
        # RULES
        # ----------------------------------------------------

        "has_title": 0,
        "has_meta_description": 0,
        "has_h1": 0,
        "has_navigation": 0,
        "has_phone": 0,
        "has_email": 0,
        "has_address": 0,
        "has_clear_cta": 0,
        "has_booking_or_appointment": 0,
        "has_images": 1,
        "image_alt_ratio": 0.20,

        "mobile_horizontal_overflow": 1,
        "mobile_small_buttons": 8,
        "mobile_overflowing_images": 15,

        # ----------------------------------------------------
        # VISUAL
        # ----------------------------------------------------

        "links": 100,
        "buttons": 40,
        "cta_count": 0,
        "images": 120,
        "images_with_alt": 20,
        "large_images": 60,
        "headings": 30,
        "h1_count": 0,
        "h2_count": 5,
        "inputs": 15,
        "nav_elements": 0,
        "sections": 40,
        "font_count": 15,
        "color_count": 35,
        "word_count": 4000,
        "page_height": 18000,
        "viewport_width": 390,
        "scroll_width": 520,
        "visible_elements": 600,
        "mobile_scroll_width": 520,
        "mobile_mobile_buttons": 40,
        "mobile_mobile_images": 120

    }


    performance_poor = predict_pagespeed(
        poor_features
    )

    ux_poor = predict_ux(
        poor_features
    )

    visual_poor = predict_visual(
        poor_features
    )

    rules_poor = run_audit_rules(
        poor_features
    )

    final_poor = calculate_final_score(

        performance_poor["score"],

        ux_poor["quality"],

        visual_poor["score"],

        rules_poor["issue_count"]

    )


    print_scenario_result(

        "SCENARIO 3 â€” POOR WEBSITE",

        performance_poor,

        ux_poor,

        visual_poor,

        rules_poor,

        final_poor

    )


    # ========================================================
    # REAL WEBSITE SCREENSHOT VISUAL AUDIT
    # ========================================================

    print()

    print("=" * 70)

    print(
        "REAL WEBSITE SCREENSHOT VISUAL AUDIT"
    )

    print("=" * 70)


    real_website_id = "website_063"


    screenshot_csv = (
        DATA_DIR /
        "advanced_screenshot_features.csv"
    )


    try:

        if not screenshot_csv.exists():

            raise FileNotFoundError(
                f"Screenshot feature CSV not found: "
                f"{screenshot_csv}"
            )


        screenshot_df = pd.read_csv(
            screenshot_csv
        )


        website_df = screenshot_df[
            screenshot_df[
                "website_id"
            ].astype(str)
            == str(real_website_id)
        ].copy()


        if website_df.empty:

            raise ValueError(
                f"No screenshot data found "
                f"for {real_website_id}"
            )


        # ----------------------------------------------------
        # DESKTOP
        # ----------------------------------------------------

        desktop_df = website_df[
            website_df[
                "device"
            ].astype(str).str.lower()
            == "desktop"
        ].copy()


        # ----------------------------------------------------
        # MOBILE
        # ----------------------------------------------------

        mobile_df = website_df[
            website_df[
                "device"
            ].astype(str).str.lower()
            == "mobile"
        ].copy()


        if desktop_df.empty:

            raise ValueError(
                "Desktop screenshot features missing."
            )


        if mobile_df.empty:

            raise ValueError(
                "Mobile screenshot features missing."
            )


        # ----------------------------------------------------
        # BASE FEATURES
        # ----------------------------------------------------

        excluded_columns = {

            "website_id",
            "device",
            "filename"

        }


        base_features = [

            column

            for column in screenshot_df.columns

            if column not in excluded_columns

        ]


        # ----------------------------------------------------
        # DESKTOP FEATURES
        # ----------------------------------------------------

        desktop_features = (
            desktop_df[
                base_features
            ]
            .iloc[0]
            .to_frame()
            .T
        )


        desktop_features.index = [
            real_website_id
        ]


        desktop_features = (
            desktop_features
            .add_prefix("desktop_")
        )


        # ----------------------------------------------------
        # MOBILE FEATURES
        # ----------------------------------------------------

        mobile_features = (
            mobile_df[
                base_features
            ]
            .iloc[0]
            .to_frame()
            .T
        )


        mobile_features.index = [
            real_website_id
        ]


        mobile_features = (
            mobile_features
            .add_prefix("mobile_")
        )


        # ----------------------------------------------------
        # COMBINE DESKTOP + MOBILE
        # ----------------------------------------------------

        screenshot_features = (
            desktop_features.join(
                mobile_features
            )
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        screenshot_result = (
            predict_advanced_screenshot(
                screenshot_features.iloc[0].to_dict()
            )
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print()

        print(
            f"Website ID        : "
            f"{real_website_id}"
        )


        print(
            f"Visual Score      : "
            f"{screenshot_result['score']}/100"
        )


        print(
            f"Visual Quality    : "
            f"{screenshot_result['quality']}"
        )


        if "confidence" in screenshot_result:

            print(
                f"Visual Confidence : "
                f"{screenshot_result['confidence']:.2f}%"
            )


        if screenshot_result.get(
            "probabilities"
        ):

            print()

            print(
                "Visual probabilities:"
            )


            for (
                label,
                probability
            ) in screenshot_result[
                "probabilities"
            ].items():

                print(
                    f"  {label}: "
                    f"{probability:.2f}%"
                )


        missing = screenshot_result.get(
            "missing_features",
            []
        )


        if missing:

            print()

            print(
                "Warning: screenshot model "
                "features missing:"
            )


            for feature in missing:

                print(
                    f"  - {feature}"
                )


        print()

        print(
            "Advanced screenshot model "
            "prediction successful."
        )


    except Exception as e:

        print()

        print(
            "REAL SCREENSHOT AUDIT ERROR:"
        )

        print(
            str(e)
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "FINAL SUMMARY"
    )

    print("=" * 70)


    print(
        f"GOOD    â†’ "
        f"{final_good['final_score']}/100 "
        f"â†’ {final_good['status']}"
    )


    print(
        f"AVERAGE â†’ "
        f"{final_average['final_score']}/100 "
        f"â†’ {final_average['status']}"
    )


    print(
        f"POOR    â†’ "
        f"{final_poor['final_score']}/100 "
        f"â†’ {final_poor['status']}"
    )


    print()

    print("=" * 70)

    print(
        "TEST COMPLETE"
    )

    print("=" * 70)
