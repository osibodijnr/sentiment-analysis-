import gradio as gr
import pickle
import re
import json
import hashlib
import os
import tempfile
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob


# ===============================================================
# COLOR PALETTE
# ===============================================================

COLOR_NAVY = "#0F172A"
COLOR_SLATE = "#334155"
COLOR_BLUE = "#3B82F6"
COLOR_BLUE_DARK = "#2563EB"
COLOR_BLUE_LIGHT = "#EFF6FF"

COLOR_BG = "#F8FAFC"
COLOR_CARD = "#FFFFFF"
COLOR_BORDER = "#E2E8F0"

COLOR_TEXT = "#0F172A"
COLOR_MUTED = "#64748B"

COLOR_POSITIVE = "#22C55E"
COLOR_NEGATIVE = "#EF4444"
COLOR_NEUTRAL = "#F59E0B"

COLOR_SUBJECTIVE = "#3B82F6"
COLOR_OBJECTIVE = "#94A3B8"

COLOR_SENTIMENT_MAP = {
    "Positive": COLOR_POSITIVE,
    "Negative": COLOR_NEGATIVE,
    "Neutral": COLOR_NEUTRAL,
}


# ===============================================================
# CUSTOM CSS
# ===============================================================

CUSTOM_CSS = f"""
:root {{
    --navy: {COLOR_NAVY};
    --slate: {COLOR_SLATE};
    --blue: {COLOR_BLUE};
    --blue-dark: {COLOR_BLUE_DARK};
    --blue-light: {COLOR_BLUE_LIGHT};
    --bg: {COLOR_BG};
    --card: {COLOR_CARD};
    --border: {COLOR_BORDER};
    --text: {COLOR_TEXT};
    --muted: {COLOR_MUTED};
}}

body {{
    background: var(--bg) !important;
}}

.gradio-container {{
    background: var(--bg) !important;
    max-width: 1250px !important;
    margin: auto !important;
}}


/* ============================================================
   GENERAL CARDS
   ============================================================ */

.gr-group,
.block {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05) !important;
}}


/* ============================================================
   MAIN HEADER
   ============================================================ */

#app-header {{
    background: linear-gradient(
        135deg,
        #0F172A 0%,
        #1E3A5F 100%
    ) !important;

    border: none !important;
    border-radius: 20px !important;
    padding: 30px 34px !important;
    margin-bottom: 22px !important;

    box-shadow:
        0 12px 30px rgba(15, 23, 42, 0.14) !important;
}}

#app-header h1 {{
    color: white !important;
    font-size: 2rem !important;
    margin-bottom: 8px !important;
}}

#app-header p {{
    color: #CBD5E1 !important;
    font-size: 1rem !important;
}}


/* ============================================================
   WELCOME BAR
   ============================================================ */

#welcome-bar {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

#welcome-bar p {{
    color: var(--slate) !important;
    font-weight: 600 !important;
}}


/* ============================================================
   BUTTONS
   ============================================================ */

button.primary,
.primary {{
    background: var(--blue-dark) !important;
    border-color: var(--blue-dark) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}

button.primary:hover,
.primary:hover {{
    background: #1D4ED8 !important;
}}

button.secondary,
.secondary {{
    background: white !important;
    border: 1px solid var(--border) !important;
    color: var(--slate) !important;
    border-radius: 10px !important;
}}

button.secondary:hover,
.secondary:hover {{
    background: #F8FAFC !important;
}}


/* ============================================================
   INPUTS
   ============================================================ */

textarea,
input[type=text],
input[type=password],
input[type=number],
input[type=file],
.wrap textarea {{
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: white !important;
}}

textarea:focus,
input:focus {{
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
}}


/* ============================================================
   TABS
   ============================================================ */

.tab-nav {{
    border-bottom: 1px solid var(--border) !important;
    gap: 5px !important;
}}

.tab-nav button {{
    border-radius: 9px 9px 0 0 !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}}

.tab-nav button.selected {{
    color: var(--slate) !important;
    border-bottom: 3px solid var(--blue) !important;
    background: white !important;
}}


/* ============================================================
   AUTH CARD
   ============================================================ */

#auth-card {{
    max-width: 460px !important;
    margin: 60px auto !important;
    background: white !important;
    border-radius: 20px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 15px 40px rgba(15, 23, 42, 0.10) !important;
    padding: 18px !important;
}}

#auth-card h1 {{
    color: var(--navy) !important;
    text-align: center !important;
}}


/* ============================================================
   YOUTUBE FEATURED SECTION
   ============================================================ */

#youtube-feature {{
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 28px !important;
    margin: 10px 0 28px 0 !important;

    box-shadow:
        0 10px 28px rgba(15, 23, 42, 0.07) !important;
}}

.youtube-heading {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 8px;
}}

.youtube-icon {{
    width: 48px;
    height: 48px;
    border-radius: 14px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: var(--blue-light);
    color: var(--blue-dark);

    font-size: 22px;
    font-weight: bold;
}}

.youtube-heading h2 {{
    color: var(--navy) !important;
    margin: 0 !important;
    font-size: 1.6rem !important;
}}

.youtube-heading p {{
    color: var(--muted) !important;
    margin: 5px 0 0 0 !important;
}}


/* ============================================================
   SECTION HEADINGS
   ============================================================ */

.section-title {{
    color: var(--navy);
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 5px;
}}

.section-description {{
    color: var(--muted);
    margin-bottom: 15px;
}}


/* ============================================================
   RESULT CARDS
   ============================================================ */

.result-card {{
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px !important;
}}


/* ============================================================
   FOOTER
   ============================================================ */

#footer {{
    color: var(--muted) !important;
    text-align: center !important;
    padding: 25px 10px !important;
    font-size: 0.9rem !important;
}}


/* ============================================================
   DATAFRAME
   ============================================================ */

.dataframe {{
    border-radius: 12px !important;
    overflow: hidden !important;
}}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 700px) {{
    #app-header {{
        padding: 22px !important;
    }}

    #app-header h1 {{
        font-size: 1.5rem !important;
    }}

    #youtube-feature {{
        padding: 18px !important;
    }}

    .youtube-heading h2 {{
        font-size: 1.25rem !important;
    }}
}}
"""


# ===============================================================
# NLTK SETUP
# ===============================================================

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ===============================================================
# LOAD TRAINED MODEL
# ===============================================================

try:
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    FEATURE_NAMES = vectorizer.get_feature_names_out()
    COEF = model.coef_[0]

except FileNotFoundError as e:
    raise SystemExit(
        f"Could not find model/vectorizer file ({e.filename}). "
        "Make sure svm_model.pkl and tfidf_vectorizer.pkl "
        "are in the same folder as this script."
    )


# ===============================================================
# USER ACCOUNTS
# ===============================================================

USERS_FILE = "users.json"


def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(password):
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def register_user(username, password, confirm_password):

    username = (username or "").strip()

    if not username or not password:
        return "Please fill in both a username and a password.", False

    if len(username) < 3:
        return "Username must be at least 3 characters.", False

    if len(password) < 6:
        return "Password must be at least 6 characters.", False

    if password != confirm_password:
        return "Passwords do not match.", False

    users = load_users()

    if username in users:
        return (
            "That username is already taken. Try logging in instead.",
            False
        )

    users[username] = hash_password(password)

    save_users(users)

    return (
        f"Account created for '{username}'. You can now log in.",
        True
    )


def authenticate_user(username, password):

    username = (username or "").strip()

    users = load_users()

    stored_hash = users.get(username)

    if stored_hash is None:
        return False, "No account found with that username."

    if stored_hash != hash_password(password):
        return False, "Incorrect password."

    return True, "Login successful."


# ===============================================================
# SENTIMENT SETTINGS
# ===============================================================

NEUTRAL_THRESHOLD = 0.15

LABEL_DISPLAY = {
    "Positive": "Positive 😊",
    "Negative": "Negative 😞",
    "Neutral": "Neutral 😐",
}


def classify_sentiment(
    decision_score,
    threshold=NEUTRAL_THRESHOLD
):

    if decision_score > threshold:
        return "Positive"

    elif decision_score < -threshold:
        return "Negative"

    return "Neutral"


def compute_three_way_scores(
    decision_score,
    threshold=NEUTRAL_THRESHOLD
):

    pos_raw = max(decision_score, 0.0)
    neg_raw = max(-decision_score, 0.0)

    neutral_raw = max(
        threshold - abs(decision_score),
        0.0
    )

    total = (
        pos_raw +
        neg_raw +
        neutral_raw
    )

    if total == 0:
        return {
            "Positive": 0.0,
            "Neutral": 1.0,
            "Negative": 0.0,
        }

    return {
        "Positive": pos_raw / total,
        "Neutral": neutral_raw / total,
        "Negative": neg_raw / total,
    }


# ===============================================================
# TEXT PREPROCESSING
# ===============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words
        and len(t) > 1
    ]

    return " ".join(tokens)


# ===============================================================
# MODEL EXPLAINABILITY
# ===============================================================

def get_top_contributions(
    cleaned_text,
    top_n=10
):

    features = vectorizer.transform(
        [cleaned_text]
    )

    nonzero_idx = features.nonzero()[1]

    if len(nonzero_idx) == 0:
        return []

    contributions = [
        (
            FEATURE_NAMES[i],
            features[0, i] * COEF[i]
        )
        for i in nonzero_idx
    ]

    contributions.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return contributions[:top_n]


def get_subjectivity(raw_text):

    score = TextBlob(
        raw_text
    ).sentiment.subjectivity

    label = (
        "Subjective (opinion)"
        if score >= 0.5
        else "Objective (fact-stating)"
    )

    return label, round(score, 3)


def make_explanation_plot(contributions):

    if not contributions:
        return None

    words = [
        c[0]
        for c in contributions
    ][::-1]

    values = [
        c[1]
        for c in contributions
    ][::-1]

    colors = [
        COLOR_POSITIVE
        if v > 0
        else COLOR_NEGATIVE
        for v in values
    ]

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.barh(
        words,
        values,
        color=colors
    )

    ax.axvline(
        0,
        color=COLOR_SLATE,
        linewidth=0.8
    )

    ax.set_xlabel(
        "Influence on prediction"
    )

    ax.set_title(
        "Words That Influenced the Prediction",
        color=COLOR_NAVY
    )

    fig.tight_layout()

    return fig


# ===============================================================
# SINGLE TEXT ANALYSIS
# ===============================================================

def predict_sentiment(text):

    if not text or not text.strip():

        return (
            "Please enter some text.",
            None,
            "",
            None,
            ""
        )

    cleaned = clean_text(text)

    if not cleaned.strip():

        return (
            "Input had no usable words after cleaning.",
            None,
            cleaned,
            None,
            ""
        )

    features = vectorizer.transform(
        [cleaned]
    )

    decision_score = model.decision_function(
        features
    )[0]

    sentiment = classify_sentiment(
        decision_score
    )

    label = LABEL_DISPLAY[
        sentiment
    ]

    scores = compute_three_way_scores(
        decision_score
    )

    contributions = get_top_contributions(
        cleaned
    )

    plot = make_explanation_plot(
        contributions
    )

    subj_label, subj_score = get_subjectivity(
        text
    )

    subjectivity_display = (
        f"{subj_label} "
        f"(score: {subj_score})"
    )

    return (
        label,
        scores,
        cleaned,
        plot,
        subjectivity_display
    )


def clear_single():

    return (
        "",
        "",
        None,
        "",
        None,
        ""
    )


# ===============================================================
# BATCH ANALYSIS
# ===============================================================

def find_date_column(df_in):

    for col in df_in.columns:

        if any(
            key in col.strip().lower()
            for key in [
                "date",
                "time",
                "created"
            ]
        ):

            return col

    return None


def make_dashboard(results_df):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8)
    )

    # -----------------------------------------------------------
    # Sentiment distribution
    # -----------------------------------------------------------

    sent_counts = (
        results_df[
            "predicted_sentiment"
        ].value_counts()
    )

    colors1 = [
        COLOR_SENTIMENT_MAP.get(
            lbl,
            "#888888"
        )
        for lbl in sent_counts.index
    ]

    axes[0, 0].bar(
        sent_counts.index,
        sent_counts.values,
        color=colors1
    )

    axes[0, 0].set_title(
        f"Sentiment Distribution "
        f"(n={len(results_df)})",
        color=COLOR_NAVY
    )

    axes[0, 0].set_ylabel(
        "Number of posts"
    )

    for i, v in enumerate(
        sent_counts.values
    ):

        axes[0, 0].text(
            i,
            v,
            str(v),
            ha="center",
            va="bottom"
        )

    # -----------------------------------------------------------
    # Subjectivity
    # -----------------------------------------------------------

    subj_counts = (
        results_df[
            "subjectivity"
        ].value_counts()
    )

    colors2 = [
        COLOR_SUBJECTIVE
        if "Subjective" in lbl
        else COLOR_OBJECTIVE
        for lbl in subj_counts.index
    ]

    axes[0, 1].bar(
        subj_counts.index,
        subj_counts.values,
        color=colors2
    )

    axes[0, 1].set_title(
        "Subjectivity Distribution",
        color=COLOR_NAVY
    )

    axes[0, 1].set_ylabel(
        "Number of posts"
    )

    axes[0, 1].tick_params(
        axis="x",
        labelrotation=10
    )

    for i, v in enumerate(
        subj_counts.values
    ):

        axes[0, 1].text(
            i,
            v,
            str(v),
            ha="center",
            va="bottom"
        )

    # -----------------------------------------------------------
    # Confidence
    # -----------------------------------------------------------

    axes[1, 0].hist(
        results_df[
            "confidence_proxy"
        ],
        bins=10,
        color=COLOR_BLUE,
        edgecolor="white"
    )

    axes[1, 0].set_title(
        "Confidence Proxy Distribution",
        color=COLOR_NAVY
    )

    axes[1, 0].set_xlabel(
        "Confidence (0-1)"
    )

    axes[1, 0].set_ylabel(
        "Number of posts"
    )

    # -----------------------------------------------------------
    # Most frequent words
    # -----------------------------------------------------------

    counter = Counter()

    for t in results_df[
        "cleaned_text"
    ]:

        counter.update(
            t.split()
        )

    top_words = counter.most_common(10)

    if top_words:

        words, freqs = zip(
            *top_words[::-1]
        )

        axes[1, 1].barh(
            words,
            freqs,
            color=COLOR_SLATE
        )

        axes[1, 1].set_title(
            "Most Frequent Words",
            color=COLOR_NAVY
        )

        axes[1, 1].set_xlabel(
            "Frequency"
        )

    else:

        axes[1, 1].axis("off")

    fig.tight_layout()

    return fig


def make_trend_chart(
    dates_raw,
    results_df
):

    try:

        dates = pd.to_datetime(
            dates_raw,
            errors="coerce"
        )

    except Exception:

        return None

    dates = pd.Series(
        dates
    ).reset_index(drop=True)

    if dates.isna().all():
        return None

    trend_df = pd.DataFrame({
        "date": dates.dt.date,
        "sentiment":
            results_df[
                "predicted_sentiment"
            ].values
    }).dropna()

    if trend_df[
        "date"
    ].nunique() < 2:

        return None

    daily_counts = (
        trend_df
        .groupby(
            ["date", "sentiment"]
        )
        .size()
        .unstack(fill_value=0)
    )

    daily_pct = (
        daily_counts
        .div(
            daily_counts.sum(axis=1),
            axis=0
        )
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(9, 3.5)
    )

    for label in [
        "Positive",
        "Neutral",
        "Negative"
    ]:

        if label in daily_pct.columns:

            ax.plot(
                daily_pct.index,
                daily_pct[label],
                marker="o",
                label=label,
                color=COLOR_SENTIMENT_MAP[
                    label
                ]
            )

    ax.set_title(
        "Sentiment Mix Over Time",
        color=COLOR_NAVY
    )

    ax.set_ylabel(
        "% of posts"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.legend()

    fig.autofmt_xdate()

    fig.tight_layout()

    return fig


def analyze_texts(
    texts,
    dates_raw=None
):

    if not texts:

        return (
            None,
            None,
            None,
            None,
            "No text to analyze."
        )

    cleaned_texts = [
        clean_text(t)
        for t in texts
    ]

    features = vectorizer.transform(
        cleaned_texts
    )

    scores = model.decision_function(
        features
    )

    labels = [
        classify_sentiment(s)
        for s in scores
    ]

    confidences = [
        min(abs(s) / 2, 1.0)
        for s in scores
    ]

    subjectivity_results = [
        get_subjectivity(t)
        for t in texts
    ]

    results_df = pd.DataFrame({

        "text": texts,

        "predicted_sentiment":
            labels,

        "confidence_proxy":
            [
                round(c, 3)
                for c in confidences
            ],

        "subjectivity":
            [
                s[0]
                for s in subjectivity_results
            ],

        "subjectivity_score":
            [
                s[1]
                for s in subjectivity_results
            ],

        "cleaned_text":
            cleaned_texts,
    })

    dashboard_fig = make_dashboard(
        results_df
    )

    trend_fig = None
    trend_note = ""

    if dates_raw is not None:

        trend_fig = make_trend_chart(
            dates_raw,
            results_df
        )

        if trend_fig is None:

            trend_note = (
                " Date information was found, "
                "but there were not enough distinct "
                "dates for a trend chart."
            )

    export_df = results_df.drop(
        columns=["cleaned_text"]
    )

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv",
        mode="w",
        newline=""
    )

    export_df.to_csv(
        tmp.name,
        index=False
    )

    tmp.close()

    pos_count = labels.count(
        "Positive"
    )

    neg_count = labels.count(
        "Negative"
    )

    neu_count = labels.count(
        "Neutral"
    )

    subj_count = sum(
        1
        for s in subjectivity_results
        if "Subjective" in s[0]
    )

    summary_text = (
        f"### Analysis Summary\n\n"
        f"**{len(results_df)}** posts analyzed\n\n"
        f"🟢 Positive: **{pos_count}**  \n"
        f"🟠 Neutral: **{neu_count}**  \n"
        f"🔴 Negative: **{neg_count}**  \n"
        f"🔵 Subjective: **{subj_count}**"
        f"{trend_note}"
    )

    return (
        export_df,
        dashboard_fig,
        trend_fig,
        tmp.name,
        summary_text
    )


def run_batch(
    file_obj,
    pasted_text
):

    texts = []
    dates_raw = None

    if file_obj is not None:

        df_in = pd.read_csv(
            file_obj.name
        )

        text_col = None

        for col in df_in.columns:

            if col.strip().lower() == "text":

                text_col = col
                break

        if text_col is None:

            text_col = df_in.columns[0]

        texts = (
            df_in[text_col]
            .astype(str)
            .tolist()
        )

        date_col = find_date_column(
            df_in
        )

        if date_col is not None:

            dates_raw = df_in[
                date_col
            ]

    elif (
        pasted_text
        and pasted_text.strip()
    ):

        texts = [
            line.strip()
            for line
            in pasted_text.split("\n")
            if line.strip()
        ]

    if not texts:

        return (
            None,
            None,
            None,
            None,
            "Please upload a CSV or paste at least one line of text."
        )

    return analyze_texts(
        texts,
        dates_raw
    )


def clear_batch():

    return (
        None,
        "",
        None,
        None,
        None,
        None,
        ""
    )


# ===============================================================
# YOUTUBE
# ===============================================================

def extract_video_id(
    url_or_id
):

    url_or_id = (
        url_or_id or ""
    ).strip()

    patterns = [

        r"(?:v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/)([A-Za-z0-9_-]{11})",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url_or_id
        )

        if match:

            return match.group(1)

    if re.fullmatch(
        r"[A-Za-z0-9_-]{11}",
        url_or_id
    ):

        return url_or_id

    return None


def fetch_youtube_comments(
    video_url_or_id,
    api_key,
    max_comments=100
):

    try:

        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

    except ImportError:

        return (
            [],
            [],
            "The google-api-python-client package is not installed. "
            "Run: pip install google-api-python-client"
        )

    if not api_key or not api_key.strip():

        return (
            [],
            [],
            "Please enter your YouTube Data API key."
        )

    video_id = extract_video_id(
        video_url_or_id
    )

    if video_id is None:

        return (
            [],
            [],
            "Couldn't parse a YouTube video ID. "
            "Paste a full YouTube URL or an 11-character video ID."
        )

    try:

        youtube = build(
            "youtube",
            "v3",
            developerKey=api_key.strip()
        )

    except Exception as e:

        return (
            [],
            [],
            f"Could not initialize YouTube API client: {e}"
        )

    texts = []
    dates = []

    next_page_token = None

    try:

        while len(texts) < max_comments:

            request = (
                youtube
                .commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(
                        100,
                        max_comments - len(texts)
                    ),
                    pageToken=next_page_token,
                    textFormat="plainText",
                    order="relevance",
                )
            )

            response = request.execute()

            for item in response.get(
                "items",
                []
            ):

                snippet = (
                    item["snippet"]
                    ["topLevelComment"]
                    ["snippet"]
                )

                texts.append(
                    snippet["textDisplay"]
                )

                dates.append(
                    snippet["publishedAt"]
                )

            next_page_token = response.get(
                "nextPageToken"
            )

            if not next_page_token:
                break

    except HttpError as e:

        status = (
            getattr(
                e,
                "status_code",
                None
            )
            or getattr(
                e.resp,
                "status",
                None
            )
        )

        if status == 403:

            return (
                texts,
                dates,
                "YouTube API returned 403 Forbidden. "
                "Check your API key, API restrictions, quota, "
                "or whether comments are disabled."
            )

        if status == 404:

            return (
                [],
                [],
                "Video not found. Check the URL or video ID."
            )

        return (
            texts,
            dates,
            f"YouTube API error: {e}"
        )

    except Exception as e:

        return (
            texts,
            dates,
            f"Unexpected error while fetching comments: {e}"
        )

    if not texts:

        return (
            [],
            [],
            "No comments were returned. "
            "Comments may be disabled or the video may not have comments."
        )

    return (
        texts,
        dates,
        ""
    )


def run_youtube_batch(
    video_url,
    api_key,
    max_comments
):

    try:

        max_comments = int(
            max_comments
        )

    except Exception:

        max_comments = 100

    max_comments = max(
        1,
        min(
            max_comments,
            500
        )
    )

    texts, dates, error = (
        fetch_youtube_comments(
            video_url,
            api_key,
            max_comments
        )
    )

    if error:

        return (
            None,
            None,
            None,
            None,
            f"### Error\n\n{error}"
        )

    (
        export_df,
        dashboard_fig,
        trend_fig,
        csv_path,
        summary_text
    ) = analyze_texts(
        texts,
        dates
    )

    summary_text = (
        f"### YouTube Analysis Complete\n\n"
        f"Fetched **{len(texts)} comments**.\n\n"
        + summary_text.replace(
            "### Analysis Summary\n\n",
            ""
        )
    )

    return (
        export_df,
        dashboard_fig,
        trend_fig,
        csv_path,
        summary_text
    )


def clear_youtube():

    return (
        "",
        "",
        100,
        "",
        None,
        None,
        None,
        None
    )


# ===============================================================
# AUTH HANDLERS
# ===============================================================

def handle_login(
    username,
    password
):

    success, message = authenticate_user(
        username,
        password
    )

    if success:

        return (
            True,
            username,
            gr.update(
                visible=False
            ),
            gr.update(
                visible=True
            ),
            f"Welcome, **{username}**!",
            "",
        )

    return (
        False,
        "",
        gr.update(
            visible=True
        ),
        gr.update(
            visible=False
        ),
        "",
        message,
    )


def handle_signup(
    username,
    password,
    confirm
):

    message, success = register_user(
        username,
        password,
        confirm
    )

    if success:

        return (
            True,
            username,
            gr.update(
                visible=False
            ),
            gr.update(
                visible=True
            ),
            f"Welcome, **{username}**!",
            message,
        )

    return (
        False,
        "",
        gr.update(
            visible=True
        ),
        gr.update(
            visible=False
        ),
        "",
        message,
    )


def handle_logout():

    return (
        False,
        "",
        gr.update(
            visible=True
        ),
        gr.update(
            visible=False
        ),
        "",
        "",
        "",
        "",
        "",
    )


# ===============================================================
# GRADIO INTERFACE
# ===============================================================

with gr.Blocks(
    title="Social Media Sentiment Analyzer",

    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate"
    ),

    css=CUSTOM_CSS
) as demo:

    logged_in_state = gr.State(False)

    username_state = gr.State("")


    # ===========================================================
    # LOGIN
    # ===========================================================

    with gr.Group(
        visible=True,
        elem_id="auth-card"
    ) as auth_section:

        gr.Markdown(
            """
            # 💬 Social Media Sentiment Analyzer

            Analyze public opinion from social media text
            using machine learning.
            """
        )

        with gr.Tabs():

            with gr.Tab("Login"):

                login_username = gr.Textbox(
                    label="Username"
                )

                login_password = gr.Textbox(
                    label="Password",
                    type="password"
                )

                login_btn = gr.Button(
                    "Log In",
                    variant="primary"
                )

                login_message = gr.Textbox(
                    label="",
                    interactive=False,
                    show_label=False
                )

            with gr.Tab("Sign Up"):

                signup_username = gr.Textbox(
                    label="Choose a username"
                )

                signup_password = gr.Textbox(
                    label="Choose a password",
                    type="password"
                )

                signup_confirm = gr.Textbox(
                    label="Confirm password",
                    type="password"
                )

                signup_btn = gr.Button(
                    "Create Account",
                    variant="primary"
                )

                signup_message = gr.Textbox(
                    label="",
                    interactive=False,
                    show_label=False
                )


    # ===========================================================
    # MAIN APPLICATION
    # ===========================================================

    with gr.Group(
        visible=False
    ) as main_app:

        # -------------------------------------------------------
        # TOP BAR
        # -------------------------------------------------------

        with gr.Row():

            welcome_text = gr.Markdown(
                "",
                elem_id="welcome-bar"
            )

            logout_btn = gr.Button(
                "Log Out",
                size="sm",
                variant="secondary"
            )


        # -------------------------------------------------------
        # HEADER
        # -------------------------------------------------------

        gr.Markdown(
            """
            # Social Media Sentiment Analyzer

            Analyze public opinion from social media content
            using **TF-IDF and Support Vector Machine (SVM)**.
            """

            ,
            elem_id="app-header"
        )


        # =======================================================
        # FEATURED YOUTUBE SECTION
        # =======================================================

        with gr.Group(
            elem_id="youtube-feature"
        ):

            gr.Markdown(
                """
                <div class="youtube-heading">
                    <div class="youtube-icon">▶</div>
                    <div>
                        <h2>YouTube Comment Analysis</h2>
                        <p>
                            Analyze public reactions to a YouTube video
                            using the trained sentiment classification model.
                        </p>
                    </div>
                </div>
                """
            )

            with gr.Row():

                with gr.Column(
                    scale=3
                ):

                    yt_url_input = gr.Textbox(
                        label="YouTube Video URL",
                        placeholder=(
                            "Paste a YouTube video URL here..."
                        )
                    )

                with gr.Column(
                    scale=1
                ):

                    yt_max_comments = gr.Number(
                        label="Maximum Comments",
                        value=100,
                        precision=0
                    )

            yt_api_key_input = gr.Textbox(
                label="YouTube Data API Key",
                type="password",
                value=os.environ.get(
                    "YOUTUBE_API_KEY",
                    ""
                ),
                placeholder=(
                    "Enter your YouTube Data API key"
                )
            )

            gr.Markdown(
                """
                **Privacy note:** Your API key is used only
                for the current request and is not saved by this application.
                """
            )

            yt_btn = gr.Button(
                "▶ Analyze YouTube Comments",
                variant="primary",
                size="lg"
            )

            yt_summary = gr.Markdown(
                "Enter a YouTube video URL and click **Analyze YouTube Comments**."
            )

            with gr.Row():

                yt_dashboard = gr.Plot(
                    label="Sentiment Distribution"
                )

                yt_trend = gr.Plot(
                    label="Sentiment Over Time"
                )

            yt_table = gr.Dataframe(
                label="Analyzed Comments",
                wrap=True
            )

            yt_download = gr.File(
                label="Download Analysis Results"
            )

            yt_btn.click(
                fn=run_youtube_batch,

                inputs=[
                    yt_url_input,
                    yt_api_key_input,
                    yt_max_comments
                ],

                outputs=[
                    yt_table,
                    yt_dashboard,
                    yt_trend,
                    yt_download,
                    yt_summary
                ]
            )


        # =======================================================
        # OTHER ANALYSIS TOOLS
        # =======================================================

        gr.Markdown(
            """
            <div class="section-title">
                Other Analysis Tools
            </div>

            <div class="section-description">
                Analyze individual social media posts or process
                larger datasets using CSV files.
            </div>
            """
        )


        with gr.Tabs():

            # ===================================================
            # SINGLE POST
            # ===================================================

            with gr.Tab(
                "Analyze a Post"
            ):

                with gr.Row():

                    with gr.Column():

                        text_input = gr.Textbox(
                            label="Enter a tweet or social media post",
                            placeholder=(
                                "e.g. I absolutely love "
                                "how smooth this update feels!"
                            ),
                            lines=5
                        )

                        with gr.Row():

                            analyze_btn = gr.Button(
                                "Analyze Sentiment",
                                variant="primary"
                            )

                            clear_btn = gr.Button(
                                "Clear"
                            )

                        gr.Examples(
                            examples=[
                                (
                                    "I absolutely love "
                                    "how smooth this update feels!"
                                ),
                                (
                                    "This is the worst service "
                                    "I've ever experienced."
                                ),
                                (
                                    "The meeting is scheduled "
                                    "for 9am tomorrow."
                                ),
                            ],

                            inputs=text_input
                        )

                    with gr.Column():

                        result_label = gr.Textbox(
                            label="Predicted Sentiment",
                            interactive=False
                        )

                        confidence_plot = gr.Label(
                            label=(
                                "Confidence Proxy "
                                "(distance from decision boundary)"
                            )
                        )

                        subjectivity_output = gr.Textbox(
                            label=(
                                "Subjectivity "
                                "(opinion vs. fact)"
                            ),
                            interactive=False
                        )

                        cleaned_output = gr.Textbox(
                            label=(
                                "Cleaned Text Used for Prediction"
                            ),
                            interactive=False
                        )

                explanation_plot = gr.Plot(
                    label="Why the Model Predicted This"
                )

                analyze_btn.click(
                    fn=predict_sentiment,

                    inputs=text_input,

                    outputs=[
                        result_label,
                        confidence_plot,
                        cleaned_output,
                        explanation_plot,
                        subjectivity_output
                    ]
                )

                text_input.submit(
                    fn=predict_sentiment,

                    inputs=text_input,

                    outputs=[
                        result_label,
                        confidence_plot,
                        cleaned_output,
                        explanation_plot,
                        subjectivity_output
                    ]
                )

                clear_btn.click(
                    fn=clear_single,

                    inputs=None,

                    outputs=[
                        text_input,
                        result_label,
                        confidence_plot,
                        cleaned_output,
                        explanation_plot,
                        subjectivity_output
                    ]
                )


            # ===================================================
            # BATCH ANALYSIS
            # ===================================================

            with gr.Tab(
                "Batch Analysis"
            ):

                gr.Markdown(
                    """
                    Upload a CSV containing social media text,
                    or paste multiple posts below.

                    If the CSV contains a date or timestamp column,
                    the system will also generate a sentiment trend chart.
                    """
                )

                with gr.Row():

                    with gr.Column():

                        csv_input = gr.File(
                            label="Upload CSV",
                            file_types=[".csv"]
                        )

                        batch_text_input = gr.Textbox(
                            label="Or Paste Posts",
                            lines=8,
                            placeholder=(
                                "I love this new update!\n"
                                "Worst experience ever.\n"
                                "Just another Monday."
                            )
                        )

                        with gr.Row():

                            batch_btn = gr.Button(
                                "Analyze Batch",
                                variant="primary"
                            )

                            batch_clear_btn = gr.Button(
                                "Clear"
                            )

                        batch_summary = gr.Markdown(
                            ""
                        )

                batch_dashboard = gr.Plot(
                    label="Batch Analysis Dashboard"
                )

                batch_trend = gr.Plot(
                    label="Trend Over Time"
                )

                batch_table = gr.Dataframe(
                    label="Results",
                    wrap=True
                )

                batch_download = gr.File(
                    label="Download Results"
                )

                batch_btn.click(
                    fn=run_batch,

                    inputs=[
                        csv_input,
                        batch_text_input
                    ],

                    outputs=[
                        batch_table,
                        batch_dashboard,
                        batch_trend,
                        batch_download,
                        batch_summary
                    ]
                )

                batch_clear_btn.click(
                    fn=clear_batch,

                    inputs=None,

                    outputs=[
                        csv_input,
                        batch_text_input,
                        batch_table,
                        batch_dashboard,
                        batch_trend,
                        batch_download,
                        batch_summary
                    ]
                )


        # =======================================================
        # FOOTER
        # =======================================================

        gr.Markdown(
            """
            ---

            <div id="footer">

            **Sentiment Analysis of Social Media Data Using Machine Learning Techniques**

            TF-IDF + Support Vector Machine (SVM)

            Neutral sentiment is assigned when the model's
            decision score falls within the configured threshold
            around the decision boundary.

            This system is a research prototype. Predictions may
            be inaccurate for sarcasm, complex negation, slang,
            or highly context-dependent text.

            Confidence values are distance-based heuristics,
            not calibrated probabilities.

            </div>
            """
        )


    # ===========================================================
    # AUTHENTICATION EVENTS
    # ===========================================================

    login_btn.click(

        fn=handle_login,

        inputs=[
            login_username,
            login_password
        ],

        outputs=[
            logged_in_state,
            username_state,
            auth_section,
            main_app,
            welcome_text,
            login_message
        ]
    )


    signup_btn.click(

        fn=handle_signup,

        inputs=[
            signup_username,
            signup_password,
            signup_confirm
        ],

        outputs=[
            logged_in_state,
            username_state,
            auth_section,
            main_app,
            welcome_text,
            signup_message
        ]
    )


    logout_btn.click(

        fn=handle_logout,

        inputs=None,

        outputs=[
            logged_in_state,
            username_state,
            auth_section,
            main_app,
            welcome_text,
            login_username,
            login_password,
            login_message
        ]
    )


# ===============================================================
# RUN APPLICATION
# ===============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7861
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )