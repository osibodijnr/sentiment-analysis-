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

# ---------------------------------------------------------------
# Color palette — shared by the CSS theme below AND every matplotlib
# chart in this file, so the UI and the charts stay visually consistent.
# ---------------------------------------------------------------
COLOR_NAVY = "#1F3B57"       # primary / headers
COLOR_BG = "#EAF1F7"         # page background (very light blue/gray)
COLOR_CARD = "#FFFFFF"       # cards/panels
COLOR_BLUE = "#2563EB"       # primary buttons / accents
COLOR_BLUE_HOVER = "#1D4ED8"
COLOR_POSITIVE = "#16A34A"   # green
COLOR_NEGATIVE = "#DC2626"   # red
COLOR_NEUTRAL = "#D97706"    # amber/yellow
COLOR_BORDER = "#E2E8F0"
COLOR_SUBJECTIVE = "#2563EB"     # reuse accent blue — distinct from sentiment colors
COLOR_OBJECTIVE = "#94A3B8"      # neutral slate gray — distinct from sentiment colors
COLOR_SENTIMENT_MAP = {"Positive": COLOR_POSITIVE, "Negative": COLOR_NEGATIVE, "Neutral": COLOR_NEUTRAL}

CUSTOM_CSS = f"""
:root {{
    --color-navy: {COLOR_NAVY};
    --color-bg: {COLOR_BG};
    --color-card: {COLOR_CARD};
    --color-blue: {COLOR_BLUE};
    --color-blue-hover: {COLOR_BLUE_HOVER};
    --color-border: {COLOR_BORDER};
}}

.gradio-container {{
    background: var(--color-bg) !important;
}}

/* ---- Header banner ---- */
#app-header {{
    background: var(--color-navy) !important;
    border-radius: 16px !important;
    padding: 22px 28px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 4px 14px rgba(31, 59, 87, 0.20) !important;
}}
#app-header h1, #app-header p, #app-header strong, #app-header span {{
    color: #FFFFFF !important;
}}

/* ---- Cards / panels: groups & blocks get white bg, rounded corners, subtle shadow ---- */
.gr-group, .block {{
    background: var(--color-card) !important;
    border-radius: 14px !important;
    border: 1px solid var(--color-border) !important;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06) !important;
}}

/* ---- Buttons ---- */
button.primary, .primary {{
    background: var(--color-blue) !important;
    border-color: var(--color-blue) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
button.primary:hover, .primary:hover {{
    background: var(--color-blue-hover) !important;
}}
button.secondary, .secondary {{
    border-radius: 10px !important;
    border-color: var(--color-border) !important;
    color: var(--color-navy) !important;
    background: var(--color-card) !important;
}}

/* ---- Inputs: rounded, subtle border, blue focus ring ---- */
textarea, input[type=text], input[type=password], input[type=number], .wrap textarea {{
    border-radius: 10px !important;
    border: 1px solid var(--color-border) !important;
}}
textarea:focus, input:focus {{
    border-color: var(--color-blue) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}}

/* ---- Tabs: cleaner underline style instead of default boxed look ---- */
.tab-nav {{
    border-bottom: 1px solid var(--color-border) !important;
    gap: 4px !important;
}}
.tab-nav button {{
    border-radius: 10px 10px 0 0 !important;
    color: #64748B !important;
    font-weight: 500 !important;
}}
.tab-nav button.selected {{
    color: var(--color-navy) !important;
    border-bottom: 3px solid var(--color-blue) !important;
    background: var(--color-card) !important;
}}

/* ---- Login/Sign-up card: styled as its own centered panel ---- */
#auth-card {{
    max-width: 460px !important;
    margin: 32px auto !important;
    background: var(--color-card) !important;
    border-radius: 18px !important;
    border: 1px solid var(--color-border) !important;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.12) !important;
    padding: 12px !important;
}}
#auth-card h1 {{
    color: var(--color-navy) !important;
    text-align: center !important;
}}

/* ---- Welcome bar text ---- */
#welcome-bar p {{
    color: var(--color-navy) !important;
    font-weight: 600 !important;
    font-size: 1.05em !important;
}}
"""

# --- One-time NLTK setup (safe to leave in; skips if already downloaded) ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Load the trained model and vectorizer ---
try:
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    FEATURE_NAMES = vectorizer.get_feature_names_out()
    COEF = model.coef_[0]  # LinearSVC binary: shape (1, n_features) -> take row 0
except FileNotFoundError as e:
    raise SystemExit(
        f"Could not find model/vectorizer file ({e.filename}). "
        "Make sure svm_model.pkl and tfidf_vectorizer.pkl are in the same folder as this script."
    )

# ---------------------------------------------------------------
# User accounts (self-service sign-up + login)
# ---------------------------------------------------------------
# NOTE on scope/security: Gradio's built-in `auth=` parameter only supports a fixed
# login screen shown BEFORE the app loads — it has no concept of self-registration.
# To let people create their own accounts, this app instead builds its own login/
# sign-up screen inside the Blocks UI, backed by a small local JSON file.
#
# This is appropriate for a final-year demo, but be upfront about its limits if asked
# in your defense: passwords are SHA-256 hashed (not plaintext), but there's no salt,
# no rate-limiting, no session expiry, and anyone with access to the source could see
# the app's own UI before logging in (Gradio doesn't hide component definitions from
# the network tab). This is "keep casual visitors out and demonstrate the concept of
# authenticated access," not production-grade security.
USERS_FILE = 'users.json'


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


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
        return "That username is already taken. Try logging in instead.", False

    users[username] = hash_password(password)
    save_users(users)
    return f"Account created for '{username}'. You can now log in.", True


def authenticate_user(username, password):
    users = load_users()
    username = (username or "").strip()
    stored_hash = users.get(username)
    if stored_hash is None:
        return False, "No account found with that username."
    if stored_hash != hash_password(password):
        return False, "Incorrect password."
    return True, "Login successful."

# --- Neutral-class threshold (Section 3.3 of the proposal) ---
# The SVM itself is still a binary classifier (Positive/Negative). A prediction is
# relabelled "Neutral" when its decision_function score falls within this margin of
# the separating hyperplane — i.e. the model wasn't confident either way.
#
# 0.15 is a reasonable starting point given this SVM's C=0.1 (heavily regularized,
# so decision scores tend to be modest), but it is a hyperparameter you should
# justify empirically rather than take on faith: run decision_function() over your
# validation set, look at the distribution of |scores|, and pick a threshold that
# flags a defensible proportion (e.g. 10-20%) as neutral. Report whatever value you
# land on, and why, in your methodology write-up.
NEUTRAL_THRESHOLD = 0.15

LABEL_DISPLAY = {
    "Positive": "Positive 😊",
    "Negative": "Negative 😞",
    "Neutral": "Neutral 😐",
}


def classify_sentiment(decision_score, threshold=NEUTRAL_THRESHOLD):
    if decision_score > threshold:
        return "Positive"
    elif decision_score < -threshold:
        return "Negative"
    return "Neutral"


def compute_three_way_scores(decision_score, threshold=NEUTRAL_THRESHOLD):
    """Turn a single decision_function value into a 3-way display breakdown.
    This is a heuristic for the UI, not a calibrated probability — see the
    confidence_proxy note in the footer."""
    pos_raw = max(decision_score, 0.0)
    neg_raw = max(-decision_score, 0.0)
    neutral_raw = max(threshold - abs(decision_score), 0.0)
    total = pos_raw + neg_raw + neutral_raw
    if total == 0:
        return {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0}
    return {
        "Positive": pos_raw / total,
        "Neutral": neutral_raw / total,
        "Negative": neg_raw / total,
    }

# --- Same cleaning function used during training — must match exactly ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)


def get_top_contributions(cleaned_text, top_n=10):
    """Return the words in this text that most influenced the prediction,
    using the SVM's learned linear weights times each word's TF-IDF value."""
    features = vectorizer.transform([cleaned_text])
    nonzero_idx = features.nonzero()[1]
    if len(nonzero_idx) == 0:
        return []
    contributions = [(FEATURE_NAMES[i], features[0, i] * COEF[i]) for i in nonzero_idx]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]


def get_subjectivity(raw_text):
    """Subjectivity/objectivity is computed on the ORIGINAL text, not the
    aggressively cleaned version — TextBlob's lexicon relies on full words
    and sentence structure that stopword removal and lemmatization would
    otherwise strip out."""
    score = TextBlob(raw_text).sentiment.subjectivity  # 0.0 (objective) -> 1.0 (subjective)
    label = "Subjective (opinion)" if score >= 0.5 else "Objective (fact-stating)"
    return label, round(score, 3)


def make_explanation_plot(contributions):
    if not contributions:
        return None
    words = [c[0] for c in contributions][::-1]
    values = [c[1] for c in contributions][::-1]
    colors = [COLOR_POSITIVE if v > 0 else COLOR_NEGATIVE for v in values]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.barh(words, values, color=colors)
    ax.axvline(0, color=COLOR_NAVY, linewidth=0.8)
    ax.set_xlabel("Influence on prediction (→ positive)")
    ax.set_title("Words that most influenced this prediction", color=COLOR_NAVY)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------
# Single-text prediction with explainability
# ---------------------------------------------------------------
def predict_sentiment(text):
    if not text or not text.strip():
        return "Please enter some text.", None, "", None, ""

    cleaned = clean_text(text)
    if not cleaned.strip():
        return "Input had no usable words after cleaning — try adding more text.", None, "", None, ""

    features = vectorizer.transform([cleaned])
    decision_score = model.decision_function(features)[0]

    sentiment = classify_sentiment(decision_score)
    label = LABEL_DISPLAY[sentiment]
    scores = compute_three_way_scores(decision_score)

    contributions = get_top_contributions(cleaned)
    plot = make_explanation_plot(contributions)

    subj_label, subj_score = get_subjectivity(text)
    subjectivity_display = f"{subj_label} (score: {subj_score})"

    return label, scores, cleaned, plot, subjectivity_display


def clear_single():
    return "", "", None, "", None, ""


# ---------------------------------------------------------------
# Batch / trend analysis
# ---------------------------------------------------------------
def find_date_column(df_in):
    for col in df_in.columns:
        if any(key in col.strip().lower() for key in ["date", "time", "created"]):
            return col
    return None


def make_dashboard(results_df):
    """Build a 2x2 dashboard: sentiment counts, subjectivity split,
    confidence distribution, and the most frequent words overall."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1. Sentiment counts
    sent_counts = results_df["predicted_sentiment"].value_counts()
    colors1 = [COLOR_SENTIMENT_MAP.get(lbl, "#888888") for lbl in sent_counts.index]
    axes[0, 0].bar(sent_counts.index, sent_counts.values, color=colors1)
    axes[0, 0].set_title(f"Sentiment Distribution (n={len(results_df)})", color=COLOR_NAVY)
    axes[0, 0].set_ylabel("Number of posts")
    for i, v in enumerate(sent_counts.values):
        axes[0, 0].text(i, v, str(v), ha="center", va="bottom")

    # 2. Subjectivity counts
    subj_counts = results_df["subjectivity"].value_counts()
    colors2 = [COLOR_SUBJECTIVE if "Subjective" in lbl else COLOR_OBJECTIVE for lbl in subj_counts.index]
    axes[0, 1].bar(subj_counts.index, subj_counts.values, color=colors2)
    axes[0, 1].set_title("Subjectivity Distribution", color=COLOR_NAVY)
    axes[0, 1].set_ylabel("Number of posts")
    axes[0, 1].tick_params(axis='x', labelrotation=10)
    for i, v in enumerate(subj_counts.values):
        axes[0, 1].text(i, v, str(v), ha="center", va="bottom")

    # 3. Confidence distribution
    axes[1, 0].hist(results_df["confidence_proxy"], bins=10, color=COLOR_BLUE, edgecolor="white")
    axes[1, 0].set_title("Confidence Proxy Distribution", color=COLOR_NAVY)
    axes[1, 0].set_xlabel("Confidence (0-1)")
    axes[1, 0].set_ylabel("Number of posts")

    # 4. Top words overall
    counter = Counter()
    for t in results_df["cleaned_text"]:
        counter.update(t.split())
    top_words = counter.most_common(10)
    if top_words:
        words, freqs = zip(*top_words[::-1])
        axes[1, 1].barh(words, freqs, color=COLOR_NAVY)
        axes[1, 1].set_title("Most Frequent Words (this batch)", color=COLOR_NAVY)
        axes[1, 1].set_xlabel("Frequency")
    else:
        axes[1, 1].axis("off")

    fig.tight_layout()
    return fig


def make_trend_chart(dates_raw, results_df):
    try:
        dates = pd.to_datetime(dates_raw, errors="coerce")
    except Exception:
        return None
    dates = pd.Series(dates).reset_index(drop=True)
    if dates.isna().all():
        return None

    trend_df = pd.DataFrame({
        "date": dates.dt.date,
        "sentiment": results_df["predicted_sentiment"].values
    }).dropna()

    if trend_df["date"].nunique() < 2:
        return None

    daily_counts = trend_df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    daily_pct = daily_counts.div(daily_counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for label in ["Positive", "Neutral", "Negative"]:
        if label in daily_pct.columns:
            ax.plot(daily_pct.index, daily_pct[label], marker="o", label=label, color=COLOR_SENTIMENT_MAP[label])
    ax.set_title("Sentiment Mix Over Time", color=COLOR_NAVY)
    ax.set_ylabel("% of posts")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def analyze_texts(texts, dates_raw=None):
    """Shared core: run the sentiment/subjectivity pipeline over a list of texts
    and build the dashboard, optional trend chart, and downloadable CSV. Used by
    both the CSV/paste batch path and the YouTube comments path below."""
    if not texts:
        return None, None, None, None, "No text to analyze."

    cleaned_texts = [clean_text(t) for t in texts]
    features = vectorizer.transform(cleaned_texts)
    scores = model.decision_function(features)

    labels = [classify_sentiment(s) for s in scores]
    confidences = [min(abs(s) / 2, 1.0) for s in scores]
    subjectivity_results = [get_subjectivity(t) for t in texts]

    results_df = pd.DataFrame({
        "text": texts,
        "predicted_sentiment": labels,
        "confidence_proxy": [round(c, 3) for c in confidences],
        "subjectivity": [s[0] for s in subjectivity_results],
        "subjectivity_score": [s[1] for s in subjectivity_results],
        "cleaned_text": cleaned_texts,
    })

    dashboard_fig = make_dashboard(results_df)

    trend_fig = None
    trend_note = ""
    if dates_raw is not None:
        trend_fig = make_trend_chart(dates_raw, results_df)
        if trend_fig is None:
            trend_note = " (Date information found but not enough distinct dates for a trend chart.)"

    # Downloadable CSV (drop the internal cleaned_text helper column)
    export_df = results_df.drop(columns=["cleaned_text"])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    export_df.to_csv(tmp.name, index=False)

    pos_count = labels.count("Positive")
    neg_count = labels.count("Negative")
    neu_count = labels.count("Neutral")
    subj_count = sum(1 for s in subjectivity_results if "Subjective" in s[0])
    summary_text = (
        f"Analyzed {len(results_df)} posts — {pos_count} positive, {neu_count} neutral, "
        f"{neg_count} negative ({subj_count} flagged as subjective/opinion-based)."
    ) + trend_note

    return export_df, dashboard_fig, trend_fig, tmp.name, summary_text


def run_batch(file_obj, pasted_text):
    texts = []
    dates_raw = None

    if file_obj is not None:
        df_in = pd.read_csv(file_obj.name)
        text_col = None
        for col in df_in.columns:
            if col.strip().lower() == "text":
                text_col = col
                break
        if text_col is None:
            text_col = df_in.columns[0]
        texts = df_in[text_col].astype(str).tolist()
        date_col = find_date_column(df_in)
        if date_col is not None:
            dates_raw = df_in[date_col]
    elif pasted_text and pasted_text.strip():
        texts = [line.strip() for line in pasted_text.split("\n") if line.strip()]

    if not texts:
        return None, None, None, None, "Please upload a CSV or paste at least one line of text."

    return analyze_texts(texts, dates_raw)


def clear_batch():
    return None, "", None, None, None, None, ""


# ---------------------------------------------------------------
# YouTube comments
# ---------------------------------------------------------------
def extract_video_id(url_or_id):
    """Accepts a full YouTube URL (watch, youtu.be, or shorts link) or a bare video ID."""
    url_or_id = url_or_id.strip()
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',   # watch?v=ID or /embed/ID
        r'youtu\.be\/([0-9A-Za-z_-]{11})',    # youtu.be/ID
        r'shorts\/([0-9A-Za-z_-]{11})',       # shorts/ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    # Fall back to treating the whole input as a bare video ID
    if re.fullmatch(r'[0-9A-Za-z_-]{11}', url_or_id):
        return url_or_id
    return None


def fetch_youtube_comments(video_url_or_id, api_key, max_comments=100):
    """Fetch top-level comments for a YouTube video via the YouTube Data API v3.
    Returns (texts, published_dates, error_message). error_message is '' on success."""
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        return [], [], (
            "The 'google-api-python-client' package isn't installed. "
            "Run: pip install google-api-python-client"
        )

    if not api_key or not api_key.strip():
        return [], [], "Please enter your YouTube Data API key."

    video_id = extract_video_id(video_url_or_id)
    if video_id is None:
        return [], [], "Couldn't parse a video ID from that input — paste a full YouTube URL or an 11-character video ID."

    try:
        youtube = build('youtube', 'v3', developerKey=api_key.strip())
    except Exception as e:
        return [], [], f"Could not initialize the YouTube API client: {e}"

    texts, dates = [], []
    next_page_token = None
    try:
        while len(texts) < max_comments:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_comments - len(texts)),
                pageToken=next_page_token,
                textFormat='plainText',
                order='relevance',
            )
            response = request.execute()

            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                texts.append(snippet['textDisplay'])
                dates.append(snippet['publishedAt'])

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
    except HttpError as e:
        status = getattr(e, 'status_code', None) or getattr(e.resp, 'status', None)
        if status == 403:
            return texts, dates, (
                "YouTube API returned 403 Forbidden — either your API key is invalid/restricted, "
                "your daily quota is exhausted, or comments are disabled for this video."
            )
        if status == 404:
            return [], [], "Video not found — check the URL/ID is correct and the video is public."
        return texts, dates, f"YouTube API error: {e}"
    except Exception as e:
        return texts, dates, f"Unexpected error while fetching comments: {e}"

    if not texts:
        return [], [], "No comments were returned — comments may be disabled or the video may have none yet."

    return texts, dates, ""


def run_youtube_batch(video_url, api_key, max_comments):
    max_comments = int(max_comments) if max_comments else 100
    max_comments = max(1, min(max_comments, 500))  # sane ceiling to protect your daily API quota

    texts, dates, error = fetch_youtube_comments(video_url, api_key, max_comments)
    if error:
        return None, None, None, None, error

    export_df, dashboard_fig, trend_fig, csv_path, summary_text = analyze_texts(texts, dates)
    summary_text = f"Fetched {len(texts)} comments. " + summary_text
    return export_df, dashboard_fig, trend_fig, csv_path, summary_text


def clear_youtube():
    return "", "", 100, "", None, None, None, None


# ---------------------------------------------------------------
# Interface
# ---------------------------------------------------------------
with gr.Blocks(
    title="Social Media Sentiment Analyzer",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=CUSTOM_CSS,
) as demo:
    logged_in_state = gr.State(False)
    username_state = gr.State("")

    # ---------------- Login / Sign Up screen ----------------
    with gr.Group(visible=True, elem_id="auth-card") as auth_section:
        gr.Markdown("# 💬 Social Media Sentiment Analyzer\nPlease log in or create an account to continue.")

        with gr.Tabs():
            with gr.Tab("Login"):
                login_username = gr.Textbox(label="Username")
                login_password = gr.Textbox(label="Password", type="password")
                login_btn = gr.Button("Log In", variant="primary")
                login_message = gr.Textbox(label="", interactive=False, show_label=False)

            with gr.Tab("Sign Up"):
                signup_username = gr.Textbox(label="Choose a username")
                signup_password = gr.Textbox(label="Choose a password", type="password")
                signup_confirm = gr.Textbox(label="Confirm password", type="password")
                signup_btn = gr.Button("Create Account", variant="primary")
                signup_message = gr.Textbox(label="", interactive=False, show_label=False)

    # ---------------- Main app (hidden until login) ----------------
    with gr.Group(visible=False) as main_app:
        with gr.Row():
            welcome_text = gr.Markdown("", elem_id="welcome-bar")
            logout_btn = gr.Button("Log Out", size="sm", variant="secondary")

        gr.Markdown(
            "# 💬 Social Media Sentiment Analyzer\n"
            "A machine learning system that classifies the sentiment of social media text as "
            "**positive**, **negative**, or **neutral**, built on a Support Vector Machine trained "
            "with TF-IDF features on the Sentiment140 dataset. Neutral is assigned when the model's "
            "decision score falls within a small margin of its decision boundary — see the footer "
            "for details.",
            elem_id="app-header"
        )

        with gr.Tabs():
        # ---------------- Tab 1: Single prediction + explainability ----------------
            with gr.Tab("Analyze a Post"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Enter a tweet or social media post",
                            placeholder="e.g. I absolutely love how smooth this update feels!",
                            lines=4
                        )
                        with gr.Row():
                            analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
                            clear_btn = gr.Button("Clear")

                        gr.Examples(
                            examples=[
                                "I absolutely love how smooth this update feels!",
                                "This is the worst service I've ever experienced.",
                                "The meeting is scheduled for 9am tomorrow.",
                            ],
                            inputs=text_input
                        )

                    with gr.Column():
                        result_label = gr.Textbox(label="Predicted Sentiment", interactive=False)
                        confidence_plot = gr.Label(label="Confidence (proxy, based on distance from decision boundary)")
                        subjectivity_output = gr.Textbox(label="Subjectivity (opinion vs. fact)", interactive=False)
                        cleaned_output = gr.Textbox(label="Cleaned text used for prediction", interactive=False)

                explanation_plot = gr.Plot(label="Why the model predicted this")

                analyze_btn.click(
                    fn=predict_sentiment,
                    inputs=text_input,
                    outputs=[result_label, confidence_plot, cleaned_output, explanation_plot, subjectivity_output]
                )
                text_input.submit(
                    fn=predict_sentiment,
                    inputs=text_input,
                    outputs=[result_label, confidence_plot, cleaned_output, explanation_plot, subjectivity_output]
                )
                clear_btn.click(
                    fn=clear_single,
                    inputs=None,
                    outputs=[text_input, result_label, confidence_plot, cleaned_output, explanation_plot, subjectivity_output]
                )

            # ---------------- Tab 2: Batch / trend analysis ----------------
            with gr.Tab("Batch Analysis"):
                gr.Markdown(
                    "Upload a CSV with a `text` column, or paste one post per line below, "
                    "to see sentiment across many posts at once. If your CSV has a date/timestamp "
                    "column, a trend chart will also be generated."
                )
                with gr.Row():
                    with gr.Column():
                        csv_input = gr.File(label="Upload CSV (optional)", file_types=[".csv"])
                        batch_text_input = gr.Textbox(
                            label="Or paste posts here (one per line)",
                            lines=8,
                            placeholder="I love this new update!\nWorst experience ever.\nJust another Monday."
                        )
                        with gr.Row():
                            batch_btn = gr.Button("Analyze Batch", variant="primary")
                            batch_clear_btn = gr.Button("Clear")
                        batch_summary = gr.Textbox(label="Summary", interactive=False)

                batch_dashboard = gr.Plot(label="Batch Analysis Dashboard")
                batch_trend = gr.Plot(label="Trend Over Time (if a date column was found)")
                batch_table = gr.Dataframe(label="Results", wrap=True)
                batch_download = gr.File(label="Download results as CSV")

                batch_btn.click(
                    fn=run_batch,
                    inputs=[csv_input, batch_text_input],
                    outputs=[batch_table, batch_dashboard, batch_trend, batch_download, batch_summary]
                )
                batch_clear_btn.click(
                    fn=clear_batch,
                    inputs=None,
                    outputs=[csv_input, batch_text_input, batch_table, batch_dashboard, batch_trend, batch_download, batch_summary]
                )

            # ---------------- Tab 3: YouTube comments ----------------
            with gr.Tab("YouTube Comments"):
                gr.Markdown(
                    "Analyze sentiment across a YouTube video's comments. Requires a YouTube Data "
                    "API v3 key — get one free from [Google Cloud Console](https://console.cloud.google.com/apis/credentials) "
                    "(enable the 'YouTube Data API v3' first). Your key is used only for this request "
                    "and is never saved to disk."
                )
                with gr.Row():
                    with gr.Column():
                        yt_url_input = gr.Textbox(
                            label="YouTube video URL or video ID",
                            placeholder="e.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                        )
                        yt_api_key_input = gr.Textbox(
                            label="YouTube Data API key",
                            type="password",
                            value=os.environ.get("YOUTUBE_API_KEY", ""),
                            placeholder="Paste your API key here"
                        )
                        yt_max_comments = gr.Number(
                            label="Max comments to fetch (capped at 500 to protect your API quota)",
                            value=100, precision=0
                        )
                        with gr.Row():
                            yt_btn = gr.Button("Fetch & Analyze Comments", variant="primary")
                            yt_clear_btn = gr.Button("Clear")
                        yt_summary = gr.Textbox(label="Summary", interactive=False)

                yt_dashboard = gr.Plot(label="Comment Sentiment Dashboard")
                yt_trend = gr.Plot(label="Sentiment Over Time (based on comment post dates)")
                yt_table = gr.Dataframe(label="Results", wrap=True)
                yt_download = gr.File(label="Download results as CSV")

                yt_btn.click(
                    fn=run_youtube_batch,
                    inputs=[yt_url_input, yt_api_key_input, yt_max_comments],
                    outputs=[yt_table, yt_dashboard, yt_trend, yt_download, yt_summary]
                )
                yt_clear_btn.click(
                    fn=clear_youtube,
                    inputs=None,
                    outputs=[yt_url_input, yt_api_key_input, yt_max_comments, yt_summary,
                             yt_table, yt_dashboard, yt_trend, yt_download]
                )

        gr.Markdown(
            "---\n"
            "Built for the project: *Sentiment Analysis of Social Media Data Using Machine Learning "
            "Techniques* — Osiako Victor Prince, Federal University Lokoja. Core model: TF-IDF + SVM "
            f"(binary Positive/Negative classifier). **Neutral is not a class the SVM was trained on** — "
            f"it is assigned when the model's decision score falls within ±{NEUTRAL_THRESHOLD} of its "
            "decision boundary, following the confidence-threshold approach described in the project's "
            "methodology. This is a research prototype; predictions may be inaccurate, especially for "
            "sarcasm, negation, or highly context-dependent text. The confidence values shown are a "
            "distance-based heuristic, not calibrated probabilities."
        )

    # ---------------- Auth wiring ----------------
    def handle_login(username, password):
        success, message = authenticate_user(username, password)
        if success:
            return (
                True, username,
                gr.update(visible=False),           # hide auth_section
                gr.update(visible=True),            # show main_app
                f"Welcome, **{username}**!",        # welcome_text
                "",                                  # clear login_message
            )
        return (
            False, "",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            message,
        )

    def handle_signup(username, password, confirm):
        message, success = register_user(username, password, confirm)
        if success:
            # Auto-login right after a successful sign-up
            return (
                True, username,
                gr.update(visible=False),
                gr.update(visible=True),
                f"Welcome, **{username}**!",
                message,
            )
        return (
            False, "",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            message,
        )

    def handle_logout():
        return (
            False, "",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            "", "", "",   # clear login fields
        )

    login_btn.click(
        fn=handle_login,
        inputs=[login_username, login_password],
        outputs=[logged_in_state, username_state, auth_section, main_app, welcome_text, login_message]
    )
    signup_btn.click(
        fn=handle_signup,
        inputs=[signup_username, signup_password, signup_confirm],
        outputs=[logged_in_state, username_state, auth_section, main_app, welcome_text, signup_message]
    )
    logout_btn.click(
        fn=handle_logout,
        inputs=None,
        outputs=[logged_in_state, username_state, auth_section, main_app, welcome_text,
                 login_username, login_password, login_message]
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
