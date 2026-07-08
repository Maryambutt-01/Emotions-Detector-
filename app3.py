import re
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
EMOTION_MAP = {0: "sadness", 1: "anger", 2: "love", 3: "surprise", 4: "fear", 5: "joy"}

EMOTION_STYLE = {
    "sadness":  {"emoji": "😢", "color": "#4A90D9"},
    "anger":    {"emoji": "😠", "color": "#E74C3C"},
    "love":     {"emoji": "❤️", "color": "#E91E8C"},
    "surprise": {"emoji": "😲", "color": "#F5A623"},
    "fear":     {"emoji": "😨", "color": "#7B5EA7"},
    "joy":      {"emoji": "😄", "color": "#F4D03F"},
}

# Fallback stopword list so the app doesn't require an nltk download at runtime.
STOPWORDS = set("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll
these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down
in out on off over under again further then once here there when where why
how all any both each few more most other some such no nor not only own same
so than too very s t can will just don don't should should've now d ll m o re
ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't
hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn
needn't shan shan't shouldn shouldn't wasn wasn't weren weren't won won't
wouldn wouldn't
""".split())


def clean_text(text: str) -> str:
    """Replicates the exact preprocessing pipeline used during training."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(ch for ch in text if not ch.isdigit())
    text = ''.join(ch for ch in text if ch.isascii())
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    logistic_model = joblib.load("logistic_regression_model.pkl")
    nb_model = joblib.load("multinomial_nb_model.pkl")
    return vectorizer, logistic_model, nb_model


vectorizer, logistic_model, nb_model = load_artifacts()

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0f1220 0%, #1a1f35 100%); }

.hero {
    text-align: center;
    padding: 1.2rem 0 0.4rem 0;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #F5A623, #E91E8C, #7B5EA7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero p { color: #9aa3c0; font-size: 1.05rem; }

div[data-testid="stTextArea"] textarea {
    border-radius: 14px;
    border: 1px solid #333a5c;
    background-color: #171b30;
    color: #f0f0f5;
    font-size: 1.05rem;
}

.result-card {
    border-radius: 20px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.08);
}
.result-emoji { font-size: 4rem; line-height: 1; }
.result-label {
    font-size: 1.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}
.result-conf { color: #c7ccdd; font-size: 0.95rem; margin-top: 0.2rem; }

.model-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    color: #c7ccdd;
    font-size: 0.8rem;
    margin-bottom: 0.6rem;
}

.footer-note { text-align:center; color:#6c7396; font-size:0.85rem; padding-top:2rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_choice = st.radio(
        "Choose a model",
        ["Logistic Regression", "Multinomial Naive Bayes", "Compare both"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### 📊 About the models")
    st.markdown(
        "- **Vectorizer:** TF-IDF (13,361 features)\n"
        "- **Logistic Regression** — accuracy ≈ **86.3%**\n"
        "- **Multinomial Naive Bayes** (TF-IDF) — trained on same split\n"
        "- **Classes:** sadness, anger, love, surprise, fear, joy"
    )
    st.markdown("---")
    st.markdown("### 🧹 Preprocessing applied")
    st.markdown(
        "1. Lowercase\n"
        "2. Remove punctuation\n"
        "3. Remove digits\n"
        "4. Remove non-ASCII / emojis\n"
        "5. Remove stopwords"
    )

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎭 Emotion Detector</h1>
    <p>Type a sentence and let NLP models tell you what emotion it carries.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Example chips
# ----------------------------------------------------------------------------
examples = [
    "I can't believe you did this, I am so mad right now",
    "I just found out I got the job, I'm overjoyed!",
    "I miss him so much it hurts every single day",
    "She surprised me with a birthday party, I was speechless",
    "I'm terrified of what might happen tomorrow",
    "I love spending quiet mornings with my family",
]

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

st.markdown("**Try an example:**")
cols = st.columns(3)
for i, ex in enumerate(examples):
    if cols[i % 3].button(ex[:40] + ("…" if len(ex) > 40 else ""), key=f"ex_{i}", use_container_width=True):
        st.session_state.text_input = ex

# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
text_input = st.text_area(
    "Your text",
    value=st.session_state.text_input,
    height=120,
    placeholder="e.g. I can't stop smiling, today has been amazing!",
    label_visibility="collapsed",
)

analyze = st.button("✨ Analyze Emotion", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------------
def predict(model, cleaned_text: str):
    vec = vectorizer.transform([cleaned_text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return EMOTION_MAP[pred], proba


def render_result_card(label: str, confidence: float, model_name: str):
    style = EMOTION_STYLE[label]
    st.markdown(f"""
    <div class="result-card" style="background: linear-gradient(160deg, {style['color']}22, #171b3000); border-color:{style['color']}55;">
        <div class="model-badge">{model_name}</div>
        <div class="result-emoji">{style['emoji']}</div>
        <div class="result-label" style="color:{style['color']};">{label}</div>
        <div class="result-conf">Confidence: {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)


def render_proba_chart(proba, model_name: str):
    labels = [EMOTION_MAP[i] for i in range(len(proba))]
    colors = [EMOTION_STYLE[l]["color"] for l in labels]
    order = np.argsort(proba)
    fig = go.Figure(go.Bar(
        x=[proba[i] * 100 for i in order],
        y=[labels[i] for i in order],
        orientation="h",
        marker_color=[colors[i] for i in order],
        text=[f"{proba[i]*100:.1f}%" for i in order],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{model_name} — probability breakdown",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(range=[0, 100], showgrid=False, title="Probability (%)"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------------
if analyze:
    if not text_input.strip():
        st.warning("Please enter some text first 🙂")
    else:
        cleaned = clean_text(text_input)

        if not cleaned.strip():
            st.warning("After cleaning, no meaningful words were left to analyze. Try a longer sentence.")
        else:
            with st.expander("🔍 See cleaned text sent to the model"):
                st.code(cleaned or "(empty)")

            st.markdown("### Results")

            if model_choice == "Compare both":
                c1, c2 = st.columns(2)
                log_label, log_proba = predict(logistic_model, cleaned)
                nb_label, nb_proba = predict(nb_model, cleaned)

                with c1:
                    render_result_card(log_label, max(log_proba), "Logistic Regression")
                with c2:
                    render_result_card(nb_label, max(nb_proba), "Multinomial Naive Bayes")

                c3, c4 = st.columns(2)
                with c3:
                    render_proba_chart(log_proba, "Logistic Regression")
                with c4:
                    render_proba_chart(nb_proba, "Multinomial Naive Bayes")

                if log_label != nb_label:
                    st.info(f"⚖️ The two models disagree: Logistic Regression says **{log_label}**, "
                            f"Naive Bayes says **{nb_label}**.")
            else:
                model = logistic_model if model_choice == "Logistic Regression" else nb_model
                label, proba = predict(model, cleaned)

                c1, c2 = st.columns([1, 1.4])
                with c1:
                    render_result_card(label, max(proba), model_choice)
                with c2:
                    render_proba_chart(proba, model_choice)

st.markdown(
    '<div class="footer-note">Built with Streamlit · TF-IDF + Logistic Regression / Multinomial Naive Bayes</div>',
    unsafe_allow_html=True,
)
