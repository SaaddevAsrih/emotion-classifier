"""Streamlit demo: type a sentence and see how likely each of the six emotions is.

Run locally:  streamlit run demo/app.py

Design idea: the page takes on the colour of the emotion the model finds.
The surer the model is, the stronger the colour. One strip shows how the
100% is shared between the six emotions, so a mixed strip means "not sure".
"""

import streamlit as st
from transformers import pipeline

MODEL_ID = "saaddevasrih/emotion-distilbert"

# One fixed colour per emotion, in a fixed order, so the strip is easy to compare
# from one sentence to the next.
COLORS = {
    "sadness": "#3D5A98",
    "joy": "#F2B705",
    "love": "#E05C8A",
    "anger": "#D7372F",
    "fear": "#6B4BA3",
    "surprise": "#1FA39A",
}
PAPER = "#F5F6FA"  # page colour when nothing is tinted
INK = "#1C1B2E"    # text colour

EXAMPLES = [
    "i cant stop smiling today",
    "i feel empty without them",
    "i am so grateful to have you",
    "i am tired of being ignored",
    "i keep worrying about tomorrow",
    "i did not expect that at all",
]

st.set_page_config(page_title="Emotion classifier", page_icon="🎭")


@st.cache_resource(show_spinner="Loading the model ...")
def load_classifier():
    """Download the model from the Hugging Face Hub once and keep it in memory.

    cache_resource (not cache_data) is the right cache for big objects like a
    model: every rerun and every visitor shares the same loaded copy.
    """
    return pipeline("text-classification", model=MODEL_ID)


def predict(text):
    """Return {emotion: probability} for all six emotions."""
    # top_k=None asks for the score of every emotion, not only the winner.
    results = load_classifier()(text, top_k=None)
    if isinstance(results[0], list):  # some versions wrap the result in an extra list
        results = results[0]
    return {result["label"]: float(result["score"]) for result in results}


def use_example(example):
    # Runs when an example button is clicked, before the page is redrawn.
    st.session_state.text = example


def page_style(tint_color, tint_percent):
    """CSS for the whole page. The background is the paper colour mixed with
    `tint_percent` % of the emotion's colour."""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600&family=Fraunces:opsz,wght@9..144,500&display=swap');

    .stApp {{
        background: color-mix(in srgb, {tint_color} {tint_percent}%, {PAPER});
        transition: background 0.5s ease;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}

    html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp div {{
        font-family: 'Figtree', system-ui, sans-serif;
        color: {INK};
    }}

    /* Every custom block below is a <div>, and every selector starts with
       .stApp, so these rules win over Streamlit's own text styles. */
    .stApp .app-name {{ font-size: 1.05rem; font-weight: 600; }}
    .stApp .app-about {{ font-size: 0.95rem; opacity: 0.75; margin: 0.15rem 0 1.2rem; max-width: 34rem; }}

    .stApp .stTextInput input {{
        font-size: 1.15rem;
        padding: 0.85rem 1rem;
        background: #FFFFFF;
        border: 1.5px solid color-mix(in srgb, {INK} 25%, transparent);
        border-radius: 10px;
    }}
    .stApp .stTextInput input:focus {{ border-color: {INK}; box-shadow: none; }}

    /* The example sentences: small, quiet pills. */
    .stApp .st-key-examples button {{
        background: transparent;
        border: 1px solid color-mix(in srgb, {INK} 28%, transparent);
        border-radius: 999px;
        padding: 0.15rem 0.8rem;
        min-height: 0;
    }}
    .stApp .st-key-examples button p {{ font-size: 0.85rem; }}
    .stApp .st-key-examples button:hover {{ background: #FFFFFF; border-color: {INK}; }}
    .stApp .st-key-examples button:focus-visible {{ outline: 2px solid {INK}; outline-offset: 2px; }}

    /* The result. */
    .stApp .emotion-word {{
        font-family: 'Fraunces', Georgia, serif;
        font-weight: 500;
        font-size: clamp(3.6rem, 16vw, 7rem);
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin: 1.2rem 0 0.9rem;
        color: color-mix(in srgb, {tint_color} 62%, {INK});
    }}
    .stApp .sureness {{ font-size: 1.05rem; font-weight: 500; margin: 0 0 1.2rem; }}

    .stApp .strip {{
        display: flex;
        height: 30px;
        border-radius: 8px;
        overflow: hidden;
        background: #FFFFFF;
        box-shadow: 0 0 0 1.5px color-mix(in srgb, {INK} 18%, transparent);
    }}
    .stApp .strip span {{ display: block; height: 100%; transition: width 0.5s ease; }}

    .stApp .scores {{ margin: 1.1rem 0 0; max-width: 20rem; }}
    .stApp .scores .row {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.26rem 0;
        font-size: 0.98rem;
        font-variant-numeric: tabular-nums;
    }}
    .stApp .scores .swatch {{ width: 0.85rem; height: 0.85rem; border-radius: 3px; flex: none; }}
    .stApp .scores .name {{ flex: 1; }}
    .stApp .scores .row:first-child {{ font-weight: 600; }}

    .stApp .footnote {{ font-size: 0.88rem; line-height: 1.5; opacity: 0.8; margin-top: 2.4rem; max-width: 34rem; }}
    .stApp .footnote a {{ color: {INK}; }}

    @media (prefers-reduced-motion: reduce) {{
        .stApp, .strip span {{ transition: none; }}
    }}
    </style>
    """


def result_html(scores):
    """HTML for the big emotion word, the strip, and the list of six scores."""
    top = max(scores, key=scores.get)
    percent = round(scores[top] * 100)
    sureness = f"The model is {percent}% sure." if percent >= 60 else f"The model is not sure: only {percent}%."

    # The strip keeps the emotions in a fixed order; each part is as wide as its share.
    segments = "".join(
        f'<span style="width:{scores.get(name, 0) * 100:.2f}%;background:{color}"></span>'
        for name, color in COLORS.items()
    )
    spoken = ", ".join(f"{name} {round(scores.get(name, 0) * 100)} percent" for name in COLORS)

    rows = "".join(
        f'<div class="row" role="listitem">'
        f'<span class="swatch" style="background:{COLORS.get(name, INK)}"></span>'
        f'<span class="name">{name}</span><span>{round(score * 100)}%</span></div>'
        for name, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    )

    return (
        f'<div class="emotion-word">{top}</div>'
        f'<div class="sureness">{sureness}</div>'
        f'<div class="strip" role="img" aria-label="Share of each emotion: {spoken}">{segments}</div>'
        f'<div class="scores" role="list">{rows}</div>'
    )


# ---------------------------------------------------------------- the page ---

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0]

text = st.session_state.text.strip()
scores = predict(text) if text else None

if scores:
    top_emotion = max(scores, key=scores.get)
    # 6% tint when the model is guessing, up to 26% when it is certain.
    tint = 6 + 20 * scores[top_emotion]
    st.markdown(page_style(COLORS.get(top_emotion, INK), round(tint)), unsafe_allow_html=True)
else:
    st.markdown(page_style(PAPER, 0), unsafe_allow_html=True)

st.markdown(
    '<div class="app-name">Emotion classifier</div>'
    '<div class="app-about">Write a sentence and a fine-tuned DistilBERT model names the emotion in it.</div>',
    unsafe_allow_html=True,
)

# key="text" ties this box to st.session_state.text in both directions.
st.text_input("Your sentence, in English. Press Enter to check it.", key="text")

with st.container(key="examples", horizontal=True):
    for example in EXAMPLES:
        st.button(example, on_click=use_example, args=(example,))

if scores:
    st.markdown(result_html(scores), unsafe_allow_html=True)
else:
    st.markdown('<div class="sureness" style="margin-top:2rem">Write a sentence to see the result.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footnote">The model always picks one of six emotions and has no neutral option, '
    "so a strip with several colours means it is unsure. A single colour is not a guarantee either: "
    "it reads “i am not happy about this at all” as joy. "
    '<a href="https://github.com/SaaddevAsrih/emotion-classifier">Code and results on GitHub</a>, '
    '<a href="https://huggingface.co/saaddevasrih/emotion-distilbert">model on Hugging Face</a>.</div>',
    unsafe_allow_html=True,
)
