"""Streamlit demo: type a sentence and see how likely each of the six emotions is.

Run locally:  streamlit run demo/app.py
"""

import streamlit as st
from transformers import pipeline

# Replace YOUR_USERNAME with your Hugging Face username.
MODEL_ID = "Saaddevasrih/emotion-distilbert"

EXAMPLES = [
    "i finally passed my exam and i cant stop smiling",
    "i miss my old friends and i feel empty without them",
    "i feel so grateful to have someone who cares about me this much",
    "i am so tired of people ignoring what i say",
    "i keep worrying that something will go wrong tomorrow",
    "i did not expect that at all and i am still amazed",
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
    """Return a list of (emotion, probability), highest first."""
    # top_k=None asks for the score of every emotion, not only the winner.
    results = load_classifier()(text, top_k=None)
    if isinstance(results[0], list):  # some versions wrap the result in an extra list
        results = results[0]
    scores = [(result["label"], float(result["score"])) for result in results]
    return sorted(scores, key=lambda pair: pair[1], reverse=True)


def use_example(example):
    # Runs when an example button is clicked, before the page is redrawn.
    st.session_state.text = example


st.title("🎭 Emotion classifier")
st.caption(
    "A DistilBERT model fine-tuned on the dair-ai/emotion dataset. "
    "It always picks one of six emotions and has no neutral option, "
    "so look at the percentages: a low top score means the model is unsure."
)

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0]

# key="text" ties this box to st.session_state.text in both directions.
st.text_input("Your sentence (English), then press Enter", key="text")
with st.expander("Examples"):
    for example in EXAMPLES:
        st.button(example, on_click=use_example, args=(example,))

text = st.session_state.text.strip()

if text:
    scores = predict(text)
    top_label, top_score = scores[0]

    st.subheader(f"{top_label} ({top_score:.0%})")
    for label, score in scores:
        st.progress(score, text=f"{label}  {score:.0%}")
else:
    st.info("Type a sentence to see the result.")
