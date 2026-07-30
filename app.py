# app.py
import streamlit as st

from src.audio_preprocessing import AudioTooShortError, UnsupportedAudioError, prepare_audio
from src.dialect_classifier import DialectClassifier, top_result

st.set_page_config(page_title="Saudi Accent Checker", page_icon="\U0001F3A4")
st.title("Saudi Accent Checker")
st.caption(
    "Record or upload Arabic speech to see confidence scores across Arabic dialects. "
    "Results are meaningless for non-Arabic speech — there is no language filter yet."
)


@st.cache_resource
def get_classifier() -> DialectClassifier:
    return DialectClassifier()


def render_scores(scores: dict) -> None:
    label, probability = top_result(scores)
    classifier = get_classifier()
    if classifier.is_saudi_label(label):
        st.success(f"Top match: **{label}** (Saudi/Gulf) — {probability:.0%} confidence")
    else:
        st.info(f"Top match: **{label}** — {probability:.0%} confidence (not Saudi/Gulf)")
    st.bar_chart(scores)


tab_record, tab_upload = st.tabs(["Record", "Upload a file"])

audio_input = None
with tab_record:
    recorded = st.audio_input("Record yourself speaking Arabic")
    if recorded is not None:
        audio_input = recorded

with tab_upload:
    uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if uploaded is not None:
        audio_input = uploaded

if audio_input is not None:
    try:
        waveform = prepare_audio(audio_input)
    except AudioTooShortError:
        st.error("That clip is too short — please provide at least 1 second of speech.")
    except UnsupportedAudioError:
        st.error("Couldn't read that audio file. Try a WAV, MP3, or M4A file.")
    else:
        with st.spinner("Analyzing accent..."):
            scores = get_classifier().predict(waveform)
        render_scores(scores)
