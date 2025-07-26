import streamlit as st
from transformers import pipeline
import textstat

# Load summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

st.title("Accessible Summary Tool for Scientific Texts")

user_input = st.text_area("Paste scientific or medical text here:", height=300)

if user_input:
    with st.spinner("Generating summary..."):
        summary = summarizer(user_input, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        flesch_score = textstat.flesch_reading_ease(summary)
        difficult_words = textstat.difficult_words(summary)
    
    st.subheader("🧾 Summary")
    st.write(summary)

    st.subheader("📊 Accessibility Metrics")
    st.markdown(f"- **Flesch Reading Ease Score:** {flesch_score:.2f}")
    st.markdown(f"- **Difficult Words Count:** {difficult_words}")

    if flesch_score < 50:
        st.warning("⚠️ This summary may be too complex for general audiences.")
    else:
        st.success("✅ This summary is relatively accessible.")
