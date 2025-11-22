import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

from interviewer import generate_questions, evaluate_answer
from ocr_utils import image_bytes_to_text

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Interviewer", layout="centered")
st.title("🎤 AI Interviewer for Student Presentations")
st.markdown("Upload your screenshot + explain your project. I will interview and evaluate you.")

# --------------------------------------------------
# 1. IMAGE UPLOAD + OCR
# --------------------------------------------------
st.subheader("1️⃣ Screen Content (Image → OCR)")

uploaded_image = st.file_uploader("Upload project screenshot / slide", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Screenshot", use_column_width=True)

    if st.button("🔍 Extract Text from Screenshot"):
        with st.spinner("Running OCR..."):
            extracted_text = image_bytes_to_text(uploaded_image.getvalue())
            st.session_state["screen_text"] = extracted_text
        st.success("✅ OCR Completed. Text updated below.")

screen_text = st.text_area(
    "📄 Screen Text (OCR Output / Slide Content)",
    value=st.session_state.get("screen_text", ""),
    height=200
)

# --------------------------------------------------
# 2. AUDIO UPLOAD + SPEECH TO TEXT
# --------------------------------------------------
st.subheader("2️⃣ Student Explanation (Audio → Text)")

audio_file = st.file_uploader("Upload your explanation audio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    if st.button("🎤 Convert Speech to Text"):
        with st.spinner("Transcribing audio..."):

            transcription = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file
            )

            st.session_state["explanation_text"] = transcription.text
            st.success("✅ Audio Transcribed Successfully!")

            st.text_area(
                "Transcription Preview",
                value=transcription.text,
                height=100,
                disabled=True
            )

explanation = st.text_area(
    "🗣️ Student Explanation (You can also type manually)",
    value=st.session_state.get("explanation_text", ""),
    height=200
)

# --------------------------------------------------
# QUESTIONS STORAGE
# --------------------------------------------------
if "questions" not in st.session_state:
    st.session_state["questions"] = None

# --------------------------------------------------
# 3. GENERATE QUESTIONS
# --------------------------------------------------
if st.button("🎯 Generate Interview Questions"):

    if not screen_text.strip() or not explanation.strip():
        st.warning("⚠️ Please provide both screen content and explanation.")
    else:
        with st.spinner("Thinking like a professor..."):
            st.session_state["questions"] = generate_questions(screen_text, explanation)
        st.success("✅ Interview questions generated!")

# --------------------------------------------------
# DISPLAY QUESTIONS
# --------------------------------------------------
if st.session_state["questions"]:

    st.subheader("📘 Interview Questions")
    for q in st.session_state["questions"]["questions"]:
        st.write("✅", q)

    st.subheader("📗 Follow-up Questions")
    for f in st.session_state["questions"]["followups"]:
        st.write("➡️", f)

    st.markdown("---")

    # --------------------------------------------------
    # STUDENT ANSWER + EVALUATION
    # --------------------------------------------------
    student_answer = st.text_area("✍️ Paste Student Answer", height=150)

    if st.button("🧠 Evaluate Answer"):
        if not student_answer.strip():
            st.warning("⚠️ Please provide an answer first.")
        else:
            with st.spinner("Evaluating like an examiner..."):
                result = evaluate_answer(
                    screen_text,
                    explanation,
                    student_answer
                )

            st.subheader("📊 Evaluation Result")
            st.metric("Technical Depth", result["technical_depth"])
            st.metric("Clarity", result["clarity"])
            st.metric("Originality", result["originality"])
            st.metric("Implementation Understanding", result["implementation_understanding"])

            st.subheader("📝 Feedback")
            st.write(result["feedback"])
