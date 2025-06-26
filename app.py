# app.py

import streamlit as st
from PIL import Image
import os

from nlp_module import answer_question
from vision_module import describe_image, get_image_description
from gemini_module import ask_gemini

st.set_page_config(page_title="AI Learning Assistant", layout="centered")
st.title("🎓 AI-Powered Learning Assistant")

st.sidebar.title("Modes")
mode = st.sidebar.radio("Choose Input Type", [
    "Text", 
    "Voice", 
    "Image Captioning", 
    "Image Question Answering", 
    "Engagement Detection"
])

if mode == "Text":
    question = st.text_input("Ask your question:")
    context = st.text_area("Provide context (like a paragraph from textbook):")
    if st.button("Get Answer") and question and context:
        answer = answer_question(question, context)
        st.success(f"📘 Answer: {answer}")

elif mode == "Image Captioning":
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img_path = f"temp_{uploaded_img.name}"
        image.save(img_path)
        caption = describe_image(img_path)
        os.remove(img_path)

        st.success(f"🖼️ Caption/Interpretation: {caption}")

elif mode == "Image Question Answering":
    uploaded_img = st.file_uploader("Upload a diagram", type=["jpg", "png", "jpeg"])
    question = st.text_input("Ask a question based on the diagram:")

    if uploaded_img and question:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Uploaded Diagram", use_container_width=True)

        with st.spinner("Analyzing image and generating answer..."):
            context = get_image_description(image)
            answer = ask_gemini(question, context)

        st.success(f"📘 Answer: {answer}")

elif mode == "Engagement Detection":
    st.warning("Engagement detection module is not yet implemented.")
