import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the T5 Model and Tokenizer
MODEL_NAME = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Streamlit UI Setup
st.title("📰 Advanced News Article Summarizer")
st.write("Enter a news article, and get a concise summary!")

# User input
article_text = st.text_area("Paste the news article here:", height=200)
summary_length = st.slider("Select summary length:", min_value=30, max_value=300, value=100, step=10)

if st.button("Summarize"):
    if article_text.strip():
        # Preprocess input
        input_text = "summarize: " + article_text
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate summary
        summary_ids = model.generate(input_ids, max_length=summary_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Display result
        st.subheader("Summary:")
        st.success(summary)
                
    else:
        st.warning("Please enter some text to summarize!")
