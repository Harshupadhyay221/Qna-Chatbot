import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Title of the app
st.title("Text Summarization App")

# User input
user_input = st.text_area("Enter the text you want to summarize:")

if st.button("Summarize"):
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    # Tokenize and generate summary
    inputs = tokenizer.encode("summarize: " + user_input, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    st.write("### Summary:")
    st.write(summary)
