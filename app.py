from transformers import pipeline
import streamlit as st

# Initialize the Hugging Face model pipeline
pipe = pipeline("text-generation", model="gpt2")

st.title("Chatbot using GPT-2")

# Input area for prompt
prompt = st.text_area("Enter your prompt:")

# Button to trigger generation
if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            try:
                # Generate response using the model pipeline
                response = pipe(prompt, max_length=100, num_return_sequences=1)
                
                # Extract and display the response text
                response_text = response[0]['generated_text']
                st.write(response_text)

            except Exception as e:
                st.error(f"An error occurred: {e}")
