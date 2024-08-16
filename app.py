import streamlit as st
from transformers import pipeline

# Set the page configuration
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto"
)

# Upload the document
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

# Read the content of the document
def read_document(file):
    text = ""
    if file.type == "text/plain":
        text = file.read().decode("utf-8")
    elif file.type == "application/pdf":
        import PyPDF2
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from docx import Document
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text

def text_splitter(raw_text):
    import textwrap
    return textwrap.wrap(raw_text, width=1000, break_long_words=False)

def generate_summary(text):
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def generate_answer(question, context):
    question_answerer = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
    result = question_answerer(question=question, context=context)
    return result['answer']

def main():
    st.header("Document GPT📄")

    if uploaded_file:
        with st.spinner("Processing document..."):
            raw_text = read_document(uploaded_file)
            text_chunks = text_splitter(raw_text)
            combined_text = " ".join(text_chunks)  # Combine text chunks for summarization

            st.subheader("Summary of the document")
            summary = generate_summary(combined_text)
            st.write(summary)

            question = st.text_input("Ask a question about the document:")

            if st.button("Ask"):
                with st.spinner("Generating answer..."):
                    if question:
                        answer = generate_answer(question, combined_text)
                        st.write(answer)
                    else:
                        st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
