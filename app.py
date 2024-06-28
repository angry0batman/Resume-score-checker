import streamlit as st
import pickle
import PyPDF2
import pandas as pd
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
import re

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the model and vectorizer
with open('resume_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Function to extract sections from resume
def extract_sections(text):
    education = re.findall(r'(education|qualifications|academic background)(.*?)(experience|skills|achievements|projects|work history)', text, re.DOTALL | re.IGNORECASE)
    skills = re.findall(r'(skills|expertise|proficiencies)(.*?)(experience|education|achievements|projects|work history)', text, re.DOTALL | re.IGNORECASE)
    achievements = re.findall(r'(achievements|accomplishments|awards)(.*?)(experience|skills|education|projects|work history)', text, re.DOTALL | re.IGNORECASE)
    
    def extract_section_content(matches):
        if matches:
            return matches[0][1].strip()
        return "Not Found"
    
    return {
        'Education': extract_section_content(education),
        'Skills': extract_section_content(skills),
        'Achievements': extract_section_content(achievements)
    }

# Streamlit app
st.title("Resume Score Checker")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    
    # Preprocess and predict
    resume_cleaned = preprocess_text(resume_text)
    resume_vectorized = vectorizer.transform([resume_cleaned])
    score = model.predict(resume_vectorized)[0]
    
    st.subheader("Resume Score:")
    st.write(f"**Score:** {score:.2f}")
    
    # Extract and display different sections
    sections = extract_sections(resume_text)
    
    st.subheader("Resume Analysis:")
    for section, content in sections.items():
        st.markdown(f"<h3>{section}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>{content}</p>", unsafe_allow_html=True)
    
    # Analysis (example: word count, most common words)
    st.subheader("Text Analysis:")
    word_count = len(resume_text.split())
    st.write(f"**Word Count:** {word_count}")
    
    words = resume_text.split()
    most_common_words = pd.DataFrame(Counter(words).most_common(10), columns=['Word', 'Count'])
    st.write("**Most Common Words:**")
    st.table(most_common_words)

    # Add CSS for better styling
    st.markdown(
        """
        <style>
        .stTable {
            margin-bottom: 20px;
        }
        h3 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        p {
            margin-bottom: 15px;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
