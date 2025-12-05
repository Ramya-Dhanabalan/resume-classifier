#!/usr/bin/env python
# coding: utf-8

# In[8]:


# IMPORT LIBRARIES
import re
import PyPDF2
import docx2txt
import pdfplumber
import docx
from docx import Document
import pandas as pd
import streamlit as st

import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import warnings
warnings.filterwarnings('ignore')

#----------------------------------------------------------------------------------------------------

st.title('RESUME CLASSIFICATION')
st.markdown('<style>h1{color: Purple;}</style>', unsafe_allow_html=True)
st.subheader('Welcome to Resume Classification App')

# Collect all extracted skill lists from resume_data
def extract_skills(resume_data):
    nlp_text = nlp(resume_data)
    noun_chunks = nlp_text.noun_chunks
    tokens = [token.text for token in nlp_text if not token.is_stop] # removing stop words and implementing word tokenization

    data = pd.read_csv("clean_it_skills.csv")
    skills = data["skills"].dropna().str.lower().tolist()     
    skillset = []

    for token in tokens: # check for one-grams (example: python)
        if token.lower() in skills:
            skillset.append(token)

    for token in noun_chunks: # check for bi-grams and tri-grams (example: machine learning)
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)   
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def getText(file):
    fullText = ""

    # If PDF File
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf_file:
            for page in pdf_file.pages:
                text = page.extract_text()
                if text:
                    fullText += text + "\n"

    # If DOCX File
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            fullText += para.text + "\n"

    return fullText


def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())  
    return resume

def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

file_type=pd.DataFrame([], columns=['Uploaded File',  'Predicted Profile','Skills',])
filename = []
predicted = []
skills = []

#-------------------------------------------------------------------------------------------------
# MAIN CODE
import pickle as pk
model = pk.load(open(r'modelDT.pkl', 'rb'))
Vectorizer = pk.load(open(r'vector.pkl', 'rb'))

upload_file = st.file_uploader('Upload Your Resumes', type= ['docx','pdf'],accept_multiple_files=True)
  
for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        cleaned = preprocess(display(doc_file))
        prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        predicted.append(prediction)
        extText = getText(doc_file)
        skills.append(extract_skills(extText))
        
if len(predicted) > 0:
    file_type['Uploaded File'] = filename
    file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted
    st.table(file_type.style.format())
    
select = ['Peoplesoft resumes','Resumes','SQL Developer Lightning insight','workday resumes']
st.subheader('Select as per Requirement')
option = st.selectbox('Fields',select)

if option == 'Peoplesoft resumes':
    st.table(file_type[file_type['Predicted Profile'] == 'Peoplesoft resumes'])
elif option == 'Resumes':
    st.table(file_type[file_type['Predicted Profile'] == 'Resumes'])
elif option == 'SQL Developer Lightning insight':
    st.table(file_type[file_type['Predicted Profile'] == 'SQL Developer Lightning insight'])
elif option == 'workday resumes':
    st.table(file_type[file_type['Predicted Profile'] == 'workday resumes'])


# In[ ]:




