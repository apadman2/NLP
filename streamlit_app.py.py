#################### Libraries
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
import spacy
import string as string
import re
import streamlit as st
st.set_page_config(layout="wide")
from spellchecker import SpellChecker
import pandas as pd

#################### Functions
def sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    x = analyzer.polarity_scores(text)
    final = pd.DataFrame({"Value": x.values()}, index=["Negative", "Neutral", "Positive", "Compound Score"])
    return st.dataframe(final)

def frequent(text):
    punctuation = string.punctuation
    for i in text:
        if i in punctuation:
            text = text.replace(i, "")
    top_ten = nltk.word_tokenize(text)
    x = Counter(top_ten)
    x = x.most_common(10)
    return st.write(str(x))

def plural(text):
    x = TextBlob(text)
    x = x.detect_language()
    return st.write(str(x))

def lang(text):
    x = TextBlob(text)
    x = x.translate('en', 'zh')
    return st.write(str(x))

def postag(text):
    x = nltk.word_tokenize(text)
    x = nltk.pos_tag(x)
    return st.write(x)

def entity_recognition(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    x = ''
    for ent in doc.ents:
        x = x + str(ent.text) + ", "
    x = x[:-2]
    return st.write(str(x))

def chunking(text):
    nltk.download('averaged_perceptron_tagger') 
    token = nltk.word_tokenize(text)
    tags = nltk.pos_tag(token)
    reg = "NP: {<DT>?<JJ>*<NN>}"
    a = nltk.RegexpParser(reg)
    x = a.parse(tags)
    return st.write(str(x))

def correction(text):
    punctuation = string.punctuation
    for i in text:
        if i in punctuation:
            text = text.replace(i, "")
    y = re.findall("[a-zA-Z,.]+", text)
    spell = SpellChecker()
    x = spell.unknown(y)
    if len(x) == 0:
        return "All the words exist in this Dictionary"
    else:
        return st.write(str(x))

###################### Frontend #########################
def main():
    st.title("NLP Application")
    x = st.text_input("Input text:", value="Enter your Text here!!")
    st.header("Sentiment Analysis:")
    sentiment(x)
    st.header("Most frequent words:")
    frequent(x)
    st.header("Language Detected:")
    plural(x)
    st.header("Converted to Chinese:")
    lang(x)
    # st.header("Part-of-speech tag:")
    # postag(x)
    st.header("Entity recognition:")
    entity_recognition(x)
    st.header("Chunking text:")
    chunking(x)
    st.header("Text correction:")
    correction(x)
###################### Backend #########################
@st.cache
def load_data():
    return 

if __name__ == "__main__":
    main()