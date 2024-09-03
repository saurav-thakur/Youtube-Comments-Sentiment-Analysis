import numpy as np
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()


def remove_retweets_rt(text):
    rt_pattern = re.compile(r'RT ')
    rt_removed = rt_pattern.sub("",text)
    return rt_removed

def remove_emojis(text):
    emoji_pattern = re.compile("&[!@#$%^&*0-9a-zA-Z]+;")
    emoji_removed = emoji_pattern.sub("",text)
    return emoji_removed

def remove_html_links(text):
    html_pattern = re.compile("https?:\/\/.*?[\s+]")
    html_removed = html_pattern.sub("",text)
    return html_removed

def correct_spelling_mistakes(text):
    text_blob = TextBlob(text)
    return str(text_blob.correct())

def remove_punctuations(text):
    text_punc = "".join([i for i in text if i not in string.punctuation])
    return text_punc

def remove_usernames(text):
    pattern = re.compile(r'@[a-zA-Z0-9_-]+')
    user_names_removed = pattern.sub("",text)
    return user_names_removed

def remove_numbers(text):
    pattern = re.compile(r'[0.0-9.0]+')
    numbers_removed = pattern.sub("",text)
    return numbers_removed

def remove_unwanted_whitespaces(text):
    pattern = re.compile(r'\s{2,}')
    whitespace_morethan_two_removed = pattern.sub(" ",text)
    return whitespace_morethan_two_removed

def stopwords_removal(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def tokenization(text):
    tokens = word_tokenize(text)
    return tokens

def lemmatize(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return text

def preprocess_data(text):
    text = text.lower()
    text = remove_retweets_rt(text)
    text = remove_emojis(text)
    text = remove_html_links(text)
    text = remove_punctuations(text)
    text = remove_usernames(text)
    text = remove_numbers(text)
    text = remove_unwanted_whitespaces(text)
    text = stopwords_removal(text)
    text = lemmatize(text)
    # text = tokenization(text)
    return text