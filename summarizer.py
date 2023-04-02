# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 19:05:42 2023

@author: hosse
"""

import streamlit as st
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

# Function to clean and preprocess text
def preprocess(text):
    sentences = sent_tokenize(text)
    stop_words = stopwords.words('english')
    clean_sentences = []

    for sentence in sentences:
        words = sentence.split()
        words = [word.lower() for word in words if word not in stop_words]
        clean_sentences.append(' '.join(words))

    return clean_sentences

# Function to calculate similarity between sentences
def sentence_similarity(sent1, sent2):
    sent1 = [word.lower() for word in sent1.split()]
    sent2 = [word.lower() for word in sent2.split()]
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in sent1:
        vector1[all_words.index(word)] += 1

    for word in sent2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)

# Function to generate summary
def generate_summary(text, num_sentences):
    sentences = preprocess(text)
    sentence_similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            sentence_similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    summary = ''
    for i in range(num_sentences):
        summary += ranked_sentences[i][1] + ' '

    return summary

# Streamlit app
st.title("Scientific Article Summarizer")

text = st.text_area("Enter your scientific article here:")
num_sentences = st.slider("Select number of sentences for summary", min_value=1, max_value=10)

if st.button("Generate Summary"):
    summary = generate_summary(text, num_sentences)
    st.write(summary)
