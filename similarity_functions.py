#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spacy.cli
import spacy
import math
import re
from stemming.porter2 import stem
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

en_core_web_lg =  spacy.load("en_core_web_lg")

def l0_similarity_text(text1, text2):
    WORD = re.compile(r"\w+")
    
    def get_l0(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        try:
            l0_vec1 = len(intersection) / len(set(vec1.keys()))
        except ZeroDivisionError:
            l0_vec1 = 0
        try:
            l0_vec2 = len(intersection) / len(set(vec2.keys()))
        except ZeroDivisionError:
            l0_vec2 = 0
        return (l0_vec1 + l0_vec2) / 2

    def text_to_vector(text):
        words = WORD.findall(text)
        new_words = []
        for word in words:
            temp_word = stem(word)
            new_words.append(temp_word)
        return Counter(new_words)

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    l0 = get_l0(vector1, vector2)
    return l0

def pairwise_similarity_text(text1, text2):
    vect = TfidfVectorizer(min_df=1, stop_words="english")                                                                                                                                                                                                   
    try:
        tfidf = vect.fit_transform([text1, text2])                                                                                                                                                                                                                       
    except ValueError as ec:
        print("no word present in both texts.")
        return 0
    pairwise_similarity = tfidf * tfidf.T
    pairwise_similarity_array = pairwise_similarity.toarray() 
    return pairwise_similarity_array[0][1]

def cosine_similarity_text(text1, text2):
    WORD = re.compile(r"\w+")

    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def text_to_vector(text):
        words = WORD.findall(text)
        return Counter(words)

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine = get_cosine(vector1, vector2)

    return cosine 
