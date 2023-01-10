#!/usr/bin/env python
# -*- coding: utf-8 -*-

from decimal import DivisionByZero
import numpy as np
import random as rd
import pandas as pd
#from textattack.augmentation import WordNetAugmenter, EmbeddingAugmenter, EasyDataAugmenter, CharSwapAugmenter
from scipy.stats import kendalltau, multinomial
from sklearn.metrics.pairwise import pairwise_distances
from random import randrange, randint, choices, uniform, random
import spacy.cli
import spacy
import pyfolding as pf
import en_core_web_lg
import en_core_web_md
import math
import re
from stemming.porter2 import stem
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

en_core_web_lg =  spacy.load("en_core_web_md")#en_core_web_lg.load()


def l0_distance_text(text1, text2):
    WORD = re.compile(r"\w+")
    
    def get_l0(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        """if len(intersection) > 0:
            print("intersection", intersection)"""
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
    """if l0 != 0: 
        print("l0", l0)"""
    return l0

def pairwise_distance_text(text1, text2):
    vect = TfidfVectorizer(min_df=1, stop_words="english")                                                                                                                                                                                                   
    try:
        tfidf = vect.fit_transform([text1, text2])                                                                                                                                                                                                                       
    except ValueError as ec:
        """print(ec)
        print("TEST")
        print(text1)
        print()
        print(text2)"""
        return 0
    pairwise_similarity = tfidf * tfidf.T
    pairwise_similarity_array = pairwise_similarity.toarray() 
    return pairwise_similarity_array[0][1]

def cosine_distance_text(text1, text2):
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

def get_distances(x1, x2, metrics=None):
    x1, x2 = x1.reshape(1, -1), x2.reshape(1, -1)
    euclidean = pairwise_distances(x1, x2)[0][0]
    same_coordinates = sum((x1 == x2)[0])
    
    #pearson = pearsonr(x1, x2)[0]
    #kendall = kendalltau(x1, x2)
    out_dict = {'euclidean': euclidean,
                'sparsity': x1.shape[1] - same_coordinates#,
                #'kendall': kendall
               }
    return out_dict        

def generate_inside_ball(center, segment, n, m=1, nlp=None):
    """
    Args:
        "center" corresponds to the target instance to explain
        Segment corresponds to the size of the hypersphere
        n corresponds to the number of instances generated
    """
    #print("m: ", m)
    # vocabulary = [w for w in nlp.vocab if w.prob >= -15]
    try:
        array_center = center.toarray()[0]
    except AttributeError:
        array_center = center
    index_words = np.where(array_center>0)
    tab = np.zeros(array_center.shape[0])
    for index in index_words:
        tab[index] = array_center[index]
    center = np.array([tab] * n)
    for i in range(n):
        for j in range(m):
            testvalue = randrange(array_center.shape[0])
            # testvalue = randrange(len(vocabulary))
            # random_word = vocabulary[testvalue].text
            index_replaced = np.random.choice(index_words[0], size=1, replace=False)
            center[i][testvalue] = 1
            center[i][index_replaced] = center[i][index_replaced] - 1
    return center

class Neighbors:
    def __init__(self, nlp_obj):
        self.word_prob = -15
        self.nlp = nlp_obj
        #self.to_check = [w for w in self.nlp.vocab if w.prob >= self.word_prob and w.has_vector]# list of all the word to check
        self.to_check = []
        for w in self.nlp.vocab:
            if w.prob >= self.word_prob and w.has_vector:
                self.to_check.append(w)
        
        if not self.to_check:
            raise Exception('No vectors. Are you using en_core_web_sm? It should be en_core_web_lg')
        
        self.n = {}#store the neighbors here

    def neighbors(self, word):
        orig_word = word
        if word not in self.n:
            if word not in self.nlp.vocab.strings:
                self.n[word] = []
            elif not self.nlp.vocab[word].has_vector:
                self.n[word] = []
            else:
                word = self.nlp.vocab[word]
                queries = [w for w in self.to_check
                            if w.is_lower == word.is_lower]
                if word.prob < self.word_prob:
                    queries += [word]
                by_similarity = sorted(
                    queries, key=lambda w: word.similarity(w), reverse=True)[1:]
                self.n[orig_word] = [(self.nlp(w.orth_)[0], word.similarity(w))
                                     for w in by_similarity[:1000]]
                                    #  if w.lower_ != word.lower_]
        return self.n[orig_word]

#First adpation to the problem

def perturb_sentence_v1(text, neighbors,max_sim=0.9,
                     forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET', 'PART']):
    ''' The function will change all words of the sentence one by one until the similarity with original text reach  max_sim'''
    # words is a list of words (must be unicode)
    # max_sim is the spacy similarity to reach
    # neighbors must be of utils.Neighbors
    # nlp must be spacy
    # forbidden: forbidden lemmas
    # forbidden_tags, words: self explanatory
    # pos: which POS to change

    tokens = neighbors.nlp(text)
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = [x.text for x in tokens]
    new_tokens=neighbors.nlp(text)
    step=0
    
    while new_tokens.similarity(tokens)>max_sim and step<200:
      i=0
      while new_tokens.similarity(tokens)>max_sim and i<len(tokens):
        raw = [x.text for x in new_tokens if x.text!="'"]
        t=tokens[i]
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            r_neighbors = [
                x[0].text
                for x in neighbors.neighbors(t.text)
                if x[0].tag_ == t.tag_]
            if len(r_neighbors)<=step:
                i+=1
                continue
            else:        
                raw[i] = r_neighbors[step]
                raw = ' '.join(raw)
                new_tokens=neighbors.nlp(raw)

        i+=1
        #when all words has been modified, we start again with further neighbors
      step+=2
    return (new_tokens.text,new_tokens.similarity(tokens))

#introducting random in the fuction

def perturb_sentence_random(text, neighbors,max_sim=0.9,
                     forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET', 'PART']):
    # words is a list of words (must be unicode)
    # neighbors must be of utils.Neighbors
    # nlp must be spacy
    # forbidden: forbidden lemmas
    # forbidden_tags, words: self explanatory
    # pos: which POS to change

    tokens = neighbors.nlp(text)
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = [x.text for x in tokens]
    new_tokens=neighbors.nlp(text)

   
    i=0
    while new_tokens.similarity(tokens)>max_sim and i<len(tokens):
        raw = [x.text for x in new_tokens if x.text!="'"]
        t=tokens[i]
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            r_neighbors = [
                x[0].text
                for x in neighbors.neighbors(t.text)
                if x[0].tag_ == t.tag_]
 
            step=abs(int(np.random.normal(0,len(r_neighbors)/3)))#choosing random neighbour
            while step>len(r_neighbors):
                step=abs(int(np.random.normal(0,len(r_neighbors)/3)))
            raw[i] = r_neighbors[step]
            raw = ' '.join(raw)            
            new_tokens=neighbors.nlp(raw)
            if verbose:
                print(raw,new_tokens.similarity(tokens))
        i+=1
    return (new_tokens.text,new_tokens.similarity(tokens))



def perturb_sentence_multiple_gs(text, nb_word_to_modify, neighbors=None, max_sim=0.9, n_iter=500, max_iter=1000,
                     forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET', 'PART']):
    # text is a list of words (must be unicode)
    # n_iter = number of sentence to return
    # neighbors must be of utils.Neighbors
    # nlp must be spacy
    # forbidden: forbidden lemmas
    # forbidden_tags, words: self explanatory
    # pos: which POS to change
    sentences=pd.DataFrame([[text,0,1]],columns=['Text','L0','Similarity'])
    
    if neighbors == None:
        neighbors = Neighbors(en_core_web_lg.load())

    tokens = neighbors.nlp(text)
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = [x.text for x in tokens]
    
    to_modify=[]
    k=0
    for i, t in enumerate(tokens):
          # determine all the word we can modify
          if (t.text not in forbidden_words and t.pos_ in pos and
                  t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
              to_modify.append(i)
    n_tot=len(to_modify)

    print("max iter ici", max_iter)
    while k<n_iter or k < max_iter:
        #print("k = ", k)
        #n=np.random.randint(1,n_tot+1) # number of word to modify
        #print("number of words to modify", nb_word_to_modify)
        pick=rd.sample(to_modify, nb_word_to_modify) # words to modify
        #print("word to modify", pick)
        raw = [x.text for x in tokens if x.text!="'"]
        for i in pick:
            t=tokens[i]
            r_neighbors = [
                    x[0].text
                    for x in neighbors.neighbors(t.text)
                    if x[0].tag_ == t.tag_]
            if len(r_neighbors) == 0:
                continue
            step=abs(int(np.random.normal(0,len(r_neighbors)/3)))
            while step>=len(r_neighbors):
                step=abs(int(np.random.normal(0,len(r_neighbors)/3)))#neighbour to take
        
            raw[i] = r_neighbors[step]
      
        raw = ' '.join(raw)         
        new_tokens=neighbors.nlp(raw)
        #print("phrase modifiée", new_tokens)
        #print("phrase de départ", tokens)
        #print("score de similarité :", new_tokens.similarity(tokens), "sur", max_sim)
        if new_tokens.similarity(tokens)<max_sim:
            continue
      
        k+=1
        sentences.loc[k]=[raw,nb_word_to_modify,new_tokens.similarity(tokens)]
        #sentences.loc[k]=[raw,n,new_tokens.similarity(tokens)]
    
    return sentences

def max_pert(text,neighbors,forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET', 'PART']):
  
    '''This function take the farthest neighbour for each word, to appoach and return the lowest similarity'''
    tokens = neighbors.nlp(text)
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = [x.text for x in tokens]
   
    new_tokens=neighbors.nlp(text)
  
    for i, t in enumerate(tokens):
        raw = [x.text for x in new_tokens if x.text!="'"]

        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            r_neighbors = [
                x[0].text
                for x in neighbors.neighbors(t.text)
                if x[0].tag_ == t.tag_]

            
            raw[i] = r_neighbors[-1]
            raw = ' '.join(raw)       
            new_tokens=neighbors.nlp(raw)
    #return raw   
    return (new_tokens.text,new_tokens.similarity(tokens))

def corect_sim(sim):
    # compute a similarity relative to the maximum similarity m
    max_pert=m[1]
    s=(1-max_pert)-(1-sim)
    s/=(1-max_pert)
    return s

if __name__ == "__main__":
    # Test of data augmenter (generates randomly perturb sentences)
    """
    text = "This is a text to test the algorithm made by me."
    print(text)
    aug = EmbeddingAugmenter(pct_words_to_swap=0.5,transformations_per_example=2)
    text_aug = aug.augment(text)
    print("Texte modifié par embedding augmenter", text_aug)
    text2="Texas lawyer Rod Ponton was left flummoxed when he discovered his face was appearing as a cat during a court session on Zoom. As his assistant tried to rectify the issue, he can be heard saying, I'm here live, I'm not a cat. Tweeting about the incident, Judge Roy Ferguson, who presided over the session, said it showed the legal community's effort to continue representing their clients in these challenging times."
    text2_aug = aug.augment(text2)
    print("deuxième texte modifié par l'augmenter", text_aug)

    aug_easy = EasyDataAugmenter()
    text_aug_easy = aug_easy.augment(text)
    print("premier texte modifier par easy data augmenter", text_aug_easy)
    """
    nlp = en_core_web_lg.load()
    #doc1=nlp(text)
    #print("similarité entre 2 textes", doc1.similarity(nlp(aug.augment(text)[0])))
    """
    s1 = 'The clouds moved like diaphanous folds in a gentle breeze.'
    s2 = 'The clouds moved like gossamer folds in a gentle breeze.'
    s3 = 'The clouds moved like translucent folds in a gentle breeze.'
    s_list = [s1, s2, s3]
    for s in s_list:
    doc = nlp(s)
    for token in doc:
        print(token.orth_, token.prob)

    # example form the docs

    doc1 = nlp("I like salty fries and hamburgers.")
    doc2 = nlp("Fast food tastes very good.")

    # Similarity of two documents
    print(doc1, "<->", doc2, doc1.similarity(doc2))
    # Similarity of tokens and spans
    french_fries = doc1[2:4]
    burgers = doc1[5]
    print(french_fries, "<->", burgers, french_fries.similarity(burgers))
    word = nlp.vocab['hamburger']
    print(word.prob)
    """

    """
    def neigbor_text(text,w_to_replace,max_sim,nlp_obj):
        neighbors
        text=npl_obj(text)
        to_check=[w for w in nlp.vocab if w.prob >= -15 and w.has_vector]
        words=[nlp_obj.vocab(w) for w in w_to_replace]
        sim={word:}
    """

    def sim_text(text1,text2,nlp=nlp):
        doc1=nlp(text1)
        doc2=nlp(text2)
        return doc1.similarity(doc2)

    # Comment for git repo
    """
    print(sim_text("I like salty fries and hamburgers.","I like salty fries and steak."))
    #Neighbors(nlp).neighbors('fries')
    #get_neighbors('',nlp)
    print("step 1")
    N=Neighbors(nlp)
    print(perturb_sentence_v1("I like salty fries and hamburgers.", N))
    print("perturb sentence V1 done")
    d=perturb_sentence_multiple("I like salty fries and hamburgers.", N, n_iter=100)
    print(d)
    print("perturb sentence multiple done")
    print(max_pert("I like salty fries and hamburgers.", N))
    print("max pert done")
    print(d.head(20))
    #d['Score']=d['Similarity'].apply(corect_sim)
    #print(d.loc[d['Score']==min(d['Score'])])
    L1=d.loc[d['L0']==1]
    print(L1)
    #print(L1.loc[d['Score']==min(L1['Score'])])
    """
    doc1 = nlp("I like salty fries and hamburgers.").vector
    doc2 = nlp("This is a good book.").vector
    doc3 = nlp("C'est le meilleur joueur rocket league").vector
    aug = EmbeddingAugmenter(pct_words_to_swap=0.5,transformations_per_example=10)
    text_aug = aug.augment("This is a test of data augmenter")
    print(text_aug)
    doc4 = np.array([nlp(x).vector for x in text_aug])
    #doc4 = np.array(text_aug)
    print(type(doc1))
    print("size of the array of perturbed sentences", doc4.shape)
    docs = np.array([doc1, doc2, doc3])
    docs = np.concatenate((docs, doc4), axis=None)
    #y = doc1.astype(np.float)
    print("size ", docs.shape)
    results = pf.FTU(docs, routine="python")
    multimodal_results = results.folding_statistics>1
    print("multimodal results", multimodal_results)
