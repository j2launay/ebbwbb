#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import en_core_web_md
from random_word import RandomWords
from nltk.corpus import wordnet
from itertools import chain
import random
import warnings
import time
warnings.simplefilter(action='ignore', category=FutureWarning)

# Python3 hack
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    def unicode(s):
        return s

class Neighbors:
    """
    Args:   Class use to store the closest neighbors for Growing Language
    """
    def __init__(self, nlp_obj):
        """
        Args:   nlp_obj: a SpaCy model
        """
        self.word_prob = -15
        self.nlp = nlp_obj
        self.to_check = []
        for w in self.nlp.vocab:
            if w.prob >= self.word_prob and w.has_vector:
                self.to_check.append(w)

        if not self.to_check:
            raise Exception('No vectors. Are you using en_core_web_sm? It should be en_core_web_lg')
        
        self.n = {}#store the neighbors here

    def neighbors(self, word):
        """
        Args:   word: word we want to store the closest neighbors according to a spaCy model
        """
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

class Counterfactuals:
    """
    class to generate text counterfactuals based on GrowingLanguage, GrowingNet, SEDC or random
    """
    def __init__(self,
                vector_to_interprete,
                obs_to_interprete,
                prediction_fn,
                method="growing_language",
                target_class=None,
                n_in_layer=200,
                first_radius=0.01,
                dicrease_radius=2,
                verbose=False,
                min_counterfactual_in_sphere=10,
                nlp=None,
                corpus=False
                ):
        """
        Args:
            vector_to_interprete: 
            obs_to_interprete: instance whose prediction is to be interpreded
            prediction_fn: prediction function, must return an integer label
            method: the counterfactual method use
            target_class: if None chose the closest one in multiclass situation otherwise, 
                        search for a specific counterfactual
            n_in_layer: Number of sentences generated at each round
            first_radius: initalisation of the similarity to the closest word
            dicrease_radius: Coefficient to increase similarity to the closest word
            min_counterfactual_in_sphere: Number of minimum counterfactual generated to stop the generation
            nlp: SpaCy model for GrowingLanguage
            corpus: input dataset use to pick words inside to use as replacement for the random method
        """
        self.vector_to_interprete = vector_to_interprete
        self.obs_to_interprete = obs_to_interprete

        self.prediction_fn = prediction_fn
        self.y_obs = prediction_fn(vector_to_interprete)
        if target_class == None:
            # Case of searching for the closest class (multi class)
            self.target_other = True
            target_class = self.y_obs[0]
        else:
            self.target_other = False
        self.target_class = target_class

        #Hyperparameters to increase the spheres
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.dicrease_radius = dicrease_radius
        
        self.verbose = verbose
        self.min_counterfactual_in_sphere = min_counterfactual_in_sphere
        if int(self.y_obs) != self.y_obs:
            raise ValueError("Prediction function should return a class (integer)")
        self.method = method

        self.corpus = corpus
        self.nlp = nlp
        self.tokens = self.nlp(str(obs_to_interprete))
        self.nb_word_in_text = len(self.tokens)
        if 'growing_net' in self.method:
            self.growing_net_synonyms = get_growing_net_set(self.tokens)
        
    def find_counterfactual(self):
        """
        Finds the decision border and returns the closest counterfactual as explanation
        """
        ennemies_, self.radius, iteration, similarities = self.exploration()
        if "growing_language" in self.method or "growing_net" in self.method:
            index_similarities = np.argsort(similarities)[0]
            ennemies_ = np.array(ennemies_)[index_similarities]
            closest_ennemy_ = ennemies_[-1]
        else:
            closest_ennemy_ = ennemies_[-1]
        
        self.e_star = closest_ennemy_
        return closest_ennemy_, ennemies_, self.radius, iteration
    
    
    def exploration(self):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing layers.
        The function use to generate artificial text depends on self.method
        """
        n_ennemies_, nb_word_modified = 0, 1
        radius_ = self.first_radius
        ennemies = []

        if self.verbose == True: 
            print("Exploring...")
        iteration = 2
        step_ = (self.dicrease_radius - 1) * radius_/2.0
        if 'growing_net' not in self.method:
            neighbors = Neighbors(self.nlp)

        while n_ennemies_ <= self.min_counterfactual_in_sphere:
            if self.verbose: 
                print("using the " + self.method + " method to find counterfactual")
            if 'growing_language' in self.method: 
                radius_growing_language = min(0.5, radius_*10)
                if radius_growing_language == 0.5 and iteration >= self.nb_word_in_text:
                    raise Exception("growing language method is not able to find closest counterfactual...")
                nb_word_modified = min(self.nb_word_in_text, max(1, self.nb_word_in_text // 10) * iteration)
                layer = perturb_sentence_language_model(self.obs_to_interprete, nb_word_modified, neighbors, max_sim = 0.9-radius_growing_language, \
                    n_iter=self.n_in_layer, verbose=self.verbose)
                preds_ = np.array(self.prediction_fn(layer['Text'].values.tolist()))
                similarities = np.array(layer['Similarity'].values.tolist())[preds_]
                layer = layer['Text'].values.tolist()
            
            elif 'growing_net' in self.method:
                layer = self.perturb_sentence_growing_net(self.obs_to_interprete, iteration, n_iter=self.n_in_layer)
                preds_ = self.prediction_fn(layer['Text'].values.tolist())
                similarities = np.array(layer['Similarity'].values.tolist())[preds_]
                layer = layer['Text'].values.tolist()
            
            elif 'sedc' in self.method:
                layer = perturb_sentence_sedc(self.obs_to_interprete, iteration, self)
                preds_ = self.prediction_fn(layer)
            
            else:
                nb_word_modified = min(self.nb_word_in_text, max(1, self.nb_word_in_text // 10) * iteration)
                if iteration >= 30 and nb_word_modified == self.nb_word_in_text:
                    raise Exception("random method is not able to find closest counterfactual...")
                layer = self.perturb_sentence_random(self.n_in_layer, nb_word_modified, corpus=self.corpus)
                preds_ = np.array(self.prediction_fn(layer))
            
            preds_ = [np.where(preds_ != self.target_class)]
            if ennemies != []:
                ennemies = np.concatenate((ennemies, np.array(layer)[preds_]), axis=None)
            else:
                ennemies = np.array(layer)[preds_]
            n_ennemies_ = ennemies.shape[0]
            radius_ = radius_ + step_
            iteration += 1
            if self.verbose: 
                print("n_ennemies", n_ennemies_)
                print("iteration ", iteration)
        if self.verbose:
            print("Final radius: ", (radius_ - step_, radius_))
            print("Final number of ennemies: ", n_ennemies_)
        return ennemies.tolist(), radius_, iteration, np.array(similarities)[preds_] if ('growing_language' in self.method or 'growing_net' in self.method) else None

    def perturb_sentence_random(self, nb_sentences, nb_words, corpus=False):
        """
        Perturb sentence by replacing randomly words from the input text
        Args:   nb_sentences: number of artificial sentences generated
                nb_words: number of words replaced in the input text
                corpus: if True replace only with words the most frequent from the input dataset
                        if False replace with words generated by the RandomWords Library (need internet connection)
        Output: Array of artificial texts
        """
        if self.verbose: 
            print('nb sentences generated', nb_sentences)
            print("nb words modified", nb_words, "over", self.nb_word_in_text)
        r = RandomWords()
        layer = []
        processed = self.nlp(unicode(self.obs_to_interprete))
        target_sentence = [x.text for x in processed]        
        nb_words = min(nb_words, len(target_sentence))
        random_words = []
        if corpus != False:
            # Pick randomly sentences from the input dataset and compute the tfidf of words in order to replace with most frequent words
            docs = random.sample(range(0, self.corpus[0].shape[0]), nb_words)
            feature_names = self.corpus[1].get_feature_names()
            for doc in docs:
                feature_index = self.corpus[0][doc,:].nonzero()[1]
                tfidf_scores = zip(feature_index, [self.corpus[0][doc, x] for x in feature_index])
                highest_tf_idf = 0
                for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
                    if s > highest_tf_idf:
                        word_highest_tf_idf = w
                random_words.append(word_highest_tf_idf)
        else:    
            while len(random_words) < max(nb_sentences, nb_words):
                # Ensure to generate a minimum of words with the RandomForest library
                try:
                    random_words += r.get_random_words(hasDictionaryDef=True)
                except TypeError:
                    continue
                except:
                    # Hack to wait before internet connection issue is solved
                    time.sleep(5)
        for i in range(int(nb_sentences)):
            # Replace randomly words from the original text with words from the set of words generated before in the function
            perturb_sentence = target_sentence[:]
            index_remove_words = random.sample(range(0, len(target_sentence)), nb_words)
            for index in index_remove_words:
                perturb_sentence.insert(index, random_words[random.randint(0, len(random_words)-1)])
                try:
                    perturb_sentence.pop(index+1)
                except IndexError:
                    perturb_sentence.pop()
            layer.append(' '.join(perturb_sentence))
        return layer

    def perturb_sentence_growing_net(self, text, nb_words_to_modify, n_iter=500):
        """
        Function to generate n_iter artificial sentences based on the WordNet knowledge bases and an input text 
        Args:   text: target sentence / list of words (must be unicode)
                nb_word_to_modify: number of word to replace for each artificial sentence
                n_iter: number of artificial sentences to return
        Return: DataFrame of artificial texts and its corresponding wup similarity
        """
        growing_net_synonyms = self.growing_net_synonyms
        try:
            processed = self.nlp(unicode(text))
        except TypeError:
            processed = self.nlp(unicode(str(text)))
        target_sentence = [x.text for x in processed]
        if nb_words_to_modify > len(target_sentence) + 2:
            raise Exception("growing_net is not able to find counterfactual or generate enough instances")
        # Ensure to modify maximum the length of the text    
        nb_words_to_modify = min(nb_words_to_modify, len(target_sentence))
        layer, similarity = [], []
        for i in range(int(n_iter)):
            # Copy the target sentence into perturb sentence  
            perturb_sentence = target_sentence[:]
            # set of value corresponding to the indexes of the words to modify
            index_remove_words = random.sample(range(0, self.nb_word_in_text), nb_words_to_modify)
            for index in index_remove_words:
                try:
                    word = self.tokens[index]
                    # Add a random word from the set of similar words associated to the input words
                    perturb_sentence.insert(index, list(growing_net_synonyms[word.text])[random.randint(0, len(growing_net_synonyms[word.text])-1)])
                    # Remove the target word
                    try:
                        perturb_sentence.pop(index+1)
                    except IndexError:
                        perturb_sentence.pop()
                except ValueError:
                    continue
            tot_sim = 0
            for perturb, target in zip(perturb_sentence, target_sentence):
                # Compute the similarity between the artificial sentence generated and the original input
                try:
                    if perturb != target:
                        temp_dist = wordnet.synsets(perturb)[0].wup_similarity(wordnet.synsets(target)[0])
                        temp_dist = 0 if temp_dist is None else temp_dist
                        tot_sim += temp_dist
                    else:
                        tot_sim += 1
                except:
                    print("not able to compute wup similarity (these must be two different types of words)")
            # Weight the wup similarity by the length of the sentence
            tot_sim = tot_sim / len(target_sentence)
            similarity.append(tot_sim)
            layer.append(' '.join(perturb_sentence))
        d = {'Text':layer, 'Similarity':similarity}
        to_return = pd.DataFrame(d)
        return to_return

def perturb_sentence_language_model(text, nb_word_to_modify, neighbors=None, max_sim=0.9, n_iter=500, max_iter=1000,
                    pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET', 'PART'], verbose=False):
    """
    Function to generate n_iter artificial sentences based on a language model and an input text 
    Args:   text: target sentence / list of words (must be unicode)
            nb_word_to_modify: number of word to replace for each artificial sentence
            n_iter: number of artificial sentences to return
            neighbors: must be of Neighbors class
            nlp: must be spacy
            pos: which POS to change
    Return: Dataframe of artificial texts and language model similarity to input text
    """
    sentences=pd.DataFrame([[text,0,1]],columns=['Text','L0','Similarity'])
    if neighbors == None:
        print("neighbors is None")
        neighbors = Neighbors(en_core_web_md.load())

    tokens = neighbors.nlp(str(text))
    pos = set(pos)
    
    to_modify=range(len(tokens))
    k, k_max_iter = 0, 0
    n_tot=len(to_modify)
    dictionnaire = {}
    for word_to_modify in to_modify:
        # Prepare a dictionnaire containing for each word in the input text, its corresponding 
        # set of similar words.
        nb_word_in_dict = 0
        dictionnaire[tokens[word_to_modify].text] = []
        for x in neighbors.neighbors(tokens[word_to_modify].text):
            if x[0].tag_ == tokens[word_to_modify].tag_:
                dictionnaire[tokens[word_to_modify].text].append(x[0].text)
                nb_word_in_dict += 1
                if nb_word_in_dict > n_iter:
                    break
                
    # Ensure to modify maximum the length of the text
    nb_word_to_modify = min(nb_word_to_modify, n_tot)
    if verbose:
        print("nb word to modify", nb_word_to_modify, "over", n_tot, "words in the target sentence")
    while k < n_iter and k_max_iter < max_iter:
        k_max_iter += 1
        # set of values corresponding to the indexes of the words to modify
        pick=random.sample(to_modify, nb_word_to_modify) 
        raw = [x.text for x in tokens if x.text!="'"]
        for i in pick:
            t=tokens[i]
            r_neighbors = dictionnaire[t.text]
            if len(r_neighbors) == 0:
                continue
            # Select randomly a word in the corresponding set of word to replace the word from the input text
            step=abs(int(np.random.normal(0,len(r_neighbors)/3)))
            while step>=len(r_neighbors):
                step=abs(int(np.random.normal(0,len(r_neighbors)/3)))

            raw[i] = r_neighbors[step]
        raw = ' '.join(raw)         
        new_tokens=neighbors.nlp(raw)
        if new_tokens.similarity(tokens)<max_sim:
            continue
        k+=1
        sentences.loc[k]=[raw,nb_word_to_modify,new_tokens.similarity(tokens)]
    return sentences  

def perturb_sentence_sedc(text, iteration, counterfactual_class, n_iter=500):
    """
    Function that generate artificial sentences by replacing randomly words from the target text with the mask "UNKWORDZ"
    Args:   text: input raw document
            iteration: number of words replaced by a mask
            counterfactual_class: a Counterfactual class that trained a natural lanugage model to process the input text
    Return: Array of artificial texts
    """
    # Split the target text document into an array of words
    processed = counterfactual_class.nlp(unicode(text))
    target_sentence = [x.text for x in processed]
    if iteration > (len(target_sentence) * 2):
        raise Exception("sedc is not able to find counterfactual or generate enough instances")
    layer = []
    # Ensure to modify maximum the length of the text
    nb_words_replace = min(iteration, len(target_sentence))
    for i in range(int(n_iter)):
        perturb_sentence = target_sentence[:]
        # set of values corresponding to the indexes of the words to modify
        index_remove_words = random.sample(range(0, len(target_sentence)), nb_words_replace)
        for index in index_remove_words:
            # replace words with a mask
            perturb_sentence.insert(index, "UNKWORDZ")
            try:
                perturb_sentence.pop(index+1)
            except IndexError:
                perturb_sentence.pop()
        layer.append(' '.join(perturb_sentence))
    return layer
    
def get_growing_net_set(tokens):
    """
    Generate a set of similar words according to WordNet for each input words in the target document
    Args:   tokens: the input phrase preprocessed by a Spacy Language Model
    Return: Dictionnary composed of inpout words as a key and similar words as the values
    """
    growing_net_synonyms = {}
    for word in tokens:
        word = word.text
        synonyms_antonyms = []
        try:
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    synonyms_antonyms.append(lemma.name())    #add the synonyms
                    if lemma.antonyms():    #When antonyms are available, add them into the list
                        synonyms_antonyms.append(lemma.antonyms()[0].name())
            growing_net_synonyms[word] = synonyms_antonyms
            try:
                synonyms = wordnet.synsets(word)[0].hyponyms()
                if synonyms != []:
                    growing_net_synonyms[word] = growing_net_synonyms[word] | set(chain.from_iterable([word.lemma_names() for word in synonyms]))
            except:
                pass
            try:
                synonyms = wordnet.synsets(word)[0].hypernyms()
                if synonyms!= []:
                    growing_net_synonyms[word] = growing_net_synonyms[word] | set(chain.from_iterable([word.lemma_names() for word in synonyms]))
            except:
                pass
        except:
            pass
        growing_net_synonyms[word] = set(growing_net_synonyms[word])
    return growing_net_synonyms
