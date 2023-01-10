#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .utils.gs_utils import Neighbors
import numpy as np
import spacy
import pandas as pd
import en_core_web_md
from random_word import RandomWords
#from textattack.augmentation import WordNetAugmenter, EmbeddingAugmenter, EasyDataAugmenter, CharSwapAugmenter
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from nltk.corpus import wordnet
from itertools import chain
import random
import warnings
import time
from nltk import pos_tag
warnings.simplefilter(action='ignore', category=FutureWarning)

# Python3 hack
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    def unicode(s):
        return s

class GrowingFields:
    """
    class to fit the Growing Fields algorithm
    
    Inputs: 
    obs_to_interprete: instance whose prediction is to be interpreded
    prediction_fn: prediction function, must return an integer label
    caps: min max values of the explored area. Right now: if not None, the minimum and maximum values of the 
    """
    def __init__(self,
                vector_to_interprete,
                obs_to_interprete,
                prediction_fn,
                method="lessconcon",
                target_class=None,
                caps=None,
                n_in_layer=200,
                first_radius=0.01,
                dicrease_radius=2,
                sparse=True,
                verbose=False,
                min_counterfactual_in_sphere=10,
                nlp=None,
                corpus=False
                ):
        """
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
        self.caps = caps
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.dicrease_radius = dicrease_radius
        self.sparse = sparse
        
        self.verbose = verbose
        self.min_counterfactual_in_sphere = min_counterfactual_in_sphere
        if int(self.y_obs) != self.y_obs:
            raise ValueError("Prediction function should return a class (integer)")
        self.method = method

        self.corpus = corpus
        self.nlp = nlp
        self.tokens = self.nlp(str(obs_to_interprete))
        self.nb_word_in_text = len(self.tokens)
        pos_tagger = True if "tagger" in method else False
        if 'wordnet' in self.method and pos_tagger:
            self.wordnet_synonyms = get_wordnet_set_pos(self.tokens)
        elif 'wordnet' in self.method:
            self.wordnet_synonyms = get_wordnet_set(self.tokens)
    
    def get_distance_text(self, text1, text2):
        doc1=self.nlp(text1)
        doc2=self.nlp(text2)
        return doc1.similarity(doc2)
        
    def find_counterfactual(self):
        """
        Finds the decision border then perform projections to make the explanation sparse.
        """
        ennemies_, self.radius, iteration, similarities = self.exploration()
        #ennemies_ = sorted(ennemies_, 
        #                         key= lambda x: self.get_distance_text(self.obs_to_interprete, x))
        if "augmenter" in self.method or 'antonym' in self.method or 'translation' in self.method or 'gpt2' in self.method:
            ennemies_ = sorted(ennemies_, key= lambda x: self.get_distance_text(self.obs_to_interprete, str(x)))
            closest_ennemy_ = ennemies_[0]
            #farthest_ennemy = ennemies_[-1]
        elif "lessconcon" in self.method or "wordnet" in self.method:
            index_similarities = np.argsort(similarities)[0]
            ennemies_ = np.array(ennemies_)[index_similarities]
            #ennemies_ = ennemies_.sort_values(["Similarity", 'L0', 'Text'], ascending=False)
            #closest_ennemy_ = ennemies_.iloc[1]["Text"]
            #farthest_ennemy = ennemies_.iloc[-1]["Text"]
            closest_ennemy_ = ennemies_[-1]
        else:
            closest_ennemy_ = ennemies_[-1]
        
        self.e_star = closest_ennemy_
        return closest_ennemy_, ennemies_, self.radius, iteration
    
    
    def exploration(self):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
        """
        n_ennemies_, nb_word_modified = 0, 1
        radius_ = self.first_radius
        ennemies = []

        if self.verbose == True: 
            print("Exploring...")
        iteration = 2
        step_ = (self.dicrease_radius - 1) * radius_/2.0
        if not ('wordnet' in self.method or 'augmenter' in self.method or 'antonym' in self.method or 'translation' in self.method or 'gpt2' in self.method):
            neighbors = Neighbors(self.nlp)

        while n_ennemies_ <= self.min_counterfactual_in_sphere:
            if 'augmenter' in self.method:
                if self.verbose: print("radius", radius_)
                #augmenter = CharSwapAugmenter(pct_words_to_swap=radius_, transformations_per_example=self.n_in_layer)
                #augmenter = WordNetAugmenter(pct_words_to_swap=radius_, transformations_per_example=self.n_in_layer)
                #augmenter = EmbeddingAugmenter(pct_words_to_swap=radius_, transformations_per_example=self.n_in_layer)
                #layer = augmenter.augment(self.obs_to_interprete)
                layer = []
                aug = naw.SynonymAug(aug_src='wordnet')
                for i in range(self.n_in_layer):
                    layer.append(aug.augment(self.obs_to_interprete))
                preds_ = np.array(self.prediction_fn(layer))
                preds_ = np.where(preds_ != self.target_class)
            elif 'antonym' in self.method:
                if self.verbose: print('radius', radius_)
                aug = naw.AntonymAug()
                layer = []
                for i in range(self.n_in_layer):
                    aug_text = aug.augment(self.obs_to_interprete)
                    layer.append(aug_text)
                preds_ = np.array(self.prediction_fn(layer))
                preds_ = np.where(preds_ != self.target_class)

                test += 2
            elif 'translation' in self.method:
                if self.verbose: print('radius', radius_)
                back_translation_aug = naw.BackTranslationAug(
                    from_model_name='facebook/wmt19-en-de', 
                    to_model_name='facebook/wmt19-de-en'
                )
                layer = []
                for i in range(self.n_in_layer):
                    layer.append(back_translation_aug.augment(self.obs_to_interprete))
                preds_ = np.array(self.prediction_fn(layer))
                
                preds_ = np.where(preds_ != self.target_class)
            elif 'gpt2' in self.method:
                aug = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2')
                if self.verbose: print('radius', radius_)
                layer = []
                for i in range(self.n_in_layer):
                    layer.append(aug.augment(self.obs_to_interprete))
                preds_ = np.array(self.prediction_fn(layer))
                preds_ = np.where(preds_ != self.target_class)
            elif 'lessconcon' in self.method: 
                if self.verbose: 
                    print("using the perturb sentence by Jacques to find counterfactual in growing fields")
                    print("nb words modified in gs lessconcon", nb_word_modified)
                radius_less_concon = min(0.5, radius_*10)
                if radius_less_concon == 0.5 and iteration >= self.nb_word_in_text:
                    print("cgt method is not able to find closest counterfactual...")
                    test += 2
                nb_word_modified = min(self.nb_word_in_text, max(1, self.nb_word_in_text // 10) * iteration)
                layer = perturb_sentence_multiple(self.obs_to_interprete, nb_word_modified, neighbors, max_sim = 0.9-radius_less_concon, \
                    n_iter=self.n_in_layer, verbose=self.verbose)
                preds_ = np.array(self.prediction_fn(layer['Text'].values.tolist()))
                similarities = layer['Similarity'].values.tolist()
                layer = layer['Text'].values.tolist()
                preds_ = [np.where(preds_ != self.target_class)]
                similarities = np.array(similarities)[preds_]
            elif 'wordnet' in self.method:
                if self.verbose:
                    print("using the wordnet to generate setences in growing fields")
                    print("size of the sentence", self.nb_word_in_text)
                layer = perturb_sentence_wordnet(self.obs_to_interprete, iteration, self, n_iter=self.n_in_layer)
                preds_ = self.prediction_fn(layer['Text'].values.tolist())
                similarities = layer['Similarity'].values.tolist()
                layer = layer['Text'].values.tolist()
                preds_ = [np.where(preds_ != self.target_class)]
                similarities = np.array(similarities)[preds_]
            elif 'lime' in self.method:
                if self.verbose:
                    print("using lime to generate sentences in growing fields")
                    print("size of the sentence", self.nb_word_in_text)
                layer = perturb_sentence_lime(self.obs_to_interprete, iteration, self)
                preds_ = self.prediction_fn(layer)
                preds_ = [np.where(preds_ != self.target_class)]
            else:
                if self.verbose: 
                    print("using the concon method to find counterfactual in growing fields")
                    print("nb words modified in gs concon", nb_word_modified)
                nb_word_modified = min(self.nb_word_in_text, max(1, self.nb_word_in_text // 10) * iteration)
                if iteration >= 30 and nb_word_modified == self.nb_word_in_text:
                    print("baseline method is not able to find closest counterfactual...")
                    test += 2 
                layer = self.perturb_sentence_concon(self.n_in_layer, nb_word_modified, corpus=self.corpus)
                preds_ = np.array(self.prediction_fn(layer))
                preds_ = np.where(preds_ != self.target_class)
            if ennemies != []:
                ennemies = np.concatenate((ennemies, np.array(layer)[preds_]), axis=None)
            else:
                ennemies = np.array(layer)[preds_]
            n_ennemies_ = ennemies.shape[0]
            if self.verbose: print("n_ennemies", n_ennemies_)
            radius_ = radius_ + step_
            iteration += 1
            if self.verbose: print("iteration ", iteration)
        if self.verbose:
            print("Final radius: ", (radius_ - step_, radius_))
            print("Final number of ennemies: ", n_ennemies_)
        return ennemies.tolist(), radius_, iteration, similarities if ('lessconcon' in self.method or 'wordnet' in self.method) else None

    def perturb_sentence_concon(self, nb_sentences, nb_words, corpus=False):
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
                try:
                    random_words += r.get_random_words(hasDictionaryDef=True)
                except TypeError:
                    continue
                except:
                    time.sleep(5)
        
        for i in range(int(nb_sentences)):
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

    def perturb_sentence_marco(words, num_samples):
        data = np.zeros((num_samples, len(words)))
        for i in range(len(words)):
            if i in present:
                continue
            probs = [1 - perturber.pr[i], perturber.pr[i]]
            data[:, i] = np.random.choice([0, 1], num_samples, p=probs)
        data[:, present] = 1
        raw_data = []
        for i, d in enumerate(data):
            r = perturber.sample(d)
            print("r", d, r)
            data[i] = r == words
            raw_data.append(' '.join(r))

def perturb_sentence_multiple(text, nb_word_to_modify, neighbors=None, max_sim=0.9, n_iter=500, max_iter=1000,
                    forbidden=[], forbidden_tags=['PRP$'],
                    forbidden_words=['be'],
                    pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET', 'PART'], verbose=False):
    # text is a list of words (must be unicode)
    # n_iter = number of sentence to return
    # neighbors must be of utils.Neighbors
    # nlp must be spacy
    # forbidden: forbidden lemmas
    # forbidden_tags, words: self explanatory
    # pos: which POS to change
    sentences=pd.DataFrame([[text,0,1]],columns=['Text','L0','Similarity'])
    if neighbors == None:
        print("neighbors is None")
        neighbors = Neighbors(en_core_web_md.load())

    tokens = neighbors.nlp(str(text))
    """forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)"""
    pos = set(pos)
    raw = [x.text for x in tokens]
    
    to_modify=range(len(tokens))
    k, k_max_iter = 0, 0
    """for i, t in enumerate(tokens):
        # determine all the word we can modify
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            to_modify.append(i)"""
    n_tot=len(to_modify)
    dictionnaire = {}
    for word_to_modify in to_modify:
        nb_word_in_dict = 0
        dictionnaire[tokens[word_to_modify].text] = []
        for x in neighbors.neighbors(tokens[word_to_modify].text):
            if x[0].tag_ == tokens[word_to_modify].tag_:
                dictionnaire[tokens[word_to_modify].text].append(x[0].text)
                nb_word_in_dict += 1
                if nb_word_in_dict > n_iter:
                    break
                
    nb_word_to_modify = min(nb_word_to_modify, n_tot)
    if verbose:
        print("nb word to modify", nb_word_to_modify, "over", n_tot, "words in the target sentence")
    while k<n_iter and k_max_iter < max_iter:
        k_max_iter += 1
        pick=random.sample(to_modify, nb_word_to_modify) # words to modify
        raw = [x.text for x in tokens if x.text!="'"]
        for i in pick:
            t=tokens[i]
            r_neighbors = dictionnaire[t.text]
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
    return sentences  

def perturb_sentence_wordnet(text, nb_words_to_modify, growing_field, n_iter=500):
    if growing_field != None:
        wordnet_synonyms = growing_field.wordnet_synonyms
    try:
        processed = growing_field.nlp(unicode(text))
    except TypeError:
        processed = growing_field.nlp(unicode(str(text)))
    target_sentence = [x.text for x in processed]
    if nb_words_to_modify > len(target_sentence) + 2:
        print("wordnet is not able to find counterfactual or generate enough instances")
        print(wordnet_synonyms)
        test += 2
    nb_words_to_modify = min(nb_words_to_modify, len(target_sentence))
    layer, similarity = [], []
    for i in range(int(n_iter)):
        perturb_sentence = target_sentence[:]
        index_remove_words = random.sample(range(0, growing_field.nb_word_in_text), nb_words_to_modify)
        # TODO faire en sorte de ne pas changer tous les mots
        for index in index_remove_words:
            try:
                word = growing_field.tokens[index]
                perturb_sentence.insert(index, list(wordnet_synonyms[word.text])[random.randint(0, len(wordnet_synonyms[word.text])-1)])
                try:
                    perturb_sentence.pop(index+1)
                except IndexError:
                    perturb_sentence.pop()
            except ValueError:
                #print("This word has not got syn", word.text)
                continue
        tot_sim = 0
        for perturb, target in zip(perturb_sentence, target_sentence):
            try:
                if perturb != target:
                    temp_dist = wordnet.synsets(perturb)[0].wup_similarity(wordnet.synsets(target)[0])
                    temp_dist = 0 if temp_dist is None else temp_dist
                    tot_sim += temp_dist#wordnet.synsets(perturb)[0].wup_similarity(wordnet.synsets(target)[0])
                else:
                    tot_sim += 1
            except:
                print("not able to compute wup similarity (these must be two different types of words)")
        tot_sim = tot_sim / len(target_sentence)
        similarity.append(tot_sim)
        layer.append(' '.join(perturb_sentence))
    d = {'Text':layer, 'Similarity':similarity}
    to_return = pd.DataFrame(d)
    return to_return

def perturb_sentence_lime(text, iteration, growing_field, n_iter=500):
    processed = growing_field.nlp(unicode(text))
    target_sentence = [x.text for x in processed]
    if iteration > (len(target_sentence) * 2):
        print("lime is not able to find counterfactual or generate enough instances")
        test += 2
    layer = []
    nb_words_replace = min(iteration, len(target_sentence))
    for i in range(int(n_iter)):
        perturb_sentence = target_sentence[:]
        index_remove_words = random.sample(range(0, len(target_sentence)), nb_words_replace)
        for index in index_remove_words:
            perturb_sentence.insert(index, "UNKWORDZ")
            try:
                perturb_sentence.pop(index+1)
            except IndexError:
                perturb_sentence.pop()
        layer.append(' '.join(perturb_sentence))
    return layer
    
def get_wordnet_set(tokens):
    wordnet_synonyms = {}
    for word in tokens:
        word = word.text
        synonyms_antonyms = []
        try:
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    synonyms_antonyms.append(lemma.name())    #add the synonyms
                    if lemma.antonyms():    #When antonyms are available, add them into the list
                        synonyms_antonyms.append(lemma.antonyms()[0].name())
            wordnet_synonyms[word] = synonyms_antonyms#set(chain.from_iterable([word.lemma_names() for word in synonyms]))
            try:
                synonyms = wordnet.synsets(word)[0].hyponyms()
                if synonyms != []:
                    wordnet_synonyms[word] = wordnet_synonyms[word] | set(chain.from_iterable([word.lemma_names() for word in synonyms]))
            except:
                pass
            try:
                synonyms = wordnet.synsets(word)[0].hypernyms()
                if synonyms!= []:
                    wordnet_synonyms[word] = wordnet_synonyms[word] | set(chain.from_iterable([word.lemma_names() for word in synonyms]))
            except:
                pass
        except:
            pass
        wordnet_synonyms[word] = set(wordnet_synonyms[word])
    return wordnet_synonyms

def get_wordnet_set_pos(tokens):
    def get_wordnet_pos(treebank_tag):
        treebank_tag = treebank_tag[0][1]
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''
    
    wordnet_synonyms = {}
    print()
    for word in tokens:
        pos_n = get_wordnet_pos(pos_tag([word.text]))
        word = word.text
        synonyms_antonyms = []
        try:
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    if get_wordnet_pos(pos_tag([lemma.name()])) == pos_n:
                        synonyms_antonyms.append(lemma.name())    #add the synonyms
                    if lemma.antonyms() and get_wordnet_pos(pos_tag([lemma.antonyms()[0].name()])) == pos_n:    #When antonyms are available, add them into the list
                        synonyms_antonyms.append(lemma.antonyms()[0].name())
            wordnet_synonyms[word] = synonyms_antonyms#set(chain.from_iterable([word.lemma_names() for word in synonyms]))
            try:
                hyponyms = wordnet.synsets(word)[0].hyponyms()
                if hyponyms != []:
                    wordnet_synonyms[word] = wordnet_synonyms[word] | set(chain.from_iterable([word.lemma_names() if pos_tag([word.lemma_names()]) == pos_n else None for word in hyponyms]))
            except:
                pass
            try:
                hypernyms = wordnet.synsets(word)[0].hypernyms()
                if hypernyms!= []:
                    wordnet_synonyms[word] = wordnet_synonyms[word] | set(chain.from_iterable([word.lemma_names() if pos_tag([word.lemma_names()]) == pos_n else None  for word in hypernyms]))
            except:
                pass
        except:
            pass
        wordnet_synonyms[word] = set(wordnet_synonyms[word])
    return wordnet_synonyms