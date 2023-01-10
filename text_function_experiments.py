from similarity_functions import l0_similarity_text, cosine_similarity_text, pairwise_similarity_text
import time
from counterfactuals_methods import Counterfactuals
import spacy
import spacy.cli
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lookups import load_lookups

class TextExplainer(object):

    def __init__(self, class_names, black_box_predict, black_box_predict_proba, vectorizer,
                    verbose=False, extend_nlp=False, corpus=False):
        """
        Args:   class_names: array of the name of the classes predicted by the classifier
                black_box_predict: classifier function to predict, given an input text document and return an int
                black_box_predict_proba: same as above but return a probability
                vectorizer: the vectorizer used by the classifier to convert an input text document
                verbose: if True print details of the search for explanation
                extend_nlp: input dataset use to extend the spacy language model to more words than the one use originally
                corpus: input dataset use to get a corpus in which the random method will pick words to replace in the original text
        """
        self.class_names = class_names
        self.black_box_predict = lambda x: black_box_predict(x)
        self.black_box_predict_proba = lambda x: black_box_predict_proba(x)
        self.vectorizer = vectorizer
        self.verbose = verbose
        if corpus != False:
            tf = TfidfVectorizer(input=corpus, analyzer='word',
                                    min_df = 0, stop_words = 'english', sublinear_tf=True)
            self.corpus =  [tf.fit_transform(corpus), tf]
        else:
            self.corpus = False
        try:
            self.nlp = spacy.load("./model/en_core_web_lg")
        except:
            nlp = spacy.load('en_core_web_lg')
            # Extends the size of the spacy vocabulary with the words from the dataset
            for test_text in extend_nlp:
                doc = nlp(test_text)
            lookups = load_lookups("en", ["lexeme_prob"])
            nlp.vocab.lookups.add_table("lexeme_prob", lookups.get_table("lexeme_prob"))
            nlp.to_disk("./model/en_core_web_lg")
            self.nlp = nlp
        # If there is a problem with the vectorizer, you need to train it on the training dataset like following
        #self.vectorizer.fit(train_data)

    def get_similarity_text(self, text1, text2):
        """
        Compute a similarity between text1 and text2 based on a language model similarity
        """
        doc1=self.nlp(text1)
        doc2=self.nlp(text2)
        return doc1.similarity(doc2)

    def compute_k_closest(self, text_methods, similarity_metrics_name, instance, remove_stopwords, counterfactual_test_set):
        """
        Compute the similarity to the k closest instances from the counterfactual_test_set
        Input:  text_methods: counterfactual method use to generate artificial sentences
                similarity_metrics_name: similarity use to measure the similarity between two texts
                instance: target instance to explain
                remove_stopwords: function to remove stopwords in a text
                counterfactual_test_set: set of text classified differently than the target instance to explain
        Output: Mean similarity of the counterfactual to the k closest texts, time to measure it and raw value of
                counterfactual, target, and closest text in the input set
        """
        to_return = []
        for text_method in text_methods:
            print(text_method + " used to measure the k closest.")
            instance_array = [instance]
            instance_vectorized = self.vectorizer.transform(instance_array)
            self.target_class = self.black_box_predict(instance_array)[0]
            
            self.counterfactuals = Counterfactuals(instance_vectorized, instance, self.black_box_predict, method=text_method, nlp=self.nlp, corpus=self.corpus, verbose=True)
            print("initialized of the counterfactual")
            closest_counterfactual_with_stopword, _, _, _ = self.counterfactuals.find_counterfactual()
            print("closest counterfactual", closest_counterfactual_with_stopword)

            closest_counterfactual = remove_stopwords(closest_counterfactual_with_stopword)
            for similarity_metric_name in similarity_metrics_name:
                time_needed = time.time()
                print(similarity_metric_name)
                if "spacy" in similarity_metric_name:
                    similarity_metric = self.get_similarity_text 
                elif "l0" in similarity_metric_name:
                    similarity_metric = l0_similarity_text
                elif "pairwise" in similarity_metric_name:
                    similarity_metric = pairwise_similarity_text
                elif "cosine" in similarity_metric_name:
                    similarity_metric = cosine_similarity_text
                else:
                    raise NameError("the name of the similarity metric is not good:", similarity_metrics_name)

                similarity_cf_target = similarity_metric(closest_counterfactual, remove_stopwords(instance))
                closest_similarity, farthest_similarity = 1, -1
                for counterfactual_test in counterfactual_test_set:
                    # Loop over the input set of counterfactual texts from the dataset to find the closest one
                    temp_similarity = similarity_metric(counterfactual_test, closest_counterfactual)
                    if temp_similarity < closest_similarity:
                        closest_similarity = temp_similarity
                        closest_true_counterfactual = counterfactual_test
                    elif temp_similarity >= farthest_similarity:
                        farthest_similarity = temp_similarity
                        farthest_true_counterfactual = counterfactual_test
                similarity_metric_result = [instance, closest_counterfactual_with_stopword, closest_counterfactual, closest_true_counterfactual, 
                                        farthest_true_counterfactual, closest_similarity, farthest_similarity, similarity_cf_target,
                                        similarity_metric_name, text_method, time.time() - time_needed]
                to_return.append(similarity_metric_result)
        return to_return
                
    def compute_stability(self, text_methods, nb_time_repetition, instance, similarity, similarity_name):
        """
        Function to measure the stability of the input text_methods for a given input text instance
        Args:   text_methods: an array of string designing the methods use to generate artificial documents 
                nb_time_repetition: number of times the function tries to find a counterfactual with a given counterfactual method
                instance: target text document
                similarity: the similarity function use to measure similarity between two texts
                similarity_name: name of the similarity function use to measure similarity between two texts
        Return: 
        """
        to_return = []
        for text_method in text_methods:
            temp_to_return = [instance]
            print(text_method)
            instance_array = [instance]
            
            instance_vectorized = self.vectorizer.transform(instance_array)
            self.target_class = self.black_box_predict(instance_array)[0]
            set_of_counterfactual_finded = []
            start_time = time.time()
            nb_not_found, i = 0, 0
            while i < nb_time_repetition:
                print("try number", i + 1, "over", nb_time_repetition)
                if nb_not_found > i:
                    raise Exception(text_method, "was not able to find a counterfactual at least 2 times over", nb_time_repetition)
                self.counterfactuals = Counterfactuals(instance_vectorized, instance, self.black_box_predict, method=text_method, verbose=False, \
                    nlp=self.nlp, corpus=self.corpus)
                try:
                    closest_counterfactual, _, _, _ = self.counterfactuals.find_counterfactual()
                    set_of_counterfactual_finded.append(closest_counterfactual)
                except:
                    # Count number of time the method is not able to find the closest counterfactual
                    print("nb not found", nb_not_found)
                    nb_not_found += 1
                    continue
                i += 1

            average_similarity, nb_counterfactual, similaritys = 0, 0, []
            # Average the similarity between all counterfactuals generated by the explainer method
            for i, first in enumerate(set_of_counterfactual_finded):
                for j, second in enumerate(set_of_counterfactual_finded):
                    if i != j:
                        temp_similarity = similarity(str(first), str(second))
                        similaritys.append(temp_similarity)
                        average_similarity += temp_similarity
                        nb_counterfactual += 1
            temp_to_return += [set_of_counterfactual_finded, average_similarity/nb_counterfactual, similaritys, 
                    similarity_name, time.time() - start_time, nb_not_found, text_method]
            to_return.append(temp_to_return)
        return to_return


def remove_stopwords(text, nlp):
    doc = nlp(text.lower())
    result = []
    for token in doc:
        if token.text in nlp.Defaults.stop_words:
            continue
        if token.is_punct:
            continue
        result.append(token.lemma_)
    return " ".join(result)