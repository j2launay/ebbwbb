import spacy
import numpy as np
import spacy.cli
import en_core_web_lg
import en_core_web_md
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from yellowbrick.cluster import KElbowVisualizer
from anchors import anchor_text
from anchors.limes.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import CountVectorizer
from growingspheres.utils.gs_utils import Neighbors
import pyfolding as pf
#from textattack.augmentation import WordNetAugmenter, EmbeddingAugmenter, EasyDataAugmenter, CharSwapAugmenter
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from growingspheres.growingfields import GrowingFields, perturb_sentence_lime, perturb_sentence_multiple, perturb_sentence_wordnet
from text_function_experiments import compute_explanations_accuracy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time

class ApeTextExplainer(object):
    """
    Args:

    """
    def __init__(self, train_data, class_names, black_box_predict, black_box_predict_proba, vectorizer,
                    multiclass = False, threshold_precision=0.95, nb_min_instance_in_field=20, 
                    percentage_artificial_instance_in_field=0.1, verbose=False, extend_nlp=False, corpus=False):
        self.class_names, self.corpus = class_names, corpus
        if corpus:
            tf = TfidfVectorizer(input=train_data, analyzer='word', #ngram_range=(1,6),
                                    min_df = 0, stop_words = 'english', sublinear_tf=True)
            self.corpus =  [tf.fit_transform(train_data), tf]
        self.train_data, self.test_data, self.label_train, self.label_test = train_test_split(train_data, black_box_predict(train_data), test_size=0.33, random_state=42)
        self.black_box_predict = lambda x: black_box_predict(x)
        self.min_instance_per_class=nb_min_instance_in_field
        self.black_box_predict_proba = lambda x: black_box_predict_proba(x)
        self.multiclass = multiclass
        self.threshold_precision = threshold_precision

        """try:
            #nlp = spacy.load('./model/en_core_web_md')
            nlp = spacy.load('./model/en_core_web_lg')
        except:
            print("nlp model not found")
            from spacy.lookups import load_lookups
            nlp = spacy.load("en_core_web_md")#spacy.load('en_core_web_lg')
            print("len", len(nlp.vocab.strings))
            for test_text in self.text_data:
                doc = nlp(test_text)
            print("len after", len(nlp.vocab.strings))

            lookups = load_lookups("en", ["lexeme_prob"])
            nlp.vocab.lookups.add_table("lexeme_prob", lookups.get_table("lexeme_prob"))
            #nlp.to_disk("./model/en_core_web_lg")
            nlp.to_disk("./model/en_core_web_md")
        """
        from spacy.lookups import load_lookups
        nlp = spacy.load('en_core_web_lg')#spacy.load("en_core_web_md")
        for test_text in extend_nlp:
            doc = nlp(test_text)

        lookups = load_lookups("en", ["lexeme_prob"])
        nlp.vocab.lookups.add_table("lexeme_prob", lookups.get_table("lexeme_prob"))
        nlp.to_disk("./model/en_core_web_lg")
        
        #nlp = spacy.load('en_core_web_lg')
        self.anchor_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)
        self.lime_explainer = LimeTextExplainer(class_names=class_names)
        self.vectorizer = vectorizer
        self.nb_min_instance_in_field = nb_min_instance_in_field
        self.percentage_artificial_instance_in_field=percentage_artificial_instance_in_field
        self.verbose = verbose
        self.nlp = nlp
        # Si il y a un probleme avec le vectorizer c'est parce qu'il n'a pas les mots du jeu de test lorsqu'il fit
        #self.vectorizer.fit(train_data)

    def generate_instances_inside_field(self, radius, closest_counterfactual, min_instance_per_class, method='concon', iteration=1, anchor_generation=False):
        """ 
        Generates instances in the  area of the hyper field until minimum instances are found from each class 
        Args: radius: Size of the hyper field
              closest_counterfactual: Counterfactual instance center of the hyper field
              min_instance_per_class: Minimum number of instances from counterfactual class and target class present in the field
        Return: Set of instances from training data and artificial instances present in the field
                Labels of these instances present in the field
        """
        nb_different_outcome, nb_same_outcome, lime_iteration = 0, 0, 0
        word_list = closest_counterfactual.split()
        len_sentence = len(word_list) 
        friends_in_field, enemies_in_field, distances_in_field = [], [], []

        while (nb_different_outcome < min_instance_per_class or nb_same_outcome < min_instance_per_class): 
            # While there is not enough instances from each class
            nb_different_outcome, nb_same_outcome = 0, 0
            if self.verbose: 
                print("min instance per class", min_instance_per_class)
                print("nb instance generated", self.nb_min_instance_in_field)            
            if 'augmenter' in method:
                print("radius", radius)
                aug = naw.SynonymAug(aug_src='wordnet')
                texts_in_field = []
                for i in range(self.nb_min_instance_in_field):
                    texts_in_field.append(aug.augment(closest_counterfactual))
                #augmenter = CharSwapAugmenter(pct_words_to_swap=radius, transformations_per_example=(int) (self.nb_min_instance_in_field))
                #augmenter = WordNetAugmenter(pct_words_to_swap=radius, transformations_per_example=(int) (self.nb_min_instance_in_field))
                #augmenter = EmbeddingAugmenter(pct_words_to_swap=radius, transformations_per_example=(int) (self.nb_min_instance_in_field))
                print("using augmenter to generate sentences")
                #texts_in_field = augmenter.augment(closest_counterfactual)
            elif 'antonym' in method:
                print("radius", radius)
                aug = naw.AntonymAug()
                texts_in_field = []
                for i in range(self.nb_min_instance_in_field):
                    texts_in_field.append(aug.augment(closest_counterfactual))
                print("using the antonym augmenter method to generate sentences")
            elif 'translation' in method:
                print("radius", radius)
                back_translation_aug = naw.BackTranslationAug(
                    from_model_name='facebook/wmt19-en-de', 
                    to_model_name='facebook/wmt19-de-en'
                )                
                texts_in_field = []
                for i in range(self.nb_min_instance_in_field):
                    texts_in_field.append(back_translation_aug.augment(closest_counterfactual))
                print("using the translation augmenter method to generate sentences")
            elif 'gpt2' in method:
                print("radius", radius)
                aug = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2')
                texts_in_field = []
                for i in range(self.nb_min_instance_in_field):
                    texts_in_field.append(aug.augment(closest_counterfactual))
                print("using the gpt2 augmenter method to generate sentences")
            elif 'lessconcon' in method:
                if self.verbose: print("using lessconcon method to generate sentences")
                texts_in_field = perturb_sentence_multiple(closest_counterfactual, iteration, neighbors=Neighbors(self.nlp), 
                                                            max_sim = radius, n_iter=self.nb_min_instance_in_field)['Text'].values.tolist()
                distances_in_field = texts_in_field['Similarity'].values.tolist()
            elif 'wordnet' in method:
                if self.verbose: print("using wordnet to generate similar sentences")
                lime_iteration = max(iteration, lime_iteration + 1)
                texts_in_field = perturb_sentence_wordnet(closest_counterfactual, lime_iteration, growing_field=self.growing_fields, n_iter=self.nb_min_instance_in_field)
                distances_in_field = texts_in_field['Similarity'].values.tolist()
                texts_in_field = texts_in_field['Text'].values.tolist()
            elif 'lime' in method:
                if self.verbose: print("using lime to generate similar sentences")
                lime_iteration = max(iteration, lime_iteration + 1)
                texts_in_field = perturb_sentence_lime(closest_counterfactual, lime_iteration, self.growing_fields, n_iter=self.nb_min_instance_in_field)
                distances_in_field = [iteration] * len(texts_in_field)
            else:
                if self.verbose: 
                    print("using concon method to generate sentences")
                    print("iteration", iteration)
                texts_in_field = self.growing_fields.perturb_sentence_concon(self.nb_min_instance_in_field, iteration)

            labels_in_field = self.black_box_predict(texts_in_field)
            if anchor_generation:
                break
            for label_field, text_in_field in zip(labels_in_field, texts_in_field):
                if label_field != self.target_class:
                    enemies_in_field.append(text_in_field)
                    nb_different_outcome += 1
                else:
                    friends_in_field.append(text_in_field)
                    nb_same_outcome += 1

            nb_same_outcome, nb_different_outcome = len(friends_in_field), len(enemies_in_field)
            proportion_same_outcome, proportion_different_outcome = nb_same_outcome/min_instance_per_class, nb_different_outcome/min_instance_per_class
            if proportion_same_outcome < 1 or proportion_different_outcome < 1:
                # data generated inside field are not enough representative so we generate more.
                self.nb_min_instance_in_field = min(max(300, len_sentence), self.nb_min_instance_in_field + \
                                min(proportion_same_outcome, proportion_different_outcome) * min_instance_per_class + iteration * min_instance_per_class)
                friends_in_field = friends_in_field[:min(nb_same_outcome, nb_different_outcome)]
                enemies_in_field = enemies_in_field[:min(nb_same_outcome, nb_different_outcome)]
                if self.verbose:
                    print("min instances in field", self.nb_min_instance_in_field)
                    print("same outcome", nb_same_outcome)
                    print("different outcome", nb_different_outcome)
            iteration = min(iteration + 1, len_sentence)
                
        if self.verbose: 
            print('There are ', nb_different_outcome, " instances from a different class in the field over ", len(texts_in_field), " total instances in the dataset.")
            print("There are : ", nb_same_outcome, " instances classified as the target instance in the field.")
        return texts_in_field, labels_in_field, distances_in_field

    def store_texts_in_field_from_class(self, target_class, texts_in_field):
        """
        Store instances sampled in the field from the given target class
        """
        index_texts_in_field = np.where([y == target_class for y in self.black_box_predict(texts_in_field)])[0]
        texts_in_field_from_class = np.array(texts_in_field)[index_texts_in_field]
        return texts_in_field_from_class
        

    def libfolding_test(self, texts_in_field):
        def convert_to_vector(texts_in_field):
            texts_np_array = []
            for text_in_field in texts_in_field:
                nlp_text_in_field = self.nlp(str(text_in_field)).vector
                texts_np_array.append(nlp_text_in_field)
            texts_np_array = np.array(texts_np_array)
            return texts_np_array
        
        friends_in_field = self.store_texts_in_field_from_class(self.target_class, texts_in_field)
        friends_np_array = convert_to_vector(friends_in_field)
        friends_results = pf.FTU(friends_np_array, routine="python")

        counterfactuals_in_field = self.store_texts_in_field_from_class(1-self.target_class, texts_in_field)
        counterfactuals_np_array = convert_to_vector(counterfactuals_in_field)
        counterfactual_results = pf.FTU(counterfactuals_np_array, routine="python")


        self.multimodal_friends_results = friends_results.folding_statistics<1
        self.friends_folding_statistics = friends_results.folding_statistics
        self.friends_pvalue = friends_results.p_value

        self.multimodal_counterfactual_results = counterfactual_results.folding_statistics<1
        self.counterfactual_folding_statistics = counterfactual_results.folding_statistics
        self.counterfactual_pvalue = counterfactual_results.p_value
        
        self.separability_index = -1
        self.multimodal_results = (self.multimodal_counterfactual_results or self.multimodal_friends_results)


        #vectors_in_field = np.array([self.nlp(x).vector for x in texts_in_field])
        #results = pf.FTU(vectors_in_field, routine="python")
        #self.multimodal_results = results.folding_statistics>1
        print("The unimodal test indicate that data are ", "multimodal." if self.multimodal_results else "unimodal.")
        """if self.multimodal_results:
            visualizer = KElbowVisualizer(KMeans(), k=(2,8))
            x_elbow = np.array(texts_in_field)
            print("type test")
            print(type(texts_in_field))
            visualizer.fit(x_elbow)
            n_clusters = visualizer.elbow_value_
            if self.verbose: print("n CLUSTERS ", n_clusters)
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(texts_in_field)
            self.counterfactuals = kmeans.cluster_centers_
            if self.verbose: print("Mean center of clusters from KMEANS ", self.counterfactuals)"""
        return self.multimodal_results

    def explain_instance(self, instance, method='concon', experiment=False):
        start_time = time.time()
        instance_array = [instance]
        instance_vectorized = self.vectorizer.transform(instance_array)
        self.target_class = self.black_box_predict(instance_array)[0]
        
        self.growing_fields = GrowingFields(instance_vectorized, instance, self.black_box_predict, method=method, nlp=self.nlp, \
                        corpus=self.corpus, verbose=self.verbose)
        print("time to initialise gf", time.time()-start_time)
        start_time = time.time()
        closest_counterfactual, ennemies_, threshold_distance, iteration = self.growing_fields.find_counterfactual()
        print("closest counterfactual: ", closest_counterfactual)
        print("time to find cf", time.time()-start_time)
        start_time = time.time()
        #closest_counterfactual_vectorized = self.vectorizer.transform([closest_counterfactual])
        #closest_counterfactual_str = ' '.join(self.vectorizer.inverse_transform(closest_counterfactual)[0])
        #print("closest counterfactual", closest_counterfactual_str)

        # We compute the sparsity distance because we are using the concon method
        distance_sentences = []
        if method=='concon':
            print("I am using concon method")
            texts_in_field, labels_in_field, _ = self.generate_instances_inside_field(self.growing_fields.radius, 
                                                    closest_counterfactual, self.min_instance_per_class)
            distance_sentences = np.ones(len(labels_in_field))
        elif 'augmenter' in method:
            print("I am using the augmenter method")
            texts_in_field, labels_in_field, _ = self.generate_instances_inside_field(0.5,#self.growing_fields.radius, 
                                                        closest_counterfactual, self.min_instance_per_class,
                                                        method=method)
            for sentence in texts_in_field:
                distance_sentences.append(self.get_distance_text(str(closest_counterfactual), sentence))
        elif 'antonym' in method:
            print("I am using the augmenter method with antonym")
            texts_in_field, labels_in_field, _ = self.generate_instances_inside_field(0.5,#self.growing_fields.radius, 
                                                        closest_counterfactual, self.min_instance_per_class,
                                                        method=method)
            for sentence in texts_in_field:
                distance_sentences.append(self.get_distance_text(str(closest_counterfactual), sentence))
        elif 'tranlation' in method:
            print("I am using the augmenter method with translation")
            texts_in_field, labels_in_field, _ = self.generate_instances_inside_field(0.5,#self.growing_fields.radius, 
                                                        closest_counterfactual, self.min_instance_per_class,
                                                        method=method)
            for sentence in texts_in_field:
                distance_sentences.append(self.get_distance_text(str(closest_counterfactual), sentence))
        elif 'gpt2' in method:
            print("I am using the augmenter method with gpt2")
            texts_in_field, labels_in_field, _ = self.generate_instances_inside_field(0.5,#self.growing_fields.radius, 
                                                        closest_counterfactual, self.min_instance_per_class,
                                                        method=method)
            for sentence in texts_in_field:
                distance_sentences.append(self.get_distance_text(str(closest_counterfactual), sentence))
        elif 'wordnet' in method:
            print("I am using wordnet method")
            texts_in_field, labels_in_field, distance_sentences = self.generate_instances_inside_field(self.growing_fields.radius,
                                                    closest_counterfactual, self.min_instance_per_class, method=method)
            
        else:
            print("I am using lessconcon method")
            #largest_distance_from_cf = self.get_distance_text(str(closest_counterfactual), instance, self.nlp)
            #smallest_distance_from_cf = self.get_distance_text(str(self.growing_fields.onevsrest[0]), instance, self.nlp)
            texts_in_field, labels_in_field, distance_sentences = self.generate_instances_inside_field(self.growing_fields.radius, 
                                                    str(closest_counterfactual), self.min_instance_per_class, method=method)
            #for sentence in texts_in_field:
            #    distance_sentences.append(self.get_distance_text(str(closest_counterfactual), sentence))
        print("time to generate instances in the spheres", time.time()-start_time)
        start_time = time.time()
        instances_in_field_vectorized = self.vectorizer.transform(texts_in_field)
        self.libfolding_test(texts_in_field)
        print("time to conduct libfolding test", time.time()-start_time)
        start_time = time.time()

        def predict_lr(texts):
            # Predicts the class of a text
            if len(texts) > 1:
                return self.black_box_predict(self.vectorizer.transform(texts))
            return int(self.black_box_predict(self.vectorizer.transform(texts))[0])
        print("libfolding test done")
        print("les données sont : ", "multimodal" if self.multimodal_results else "unimodal")
        print()

        print("searching for anchors explanation")
        anchor_exp = self.anchor_explainer.explain_instance(instance, predict_lr, threshold=self.threshold_precision, use_proba=True)
        print("anchors explanation find, now let's go for Lime !")
        #self.lime_exp = self.lime_explainer.explain_instance(closest_counterfactual_str, self.black_box_predict_proba, num_features=6)
        print("closest counterfactual", closest_counterfactual)

        linear_exp = self.lime_explainer.explain_instance(closest_counterfactual, self.black_box_predict_proba,
                        sentences_in_field=instances_in_field_vectorized,
                        labels_in_field=labels_in_field,
                        distance_sentences=distance_sentences,
                        num_features=len(closest_counterfactual))
        if self.verbose:
            print()
            print("TODO: Deal avec les mots qui ne sont pas vectorizé par le vectorizer (ou pas retrouvé dans l'explication de explanation.py ou lime_text.py dans limes)")
            print('Lime explanation for class %s' % self.class_names[self.target_class])
            print('\n'.join(map(str, linear_exp.as_list(vectorizer=self.vectorizer))))
            print('Anchor: %s' % (' AND '.join(anchor_exp.names())))

            print("predict observation: ",self.black_box_predict(instance_array)[0])
            #print("class of the closest counterfactual: ", self.black_box_predict([closest_counterfactual_str])[0])
            print("class of the closest counterfactual: ", self.black_box_predict([closest_counterfactual])[0])

        if experiment:
            return compute_explanations_accuracy(self, closest_counterfactual, instance, texts_in_field, linear_exp, anchor_exp, method, self.growing_fields.radius)

    def get_distance_text(self, text1, text2):
        doc1=self.nlp(text1)
        doc2=self.nlp(text2)
        return doc1.similarity(doc2)
