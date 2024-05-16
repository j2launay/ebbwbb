import numpy as np
import pandas as pd
import spacy
import time
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import DistilBertForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from counterfactuals_methods import Counterfactuals
from generate_dataset import generate_dataset, preparing_dataset

if __name__ == "__main__":
    dataset_names = ["fake", "polarity", "spam"] # polarity, spam, fake, religion, baseball, ag_news
    text_methods=['growing_language', 'growing_net', 'sedc']
    bert_model = False # If set on True the experiments are conducted on DistillBert model
    vectorizer = CountVectorizer(min_df=1)
    #vectorizer = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english', sublinear_tf=True)

    if not bert_model:
        models = [MLPClassifier(hidden_layer_sizes= (100, 4), random_state=1),
                    MLPClassifier(hidden_layer_sizes= (64, 64), random_state=1),
                    MLPClassifier(hidden_layer_sizes= (128, 128), random_state=1),
                    MLPClassifier(hidden_layer_sizes= (256, 256), random_state=1),
                    RandomForestClassifier(n_estimators=500, random_state=1)]
        
    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 100
    columns_name = ["original text", "counterfactual", 'text_method', 'model_name', 'dataset_name'] 
    
    nlp = spacy.load("./model/en_core_web_lg")
    def remove_stopwords(text):
        """
        Remove stopwords in a text
        """
        doc = nlp(text.lower())
        result = []
        for token in doc:
            if token.text in nlp.Defaults.stop_words:
                continue
            if token.is_punct:
                continue
            result.append(token.lemma_)
        return " ".join(result)
    
    def get_similarity_text(text1, text2):
        """
        Compute a similarity between text1 and text2 based on a language model similarity
        """
        doc1=nlp(text1)
        doc2=nlp(text2)
        return doc1.similarity(doc2)
        
    for dataset_name in dataset_names:                                                                          
        x, y, class_names = generate_dataset(dataset_name)
        if bert_model:
            # The DistillBert models must have been finetuning in train_bert.py before
            if dataset_name == 'polarity':
                bert = DistilBertForSequenceClassification.from_pretrained("./model/bert/") 
            elif "fake" in dataset_name:
                bert = DistilBertForSequenceClassification.from_pretrained("./model/fake/bert/")
            elif "spam" in dataset_name:
                bert = DistilBertForSequenceClassification.from_pretrained("./model/spam/bert/")
            else:
                bert = DistilBertForSequenceClassification.from_pretrained("./model/ag_news/bert/")
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
            def tokenize_function(examples):
                return tokenizer(examples, padding="max_length", truncation=True)

            pipe = TextClassificationPipeline(model=bert, tokenizer=tokenizer, batch_size=100, return_all_scores=True)#batch_size=64
            models = [bert]

        for text_method in text_methods:
            for nb_model, model in enumerate(models):
                model_name = type(model).__name__

                x_train, x_test, y_train, y_test, x_train_vectorize, x_test_vectorize, vectorizer = preparing_dataset(x, y, dataset_name, vectorizer)
                print("###", model_name,"training on", dataset_name, "dataset.")
                print("###", text_method,"method to generate text.")
                
                if not bert_model:
                    model = model.fit(x_train, y_train)
                    print('### Accuracy:', sum(model.predict(x_test) == y_test)/len(y_test))
                
                cnt = 0
                if "Bert" not in model_name:
                    def model_predict(text):
                        if type(text) is str or type(text) is list:
                            text = vectorizer.transform(text)
                        return model.predict(text)
                    
                    def model_predict_proba(text):
                        return model.predict_proba(vectorizer.transform(text))
                else:
                    if dataset_name == "ag_news":
                        def model_predict(text):
                            predictions = pipe(text)
                            labels = []
                            for prediction in predictions:
                                if prediction[0]["score"] > 0.25:
                                    labels.append(0)
                                elif prediction[1]["score"] > 0.25:
                                    labels.append(1)
                                elif prediction[2]["score"] > 0.25:
                                    labels.append(2)
                                else:
                                    labels.append(3)
                            return labels if len(labels) > 1 else labels[0]
                        def model_predict_proba(text):
                            predictions = pipe(text)
                            temp_labels = []
                            for prediction in predictions:
                                temp_labels.append([prediction[0]['score'], prediction[1]['score'], prediction[2]['score'], prediction[3]['score']]) 
                            labels = np.array([np.array(xi) for xi in temp_labels])
                            return labels
                    else: 
                        def model_predict(text):
                            predictions = pipe(text)
                            labels = []
                            for prediction in predictions:
                                labels.append(0 if prediction[0]['score'] > 0.5 else 1) 
                            return labels if len(labels) > 1 else labels[0]
                        def model_predict_proba(text):
                            predictions = pipe(text)
                            temp_labels = []
                            for prediction in predictions:
                                temp_labels.append([prediction[0]['score'], prediction[1]['score']]) 
                            labels = np.array([np.array(xi) for xi in temp_labels])
                            return labels
                                    
                def generate_test_set(target_class):
                    """ 
                    Store texts classified as the target_class from the test set 
                    """
                    counterfactual_test_set_temp = np.array(x_test_vectorize[:300])[np.where([y != target_class for y in model_predict(x_test_vectorize[:300])])[0]].tolist()
                    counterfactual_test_set = []
                    for cf_test in counterfactual_test_set_temp:
                        counterfactual_test_set.append(remove_stopwords(cf_test))
                    return counterfactual_test_set
                
                cnt_not_find = 0
                generated_counterfactuals = []
                for instance_to_explain in x_test_vectorize[1:]: 
                    if cnt >= max_instance_to_explain:
                        break
                    print()
                    print()
                    print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                    print("###", text_method,"method to generate text.")
                    print("### Models ", model_name, "in", dataset_name)
                    
                    instance_to_explain = instance_to_explain.encode('utf-8').decode("utf-8")
                    target_class = model_predict([instance_to_explain])
                    print("instance to explain:", instance_to_explain)
                    try:
                        counterfactual_technique = Counterfactuals([instance_to_explain],
                                                                instance_to_explain,
                                                                model_predict,
                                                                method=text_method,
                                                                nlp=nlp)
                        start_time = time.time()
                        counterfactual, ennemies_, radius, iteration = counterfactual_technique.find_counterfactual()
                        runtime = time.time() - start_time
                        if text_method == "growing_net" or text_method == "growing_language":
                            # Convert the counterfactual generated into a string
                            counterfactual_numpy = counterfactual[0]
                            counterfactual = ''
                            for word in counterfactual_numpy:
                                counterfactual += str(word)
                        print()
                        print("counterfactual")
                        print(counterfactual)
                        counterfactual_test_set = generate_test_set(target_class)
                        closest_similarity = 1
                        for counterfactual_test in counterfactual_test_set:
                            # Loop over the input set of counterfactual texts from the dataset to find the closest one
                            temp_similarity = get_similarity_text(counterfactual_test, counterfactual)
                            if temp_similarity < closest_similarity:
                                closest_similarity = temp_similarity
                                closest_true_counterfactual = counterfactual_test
                        
                        generated_counterfactuals.append([instance_to_explain, counterfactual, remove_stopwords(counterfactual),
                                                        closest_true_counterfactual, type(vectorizer).__name__,
                                                        text_method, model_name, dataset_name, runtime, cnt_not_find])
                        pd_counterfactual = pd.DataFrame(generated_counterfactuals, columns=['target instance', 'closest counterfactual', 
                                                                                            'closest_counterfactual \stopword', 
                                                                                            'closest_true_instance', 'vectorizer', 'text_method', 
                                                                                            'model_name', 'dataset_name', 'runtime', 'not find'])
                        cnt += 1
                        
                        # Store results in a csv file
                        if "MLP" in model_name:
                            os.makedirs(os.path.dirname("./results/"+dataset_name+"/"+model_name+"_"+str(nb_model)+"/"), exist_ok=True)
                            filename_end = "_" + str(nb_model) + "/counterfactual_generated_" + text_method + "_" + type(vectorizer).__name__ + ".csv"
                        else:
                            os.makedirs(os.path.dirname("./results/"+dataset_name+"/"+model_name+"/"), exist_ok=True)
                            filename_end = "/counterfactual_generated_" + text_method + "_" + type(vectorizer).__name__ + ".csv"
                        pd_counterfactual.to_csv("./results/" + dataset_name + "/" + model_name + filename_end)

                    except Exception as inst:
                        print(inst)
                        cnt_not_find += 1
                        continue
                    
