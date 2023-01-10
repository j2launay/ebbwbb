from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
import spacy
from generate_dataset import generate_dataset, preparing_dataset
from text_function_experiments import TextExplainer

if __name__ == "__main__":
    dataset_names = ["fake"]# fake, polarity, or spam
    models = [MLPClassifier(random_state=1)]
    models = [RandomForestClassifier(n_estimators=20, random_state=1)]
    
    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 100
    """ All the variable necessaries for generating the graph results """
    graph = True
    models_name = []
                                                                                    
    text_methods=['growing_language']#['growing_net, growing_language, sedc, random
    metrics_similarity = ["l0", "pairwise", "cosine", "spacy"]
    for dataset_name in dataset_names:
        x, y, class_names = generate_dataset(dataset_name)
        for nb_model, model in enumerate(models):
            model_name = type(model).__name__
            models_name.append(model_name)

            x_train, x_test, y_train, y_test, x_train_vectorize, x_test_vectorize, vectorizer = preparing_dataset(x, y, dataset_name)
            pd.DataFrame(x_test_vectorize).to_csv('./dataset/' + dataset_name + "_test.csv")
            print("###", model_name,"training on", dataset_name, "dataset.")
            
            model = model.fit(x_train, y_train)

            print('### Accuracy:', sum(model.predict(x_test) == y_test)/len(y_test))
            cnt = 0

            def model_predict(text):
                if type(text) is str or type(text) is list:
                    text = vectorizer.transform(text)
                return model.predict(text)
            def model_predict_proba(text):
                return model.predict_proba(vectorizer.transform(text))

            nlp = spacy.load('en_core_web_lg')

            def remove_stopwords(text):
                doc = nlp(text.lower())
                result = []
                for token in doc:
                    if token.text in nlp.Defaults.stop_words:
                        continue
                    if token.is_punct:
                        continue
                    result.append(token.lemma_)
                return " ".join(result)
             
            def generate_test_set(target_class):
                counterfactual_test_set_temp = np.array(x_test_vectorize[:300])[np.where([y != target_class for y in model_predict(x_test_vectorize[:300])])[0]].tolist()
                counterfactual_test_set = []
                for cf_test in counterfactual_test_set_temp:
                    counterfactual_test_set.append(remove_stopwords(cf_test))
                return counterfactual_test_set

            explainer = TextExplainer(class_names, model_predict, model_predict_proba, vectorizer, verbose=True, extend_nlp=x_train_vectorize)
            counterfactual_test_set_0, counterfactual_test_set_1 = generate_test_set(0), generate_test_set(1)

            for instance_to_explain in x_test_vectorize: 
                if cnt == max_instance_to_explain:
                    break
                print()
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("###", text_methods, "method to generate text.")
                print("### Models", model_name, " ", nb_model + 1, "over", len(models), "in", dataset_name)
                
                instance_to_explain = instance_to_explain.encode('utf-8').decode("utf-8")
                print("instance to explain:", instance_to_explain)
                print("### k closest evaluation")
                target_class = model_predict([instance_to_explain])[0]

                counterfactual_test_set = counterfactual_test_set_0 if target_class == 0 else counterfactual_test_set_1
                try:
                    new_similaritys = explainer.compute_k_closest(text_methods, metrics_similarity, instance_to_explain, remove_stopwords, counterfactual_test_set)
                    try:
                        similaritys += new_similaritys
                    except NameError:
                        similaritys = new_similaritys
                        
                    pd_similarity = pd.DataFrame(similaritys, columns=['target instance', 'closest counterfactual', 'closest_counterfactual \stopword', 
                                    'closest_true_counterfactual', 'farthest_true_counterfactual', 'closest_similarity', 'farthest_similarity', 'similarity_cf_target',
                                    'similarity_metric', 'text_method', 'time'])
                    cnt += 1
                except Exception as inst:
                    print(inst)
                    continue
                if graph: 
                    os.makedirs(os.path.dirname("./results/"+dataset_name+"/"+model_name+"/"), exist_ok=True)
                    filename_end = "/k_closest_" + text_methods[0] + ".csv"
                    pd_similarity.to_csv("./results/" + dataset_name + "/" + model_name + filename_end)
