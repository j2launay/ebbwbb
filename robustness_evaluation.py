from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from find_decision_function import preparing_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from generate_dataset import generate_dataset
from storeExperimentalInformations import store_experimental_informations
import ape_text
from text_function_experiments import compute_robustness
from lime.lime_text import LimeTextExplainer

if __name__ == "__main__":
    dataset_names = ["polarity"] # religion, polarity, spam, fake, baseball
    models = [GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1)]
    # Fake concon (47) less (51) Polarity Concon (18) less (55) Spam Concon (4) less (50)
    models = [LogisticRegression(random_state=1)]
    # Fake less (54) Polarity Concon (25) less (44) Spam Concon (6) less (25)
    models = [MLPClassifier(random_state=1)]
    # Fake less (54) Polarity Concon (21) less (47) Spam Concon (16) less (31)
    models = [MultinomialNB()]   
    # Fake less (54) Polarity Concon (26) less (46) Spam Concon (21) less (33)
    models = [RandomForestClassifier(n_estimators=20, random_state=1)]
    # Fake less (55) Polarity Concon (21) less (47) Spam Concon (11) less (29)
    models = [VotingClassifier(estimators=[('lr', LogisticRegression()), ('nb', MultinomialNB())], voting="soft")]
    # Fake concon (52) less (48) Polarity Concon (24) less (49) Spam Concon (3) less (31)
    
    """models = [#RandomForestClassifier(n_estimators=20), LogisticRegression(),
                #MultinomialNB(),
                #GradientBoostingClassifier(n_estimators=20, learning_rate=1.0),
                tree.DecisionTreeClassifier(),
                RidgeClassifier(), 
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB())], voting="soft"),
                MLPClassifier(random_state=1), 
                LogisticRegression()]"""
    
    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 25
    nb_times_repetition = 5
    """ All the variable necessaries for generating the graph results """
    graph = True
    threshold_interpretability = 0.95
    models_name = []
                                                                                    
    text_methods=['wordnet_tagget']#['wordnet']#['concon']#, 'concon', 'lessconcon', 'augmenter', 'antonym', 'translation', 'cgt']
    corpus = True if 'corpus' in text_methods[0] else False

    for dataset_name in dataset_names:
        #if graph: experimental_informations = store_experimental_informations(columns_name)
        x, y, class_names = generate_dataset(dataset_name)
        for nb_model, model in enumerate(models):
            model_name = type(model).__name__
            models_name.append(model_name)
            #if graph: experimental_informations.initialize_per_models("./results/"+dataset_name+"/"+model_name+"/")# + linear_model_name)

            x_train, x_test, y_train, y_test, x_train_vectorize, x_test_vectorize, vectorizer = preparing_dataset(x, y, dataset_name, model)
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
            
            explainer = ape_text.ApeTextExplainer(x_train_vectorize, class_names, model_predict, model_predict_proba, vectorizer, verbose=True, extend_nlp=x_train_vectorize, corpus=corpus)
            
            for instance_to_explain in x_test_vectorize: 
                if cnt == max_instance_to_explain:
                    break
                print()
                print()
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("###", text_methods[0],"method to generate text.")
                print("### Models", model_name, " ", nb_model + 1, "over", len(models), "in", dataset_name)
                print("### robustness evaluation")

                instance_to_explain = instance_to_explain.encode('utf-8').decode("utf-8")
                print("instance to explain:", instance_to_explain)
                try:
                    #test += 2
                #except NameError:
                    new_robustness = compute_robustness(explainer, text_methods, nb_times_repetition, instance_to_explain, explainer.get_distance_text, "spacy")
                    try:
                        robustness += new_robustness
                    except NameError:
                        robustness = new_robustness
                        
                    pd_robustness = pd.DataFrame(robustness, columns=['target instance', 'set of counterfactuel find', 'average distance', 
                                    'all distances', 'distance metric', 'time needed', 'nb times cf not found', 'text method'])
                    cnt += 1
                except Exception as inst:
                    print(inst)
                    continue
                if graph: 
                    os.makedirs(os.path.dirname("./results/"+dataset_name+"/"+model_name+"/"), exist_ok=True)
                    pd_robustness.to_csv("./results/"+dataset_name+"/"+model_name+"/robustness_" + text_methods[0] + ".csv")
