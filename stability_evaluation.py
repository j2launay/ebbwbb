from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
from generate_dataset import generate_dataset, preparing_dataset
from text_function_experiments import TextExplainer

if __name__ == "__main__":
    dataset_names = ["polarity"] # polarity, spam, or fake
    models = [MLPClassifier(random_state=1)]
    models = [RandomForestClassifier(n_estimators=20, random_state=1)]
    
    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 100
    nb_times_repetition = 5
    """ All the variable necessaries for generating the graph results """
    graph = True
    models_name = []
                                                                                    
    text_methods=['growing_net']# growing_language, growing_net, random, sedc
    corpus = True if 'corpus' in text_methods[0] else False

    for dataset_name in dataset_names:
        x, y, class_names = generate_dataset(dataset_name)
        for nb_model, model in enumerate(models):
            model_name = type(model).__name__
            models_name.append(model_name)

            x_train, x_test, y_train, y_test, x_train_vectorize, x_test_vectorize, vectorizer = preparing_dataset(x, y, dataset_name)
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
            
            explainer = TextExplainer(class_names, model_predict, model_predict_proba, vectorizer, verbose=True, 
                        extend_nlp=x_train_vectorize, corpus=x_train_vectorize if corpus else False)
            
            for instance_to_explain in x_test_vectorize: 
                if cnt == max_instance_to_explain:
                    break
                print()
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("###", text_methods[0],"method to generate text.")
                print("### Models", model_name, " ", nb_model + 1, "over", len(models), "in", dataset_name)
                print("### stability evaluation")

                instance_to_explain = instance_to_explain.encode('utf-8').decode("utf-8")
                print("instance to explain:", instance_to_explain)
                try:
                    new_stability = explainer.compute_stability(text_methods, nb_times_repetition, instance_to_explain, explainer.get_similarity_text, "spacy")
                    try:
                        stability += new_stability
                    except NameError:
                        stability = new_stability
                        
                    pd_stability = pd.DataFrame(stability, columns=['target instance', 'set of counterfactuel find', 'average similarity', 
                                    'all similaritys', 'similarity metric', 'time needed', 'nb times cf not found', 'text method'])
                    cnt += 1
                except Exception as inst:
                    print(inst)
                    continue
                if graph: 
                    os.makedirs(os.path.dirname("./results/"+dataset_name+"/"+model_name+"/"), exist_ok=True)
                    pd_stability.to_csv("./results/"+dataset_name+"/"+model_name+"/stability_" + text_methods[0] + ".csv")
