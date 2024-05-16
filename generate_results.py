import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot
import torch
import pylev
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer, util

sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def sentence_bert_distance(a, b):
    """Metric to measure distance between two tests a and b with a sentence Transformer"""
    embedding_a = sentence_transformer.encode(a, convert_to_tensor=True)
    embedding_b = sentence_transformer.encode(b, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding_a, embedding_b)#compute/find the highest similarity scores
    index_highest_similarity = torch.argmax(cosine_scores)
    if isinstance(b, list):
        return cosine_scores[0][index_highest_similarity].item()
    else:
        return cosine_scores[index_highest_similarity].item()

def levenshtein(a, b):
    """Metric to measure distance between two tests a and b with the Levenshtein distance"""
    a = a.split(" ")
    b = b.split(" ")
    return pylev.levenshtein(a,b)


def store_file(metric):
    """
    Group results about {metric} for every 
    explanation technique, dataset, black box, vectorizers 
    into a single file name {metric}.csv
    """
    pd_data = pd.DataFrame()
    
    explainers = [('sedc', 'SEDC'), ('growing_net', 'Growing Net'),
                    ('growing_language', 'Growing Language'), ('polyjuice','Polyjuice'), 
                    ('xspells','xSPELLS'), ('counterfactualgan', 'CounterfactualGAN')]
    datasets = ['fake', 'spam', 'polarity']
    black_boxes = [('dnn', 'Neural Network'), ('rf', 'Random Forest'), ('dnn_64', 'Neural Network 64'), 
                   ('dnn_128', 'Neural Network 128'), ('dnn_256', 'Neural Network 256'), ('bert', 'BERT')]
    vectorizers = ['', '_TfidfVectorizer']

    for dataset in datasets:
        for model_name in black_boxes:
            for vectorizer_name in vectorizers:
                for method in explainers:
                    try:
                        data = pd.read_csv("./results/" + method[0] + "/" + dataset + "_" + model_name[0] + "_" + 
                                       method[0] + vectorizer_name + ".csv")
                        try:
                            temp_data = pd.DataFrame({metric: data[metric],
                                                  "dataset": [dataset] * data['target instance'].shape[0],
                                                  "model": [model_name[1]]*data['target instance'].shape[0],
                                                  "vectorizer": [vectorizer_name]*data['target instance'].shape[0],
                                                  "method": [method[1]]*data['target instance'].shape[0]})
                        except KeyError:
                            print("./results/" + method[0] + "/" + dataset + "_" + model_name[0] + "_" + 
                                       method[0] + vectorizer_name + ".csv")
                            print(data.columns)
                            temp_data = pd.DataFrame({metric: data[metric],
                                                  "dataset": [dataset] * data['target instance'].shape[0],
                                                  "model": [model_name[1]]*data['target instance'].shape[0],
                                                  "vectorizer": [vectorizer_name]*data['target instance'].shape[0],
                                                  "method": [method[1]]*data['target instance'].shape[0]})
                        if pd_data.empty:
                            pd_data = temp_data
                        else:
                            pd_data = pd.concat([pd_data, temp_data])
                        pd_data.to_csv("./results/" + metric + ".csv")
                    except FileNotFoundError:
                        print("file not found", "./results/" + method[0] + "/" + dataset + "_" + model_name[0] + "_" + 
                                       method[0] + vectorizer_name + ".csv")
    

def pyplot_display_information(xlabel, ylabel, filename, ylim=True):
    """Function to set all the parameters to plot the boxplot"""
    if ylim: pyplot.ylim([0, 1.05])
    
    pyplot.legend(fontsize=25, loc=(1.04, 0))
    pyplot.xlabel(xlabel, fontsize=25)
    pyplot.ylabel(ylabel, fontsize=25)
    pyplot.xticks(size = 20)
    pyplot.yticks(size = 20)
    pyplot.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    pyplot.show(block=False)
    pyplot.pause(0.1)
    pyplot.close('all')

def generate_boxplot(metric, metric_name, ylabel, ylim=True, vectorizer=False):
    """
    Generate a boxplot with the results about {metric}, with the name display as {metric_name}"""
    sns.set(style="white")
    pyplot.figure(figsize=(28,8))
    pyplot.figure(figsize=(20,8))
    pd_data = pd.read_csv("./results/" + metric + ".csv")
    pd_data.drop(pd_data[pd_data['model'] == "Neural Network 64"].index, inplace = True)
    pd_data.drop(pd_data[pd_data['model'] == "Neural Network 128"].index, inplace = True)
    pd_data.drop(pd_data[pd_data['model'] == "Neural Network 256"].index, inplace = True)
    pd_data.drop(pd_data[pd_data['model'] == "dnn_64"].index, inplace = True)
    pd_data.drop(pd_data[pd_data['model'] == "dnn_128"].index, inplace = True)
    pd_data.drop(pd_data[pd_data['model'] == "dnn_256"].index, inplace = True)

    pd_data['model'].mask(pd_data['model'] == 'dnn', "Neural Network", inplace=True)
    #pd_data['model'].mask(pd_data['model'] == 'Neural Network', "Réseau Neur.", inplace=True)
    pd_data['model'].mask(pd_data['model'] == "rf", 'Random Forest', inplace=True)
    #pd_data['model'].mask(pd_data['model'] == "Random Forest", 'Forêt Aléat.', inplace=True)
    pd_data['model'].mask(pd_data['model'] == "bert", "DistillBERT", inplace=True)

    pd_data['method'].mask(pd_data['method'] == 'sedc', "SEDC", inplace=True)
    pd_data['method'].mask(pd_data['method'] == 'growing_net', "Growing Net", inplace=True)
    pd_data['method'].mask(pd_data['method'] == 'growing_language', "Growing Language", inplace=True)
    pd_data['method'].mask(pd_data['method'] == 'xspells', "xSPELLS", inplace=True)
    pd_data['method'].mask(pd_data['method'] == 'counterfactualgan', "cfGAN", inplace=True)
    pd_data['method'].mask(pd_data['method'] == 'polyjuice', "Polyjuice", inplace=True)
    #pd_data['method'].mask(pd_data['method'] == 'CounterfactualGAN', "counterfactualGAN", inplace=True)
    
    if vectorizer:
        pd_data.reset_index(inplace=True)
        pd_data.drop(['index', 'Unnamed: 0'], axis=1, inplace=True)
        for index, row, in pd_data.iterrows():
            if row['vectorizer'] == "_TfidfVectorizer":
                pd_data.at[index, "model"] = pd_data.iloc[index]["model"] + " Tf-Idf"
                print(pd_data.iloc[index]["model"])
            elif "BERT" not in row['model']:
                pd_data.at[index, "model"] = pd_data.iloc[index]["model"] + " Count"
                print(pd_data.iloc[index]["model"])
    else:
        pd_data.drop(pd_data[pd_data['vectorizer'] == "_TfidfVectorizer"].index, inplace=True)
        pd_data.reset_index(inplace=True)
        pd_data.drop(['index', 'Unnamed: 0'], axis=1, inplace=True)
    
    print(pd_data)
    if metric_name == "perplexity":
        pd_data["perplexity"] = 1 - pd_data["perplexity"]/pd_data["perplexity"].max()
    if metric_name == "distance counterfactual target instance bert":
        pd_data["distance counterfactual target instance bert"] = pd_data["distance counterfactual target instance bert"].max() - pd_data["distance counterfactual target instance bert"]
        indexes = []
        for index, row in pd_data.iterrows(): 
            if row["distance counterfactual target instance bert"] > 1:
                indexes.append(index)
        pd_data.drop(indexes, inplace=True)
    sns.boxplot(x="model", y=metric_name, hue="method", data=pd_data, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="Method", ylabel=ylabel, 
                               filename="./results/" +  metric_name + ".png", 
                               ylim=ylim)

def measure_distance_cf_target_lev_bert():
    """
    Measure the Levenshtein and BERT distance between the target instance and
    the closest counterfactual generated by each explanation technique on each 
    black box, vectorizer, and dataset.
    Store the aggregated results in two csv files, each corresponding to the metric employed. 
    """
    datasets = ['spam', 'polarity', 'fake']
    models = ['bert', 'rf', 'dnn', 'dnn_64', 'dnn_128', 'dnn_256']
    vectorizers = ['', '_TfidfVectorizer']
    methods = ['sedc', 'growing_net', 'growing_language', 'polyjuice', 'xspells', 'counterfactualgan']
    
    for dataset in datasets:
        for model_name in models:
            for vectorizer_name in vectorizers:
                for method in methods:
                    try:
                        data = pd.read_csv("./results/" + method + "/" + dataset + "_" + model_name + "_" + 
                                       method + vectorizer_name + ".csv")
                        distance_cf_target_lev, distance_cf_target_bert = [], []
                        for index, row in data.iterrows():
                            target = row["target instance"]
                            counterfactual = row['closest counterfactual']
                            try:
                                distance_cf_target_lev.append(levenshtein(target, counterfactual))
                                distance_cf_target_bert.append(sentence_bert_distance(target, counterfactual))
                            except AttributeError:
                                distance_cf_target_lev.append(None)
                                distance_cf_target_bert.append(None)
                                
                        data['distance counterfactual target instance lev'] = distance_cf_target_lev
                        data['distance counterfactual target instance bert'] = distance_cf_target_bert
                        data.reset_index(drop=True, inplace=True)
                        try:
                            data.drop(data.filter(regex="Unname"),axis=1, inplace=True)
                        except KeyError:
                            print()
                        data.to_csv("./results/" + method + "/" + dataset + "_" + model_name + "_" + 
                                       method + vectorizer_name + ".csv")
                    except FileNotFoundError:
                        print("file not found", "./results/" + method + "/" + dataset + "_" + model_name + "_" + 
                                       method + vectorizer_name + ".csv")

def measure_perplexity():
    """
    Measure the perplexity of the counterfactual generated by each explanation technique on
    each black box, classifier and dataset by measuring the mean squared error loss of a Large GPT2 model
    This function generates a perplexity.csv file with the perplexity of each counterfactual
    """
    model_id = "gpt2-large"
    model_base = GPT2LMHeadModel.from_pretrained(model_id)#.to(device)
    tokenizer_base = GPT2TokenizerFast.from_pretrained(model_id)
    pd_perplexity = pd.DataFrame()

    datasets = ['spam', 'polarity', 'fake']
    models = ['rf', 'dnn', 'dnn_64', 'dnn_128', 'dnn_256', 'bert']
    vectorizers = ['', '_TfidfVectorizer']
    methods = ['sedc', 'growing_net', 'growing_language', 'polyjuice', 'xspells', 'counterfactualgan']
    pd_all_distance = pd.DataFrame()
    for method in methods:
        for dataset in datasets:
            for model_name in models:
                for vectorizer_name in vectorizers:
                    try:
                        data = pd.read_csv("./results/" + method + "/" + dataset + "_" + model_name + "_" + 
                                       method + vectorizer_name + ".csv")               
                        perplexity = []
                        for index, row in data.iterrows():
                            perturb_text = row['closest counterfactual']
                            print()
                            print('perturb', perturb_text)
                            perturb_text_vectorized = tokenizer_base(perturb_text, return_tensors="pt")
                            loss = model_base(input_ids=perturb_text_vectorized["input_ids"], labels=perturb_text_vectorized["input_ids"]).loss
                            print("perplexity on perturb:", loss)
                            perplexity.append(float(loss))
                            if pd_all_distance.empty:
                                pd_all_distance = pd.DataFrame({"perplexity": [float(loss)],
                                            "dataset name": dataset, "model": model_name, "method": method,
                                            "vectorizer": vectorizer_name})
                            else:
                                pd_all_distance = pd.concat([pd_all_distance, pd.DataFrame({"perplexity": [float(loss)],
                                            "dataset name": [dataset], "model": [model_name], "method": [method],
                                            "vectorizer": vectorizer_name})])
                            if pd_perplexity.empty:
                                pd_perplexity = pd.DataFrame({"perplexity": [float(loss)],
                                            "dataset name": dataset, "model": model_name, "method": method,
                                            "vectorizer": vectorizer_name})
                            else:
                                pd_perplexity = pd.concat([pd_perplexity, pd.DataFrame({"perplexity": [float(loss)],
                                            "dataset name": [dataset], "model": [model_name], "method": [method],
                                            "vectorizer": vectorizer_name})])
                            pd_perplexity.to_csv("./results/perplexity/" + method + ".csv")
                            pd_all_distance.to_csv("./results/perplexity.csv")
                            print(pd_all_distance)
                    except FileNotFoundError:
                        print("file not found")  

def output_runtime():
    datasets = ['fake', 'spam', 'polarity']
    models = ['Neural Network', 'Random Forest', 'BERT']#, 'dnn_64', 'dnn_128', 'dnn_256']
    vectorizers = ['_TfidfVectorizer']#, '']
    methods = ['SEDC', 'Growing Net', 'Growing Language', 'Polyjuice', 'CounterfactualGAN', 'xSPELLS']
    data = pd.read_csv("./results/runtime.csv")
    for dataset in datasets:
        for model_name in models:
            for method in methods:
                for vectorizer_name in vectorizers:
                    temp_data = data.loc[(data['model'] == model_name) & (data['dataset'] == dataset) & (data['method'] == method)]
                    print()
                    print(dataset, model_name, method, temp_data['runtime'].mean(), temp_data['runtime'].std())       

def output_recall():
    datasets = ['fake', 'spam', 'polarity']
    models = ['dnn', 'rf', 'bert']#, 'dnn_64', 'dnn_128', 'dnn_256']
    vectorizers = ['_TfidfVectorizer']#, '']
    methods = ['sedc', 'growing_net', 'growing_language', 'polyjuice', 'xspells', 'counterfactualgan']
    
    for method in methods:
        for dataset in datasets:
            for model_name in models:
                for vectorizer_name in vectorizers:
                    if "bert" in model_name:
                        vectorizer_name = ""
                    try:
                        data = pd.read_csv("./results/" + method + "/" + dataset + "_" + model_name + "_" + 
                                       method + vectorizer_name + ".csv")
                        print()
                        print(method, model_name, vectorizer_name, dataset, "nb not find:",
                              data['not find'].iloc[-1], "over", 
                              data['not find'].iloc[-1] + data['closest counterfactual'].shape[0], 
                              "equal", round(1 - data['not find'].iloc[-1] / 
                              (data['not find'].iloc[-1] + data['closest counterfactual'].shape[0]), 2))
                        
                    except FileNotFoundError:
                        print("file not found", "./results/" + method + "/" + dataset + "_" + model_name + "_" + 
                                       method + vectorizer_name + ".csv")

if __name__ == "__main__":
    #store_file("distance counterfactual target instance lev")
    #store_file("distance counterfactual target instance bert")
    #store_file("runtime")
    #measure_perplexity()
    #measure_distance_cf_target_lev_bert()
    generate_boxplot("distance counterfactual target instance bert", "distance counterfactual target instance bert", "Minimality BERT", vectorizer=True)
    generate_boxplot("distance counterfactual target instance lev", "distance counterfactual target instance lev", "Minimality Levenshtein", False, vectorizer=True)
    #generate_boxplot("perplexity", "perplexity", "perplexity")
    #output_runtime()
    #output_recall()