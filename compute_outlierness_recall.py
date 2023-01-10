import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import spacy
from generate_dataset import generate_dataset, preparing_dataset
from counterfactuals_methods import Counterfactuals
from text_function_experiments import TextExplainer, remove_stopwords
import csv
import matplotlib.pyplot as pyplot
import seaborn as sns

def pyplot_display_information(xlabel, ylabel, filename, ylim=True):
    if ylim: pyplot.ylim([0, 1.1])
    pyplot.legend(fontsize=12)
    pyplot.xlabel(xlabel, fontsize=18)
    pyplot.ylabel(ylabel, fontsize=18)
    pyplot.xticks(size = 15)
    pyplot.yticks(size = 15)
    pyplot.savefig(filename,bbox_inches='tight', pad_inches=0)
    pyplot.show(block=False)
    pyplot.pause(0.1)
    pyplot.close('all')

def split_per_class(x_test, x_test_vectorize, model_predict):
    """
    Function to split and store element from both classes to speed up the outlierness
    measurement. Texts are split and processed for spacy language model.
    Args:   x_test: array of raw test texts
            x_test_vectorize: array of test texts vectorized 
            model_predict: function of the classifier that takes as input a text and return the class
    Output: raw texts of class 0 and 1, texts preprocessed for spacy of class 0 and 1.
    """
    print("vectorize", x_test_vectorize[:10])
    print("raw text", x_test[:10])
    x_test_class0, x_test_class1 = [], []
    nlp_class0, nlp_class1 = [], []
    for x, x_vectorize in zip(x_test, x_test_vectorize):
        x_stopword = remove_stopwords(x_vectorize, nlp)
        if model_predict(x)[0] == 1:    
            x_test_class1.append(x_stopword)
            nlp_class1.append(nlp(x_stopword)) 
        else: 
            x_test_class0.append(x_stopword)
            nlp_class0.append(nlp(x_stopword))
    print("Number of texts in class 0", len(x_test_class0))
    print("Number of texts in class 1", len(x_test_class1))
    return x_test_class0, x_test_class1, nlp_class0, nlp_class1

def get_similarity_text(text1, text2):
    doc1=nlp(text1)
    doc2=nlp(text2)
    return doc1.similarity(doc2)

def compute_outlierness_and_recall(dataset_name, method, model, only_recall=True):
    """
    Function to measure the outlierness and recall of each counterfactual method
    Args:   dataset_name: string of the dataset name
            method: string of the counterfactual method
            model: classifier scikit learn model
            only_recall: boolean to compute only the recall instead of both outlierness and recall
    """
    def to_csv_dataframe(results, recall):
        pd_results = pd.DataFrame(results)
        model_name = "RF" if "Random" in type(model).__name__ else "DNN"
        if recall:
            pd_results.to_csv("./results/recall/" + method + "_" + dataset_name + "_" + model_name + ".csv", index=False)
        else:
            pd_results.to_csv("./results/outlierness/"  + method + "_" + dataset_name + "_" + model_name + ".csv", index=False)

    pd_results = pd.DataFrame()
    model_name = "RF" if "Random" in type(model).__name__ else "DNN"
    if method == "xspells":
        results = pd.read_csv("./results/xspells/" + dataset_name + "_" + model_name + ".csv")
    else:
        results = pd.read_csv("./results/counterfactual_gan/cf_gan_" + dataset_name + ".csv")
    try:
        results.drop(["Unnamed: 0"], axis=1, inplace=True)
    except:
        print()

    if method == "xspells" or method == "cfgan":
        # if xspells or cfgan no need to compute again the counterfactual
        targets = results['original text']
        cfs = results['counter exemplars']
    else:
        targets = x_train_vectorize
        cfs = x_train_vectorize

    x, y, class_names = generate_dataset(dataset_name)
    x_train, x_test, y_train, _, x_train_vectorize, x_test_vectorize, vectorizer = preparing_dataset(x, y, dataset_name)

    model = model.fit(x_train, y_train)
    def model_predict(text):
        if type(text) is str or type(text) is list:
            text = vectorizer.transform(text)
        return model.predict(text)
    def model_predict_proba(text):
        return model.predict_proba(vectorizer.transform(text))

    if "sedc" in method or "random" in method:
        explainer = TextExplainer(class_names, model_predict, model_predict_proba, 
                        vectorizer, verbose=False, extend_nlp=x_train_vectorize, corpus=x_train_vectorize)

    x_test_class_0, x_test_class_1, nlp_class0, nlp_class1 = split_per_class(x_test, x_test_vectorize, model_predict)
    nb_instance = 1
    recall = pd.DataFrame()
    for target, closest_counterfactual in zip(targets, cfs):
        if nb_instance >= 100:
            break
        if "xspells" not in method and "cfgan" not in method:#then search for the closest counterfactual
            instance_array = [target]
            instance_vectorized = vectorizer.transform(instance_array)
                
            if "sedc" in method or "random" in method:
                growing_fields = Counterfactuals(instance_vectorized, target, model_predict, method=method, nlp=explainer.nlp, corpus=explainer.corpus)
            else:
                growing_fields = Counterfactuals(instance_vectorized, target, model_predict, method=method, nlp=nlp)
            print()
            print("initialized", method)
            nb_instance += 1
            try:
                #test += 2
            #except NameError:
                closest_counterfactual, _, _, _ = growing_fields.find_counterfactual()
                print("closest counterfactual", closest_counterfactual)
                print("find by", method)

                recall = recall.append({"target": target, "counterfactual": closest_counterfactual, "method": method, "black_box": model_name}, ignore_index=True)
                to_csv_dataframe(recall, True)
            except Exception as e:
                # if no counterfactual found, then replace the closest counterfactual by None to measure the recall
                recall = recall.append({"target": target, "counterfactual": None, "method": method, "black_box": model_name}, ignore_index=True)
                to_csv_dataframe(recall, True)
                print(e)
                continue
        if not only_recall: # then compute the outlierness
            if "xspells" in method:# split the array of counterfactuals to take only the first one
                try:
                    closest_counterfactual = closest_counterfactual.split("',")
                    closest_counterfactual = closest_counterfactual[0][2:]
                except AttributeError:
                    print("attribute error", closest_counterfactual)

            try:
                target_class = model_predict(vectorizer.transform([closest_counterfactual]))[0]
            except:
                continue
            print("nb", nb_instance, "over", len(targets))
            nb_instance += 1

            # store the texts of the counterfactual class from the input dataset to measure outlierness
            if target_class == 0:
                test_loop = x_test_class_0
                test_loop_nlp = nlp_class0 
            else: 
                test_loop = x_test_class_1
                test_loop_nlp = nlp_class1
            closest_counterfactual_nlp = nlp(str(closest_counterfactual))
            largest_similarity = 0
            # Measure the outlierness (highest similarity to the counterfactual text from the dataset)
            for text, nlp_text in zip(test_loop, test_loop_nlp):
                if nlp_text.similarity(closest_counterfactual_nlp) > largest_similarity:
                    largest_similarity = nlp_text.similarity(closest_counterfactual_nlp)
                    closest_true_text = text
                
            pd_results = pd_results.append({method + ' similarity spacy': largest_similarity, 'target text': target, 
                    'closest counterfactual':closest_counterfactual, 'closest true text': closest_true_text}, ignore_index=True)
            to_csv_dataframe(pd_results, False)

def get_counterfactual_gan_data(dataset_name):
    """Function to store the 100 first counterfactuals generated by cfGAN in a file"""
    original_data = pd.read_csv("./results/counterfactual_gan/counterfactual_gan_" + dataset_name + ".tsv", sep='\t', names=['text', 'label'], header=None)
    prediction = pd.read_json( "./results/counterfactual_gan/counterfactual_gan_" + dataset_name + "_cf.json")
    prediction = prediction['counterfactuals'][0][:100]# Get only the 100 counterfactuals generated by cfGAN
    counterfactual_gan = pd.DataFrame({'original text':original_data['text'], 'counter exemplars':prediction})
    counterfactual_gan.to_csv("./results/counterfactual_gan/cf_gan_" + dataset_name + ".csv")

def transform_to_tsv(filename):
    df = pd.read_csv('./dataset/counterfactual_gan/' + filename + '.csv')
    df=df.reindex(columns=['message', 'label'])
    df.to_csv('./dataset/new' + filename + '.csv', index=False)
    with open('./dataset/counterfactual_gan/new' + filename + '.csv','r') as csvin, open('./dataset/counterfactual_gan/' + filename + '.tsv', 'w') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout, delimiter='\t')
        for i, row in enumerate(csvin):
            if i > 1:
                tsvout.writerow(row)

def generate_outlierness_boxplot():
    sns.set(style="darkgrid")
    pd_results = pd.DataFrame()
    for dataset in ["spam", "fake", "polarity"]:
        for black_box in [('dnn', 'Neural Network'), ('rf', 'Random Forest')]:
            for method in [('sedc', 'sedc'), ('growing_net', 'Grow. Net'), ('growing_language', 'Grow. Lang.'), ('cfgan', 'cfGAN'), ('xspells', 'xSPELLS')]:
                pd_result = pd.read_csv("./results/outlierness/" + method[0] + "_" + dataset + "_" + black_box[0] + ".csv")
                pd_result.rename(columns={pd_result.columns[0]: "similarity"}, inplace=True)
                pd_result["method"] = method[1]
                pd_result["black box"] = black_box[1]
                pd_results = pd.concat([pd_results, pd_result])
            print(pd_results)
            sns.boxplot(x="method", y="similarity", data=pd_results, palette="Set1", width=0.6)
            pyplot_display_information(xlabel="method", ylabel="similarity", filename="./results/outlierness/outlierness_" + dataset + "_" + black_box[0] + ".png")
        sns.boxplot(x="black box", y="similarity", data=pd_results, palette="Set1", width=0.6, hue="method")
        pyplot_display_information(xlabel="black box", ylabel="similarity", filename="./results/outlierness/outlierness_" + dataset + ".png")
    sns.boxplot(x="black box", y="similarity", data=pd_results, palette="Set1", width=0.6, hue="method")
    pyplot_display_information(xlabel="black box", ylabel="similarity", filename="./results/outlierness/outlierness.png")

def generate_minimality_boxplot():
    sns.set(style="darkgrid")
    pd_results = pd.DataFrame()
    for dataset in ["spam", "fake", "polarity"]:
        for black_box in [('dnn', 'Neural Network'), ('rf', 'Random Forest')]:
            for transparent_method in [('sedc', 'sedc'), ('growing_net', 'Grow. Net'), ('growing_language', 'Grow. Lang.')]:#, ('random', 'random')]:
                for opaque_method in [('cf_gan', 'cfGAN'), ('xspells', 'xSPELLS')]:
                    pd_result = pd.read_csv("./results/similarity/" + dataset + "_" + opaque_method[0] +  "_" + transparent_method[0] + "_similarity.csv")
                    pd_result = pd_result.loc[pd_result['metric'] == "spacy"]
                    print(pd_result)
                    transparent_result = pd_result['wb similarity']
                    opaque_result = pd_result['bb similarity']
                    pd_black_box = pd_result['bb']
                    pd_black_box = pd_black_box.replace(to_replace=("DNN", 'RF'), value=(('Neural Network', "Random Forest")))
                    pd_transparent = pd.DataFrame({"similarity": transparent_result, "method":transparent_method[1], "black box": pd_black_box})
                    pd_opaque = pd.DataFrame({"similarity": opaque_result, "method":opaque_method[1], "black box": pd_black_box})
                    pd_result = pd.concat([pd_transparent, pd_opaque])
                    pd_results = pd.concat([pd_results, pd_result])
            print(pd_results)
            sns.boxplot(x="method", y="similarity", data=pd_results, palette="Set1", width=0.6)
            pyplot_display_information(xlabel="method", ylabel="similarity", filename="./results/minimality/minimality_" + dataset + "_" + black_box[0] + ".png")
        sns.boxplot(x="black box", y="similarity", data=pd_results, palette="Set1", width=0.6, hue="method")
        pyplot_display_information(xlabel="black box", ylabel="similarity", filename="./results/minimality/minimality_" + dataset + ".png")
    sns.boxplot(x="black box", y="similarity", data=pd_results, palette="Set1", width=0.6, hue="method", hue_order=['sedc', 'Grow. Net', "Grow. Lang.", "cfGAN", "xSPELLS"])
    pyplot_display_information(xlabel="black box", ylabel="similarity", filename="./results/minimality/minimality.png")

def generate_recall_boxplot():
    # Measure the recall of xspells and counterfactualGAN directly by comparing the number of counterfactuals found when computing them
    pd_results = pd.DataFrame()
    for method in [('sedc', 'sedc'), ('growing_net', 'Growing Net'), ('growing_language', 'Growing Language')]:#, ('xspells', 'xSPELLS')]:
        for dataset in ["spam", "fake", "polarity"]:
            for black_box in ['DNN', 'RF']:
                pd_result = pd.read_csv("./results/recall/" + method[0] + "_" + dataset + "_" + black_box + ".csv")
                counterfactuals = pd_result['counterfactual']
                print("RECALL", counterfactuals.count()/len(counterfactuals))
                print("with", method)
                print("on", black_box, dataset)
                pd_result_temp = pd.DataFrame({'recall':[counterfactuals.count()/len(counterfactuals)], 'method':[method[1]], \
                    "black box": [black_box], "dataset": [dataset]})
                pd_results = pd.concat([pd_results, pd_result_temp])
            print(pd_results)
            sns.boxplot(x="method", y="recall", data=pd_results, palette="Set1", width=0.6)
            pyplot_display_information(xlabel="method", ylabel="recall", filename="./results/recall/recall_" + dataset + "_" + black_box[0] + ".png")
        sns.boxplot(x="black box", y="recall", data=pd_results, palette="Set1", width=0.6, hue="method")
        pyplot_display_information(xlabel="black box", ylabel="recall", filename="./results/recall/recall_" + dataset + ".png")
        print(pd_results.groupby("method", as_index=False)["recall"].mean())
    sns.boxplot(x="black box", y="recall", data=pd_results, palette="Set1", width=0.6, hue="method")
    pyplot_display_information(xlabel="black box", ylabel="recall", filename="./results/recall/recall.png")

if __name__ == "__main__":
    compute = True
    generate_outlierness_array = True
    generate_recall_array = True
    generate_minimality = True

    datasets_name = ['polarity']#, 'spam', 'fake', 'polarity']#
    methods = ['random']#'sedc', 'growing_language']#'growing_net']#'xspells']#'cfgan']#
    black_boxes_explanation = ['cf_gan', 'xspells']
    models = [MLPClassifier(), RandomForestClassifier()]#
    
    if generate_outlierness_array:
        compute = False
        generate_outlierness_boxplot()
    if generate_recall_array:
        compute = False
        generate_recall_boxplot()
    if generate_minimality:
        compute = False
        generate_minimality_boxplot()
    if compute:
        nlp = spacy.load('en_core_web_lg')
        for dataset_name in datasets_name:
            # these commented lines are not needed since the results are already stored in "cf_gan_" + dataset_name + ".csv"
            """transform_to_tsv(dataset_name)
            get_counterfactual_gan_data(dataset_name)"""
            for method in methods:
                for model in models:
                    compute_outlierness_and_recall(dataset_name, method, model)
