import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pyplot

def store_all_data():
    print("store all data")
    metrics = ["accuracy", "k_closest", "robustness"]
    black_boxes = ['GradientBoostingClassifier', 'LogisticRegression', 'MLPClassifier', 'MultinomialNB', 'RandomForestClassifier', 'VotingClassifier']
    datasets = ['spam', "fake", "polarity"]
    explanation_methods = ['concon', 'concon_corpus', 'lessconcon', '', 'concon_temp', 'concon_corpus_temp', 'lessconcon_temp', '_temp']
    for metric in metrics:
        all_data = []
        for dataset in datasets:
            for black_box in black_boxes:
                for explanation_method in explanation_methods:
                    try:
                        print('./results/' + dataset + '/' + black_box + "/" + metric + "_" + explanation_method + ".csv")
                        data = pd.read_csv('./results/' + dataset + '/' + black_box + "/" + metric + "_" + explanation_method + ".csv")
                    except FileNotFoundError:
                        continue
                    all_data.append(data)
        pd_final = pd.concat(all_data)
        pd_final.reset_index(drop=True, inplace=True)
        try:
            pd_final.drop(['Unnamed: 0'], axis=1, inplace=True)
        except KeyError:
            print()
        pd_final.to_csv("./results/pd_final_" + metric + ".csv")

def generate_accuracy_boxplot_linear():
    print("generate accuracy boxplot linear")
    sns.set(style="darkgrid")
    pyplot.figure(figsize=(22,10))
    pd_final = pd.read_csv("./results/pd_final_accuracy.csv")
    pd_final.drop(['Unnamed: 0'], axis=1, inplace=True)
    lse_accuracy, ls_accuracy, lime_accuracy = pd_final['lse_accuracy'].to_frame(), pd_final['ls_accuracy'].to_frame(), pd_final['lime_accuracy'].to_frame()
    lse_accuracy['text_method'], ls_accuracy['text_method'], lime_accuracy['text_method'] = pd_final['text_method'], pd_final['text_method'], pd_final['text_method']
    lse_accuracy.rename(columns={"lse_accuracy": "accuracy"}, inplace=True)
    ls_accuracy.rename(columns={"ls_accuracy": "accuracy"}, inplace=True)
    lime_accuracy.rename(columns={"lime_accuracy": "accuracy"}, inplace=True)
    
    lse_accuracy['method'], ls_accuracy['method'], lime_accuracy['method'] = 'LSe', 'LS', 'LIME'
    new_pd_final = pd.concat([lse_accuracy, ls_accuracy, lime_accuracy])
    sns.boxplot(x="text_method", y="accuracy", hue="method", data=new_pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="method", ylabel="accuracy", filename="./results/accuracy_lse_ls_lime.png")

def generate_accuracy_boxplot_lse_ls_lime_anchor_dt():
    print("generate accuracy boxplot lse ls lime anchor dt")
    sns.set(style="darkgrid")
    pyplot.figure(figsize=(22,10))
    pd_final = pd.read_csv("./results/pd_final_accuracy.csv")
    pd_final.drop(['Unnamed: 0'], axis=1, inplace=True)
    lse_accuracy, ls_accuracy, lime_accuracy, anchor_accuracy, dt_accuracy = pd_final['lse_accuracy'].to_frame(), \
        pd_final['ls_accuracy'].to_frame(), pd_final['lime_accuracy'].to_frame(), pd_final['anchor_accuracy'].to_frame(), pd_final['decision_tree_accuracy'].to_frame()
    lse_accuracy['text_method'], ls_accuracy['text_method'], lime_accuracy['text_method'], anchor_accuracy['text_method'], dt_accuracy['text_method'] = \
        pd_final['text_method'], pd_final['text_method'], pd_final['text_method'], pd_final['text_method'], pd_final['text_method']
    lse_accuracy.rename(columns={"lse_accuracy": "accuracy"}, inplace=True)
    ls_accuracy.rename(columns={"ls_accuracy": "accuracy"}, inplace=True)
    lime_accuracy.rename(columns={"lime_accuracy": "accuracy"}, inplace=True)
    anchor_accuracy.rename(columns={"anchor_accuracy": "accuracy"}, inplace=True)
    dt_accuracy.rename(columns={"decision_tree_accuracy": "accuracy"}, inplace=True)
    
    lse_accuracy['method'], ls_accuracy['method'], lime_accuracy['method'], anchor_accuracy['method'], dt_accuracy['method'] = 'LSe', 'LS', 'LIME', 'Anchors', 'DT'
    new_pd_final = pd.concat([lse_accuracy, ls_accuracy, lime_accuracy, anchor_accuracy, dt_accuracy])
    sns.boxplot(x="text_method", y="accuracy", hue="method", data=new_pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="method", ylabel="accuracy", filename="./results/accuracy_lse_ls_lime_anchor_dt.png")

def generate_accuracy_boxplot_ape():
    print("generate accuracy boxplot ape")
    sns.set(style="darkgrid")
    pyplot.figure(figsize=(22,10))
    pd_final = pd.read_csv("./results/pd_final_accuracy.csv")
    pd_final.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    cf, fr, fr_pvalue, cf_pvalue = pd_final['explainer.counterfactual_folding_statistics'], pd_final['explainer.friends_folding_statistics'], \
        pd_final['explainer.friends_pvalue'], pd_final['explainer.counterfactual_pvalue']
    multimodal_index = list(
            set(np.where(cf.to_numpy() >= 1)[0].tolist()) |\
            set(np.where(fr_pvalue.to_numpy() > 0.5)[0].tolist()) |\
            set(np.where(cf_pvalue.to_numpy() > 0.5)[0].tolist()) |\
            set(np.where(fr.to_numpy() >= 1)[0].tolist()))
    unimodal_index = list(set(range(0, pd_final.shape[0])) - set(multimodal_index))

    unimodal_result, multimodal_result = pd_final.loc[unimodal_index], pd_final.loc[multimodal_index]

    lse_accuracy, ls_accuracy, lime_accuracy, anchor_accuracy, dt_accuracy, lse_unimodal_accuracy, lse_multimodal_accuracy, \
        anchor_multimodal_accuracy, dt_multimodal_accuracy = pd_final['lse_accuracy'].to_frame(), \
            pd_final['ls_accuracy'].to_frame(), pd_final['lime_accuracy'].to_frame(), pd_final['anchor_accuracy'].to_frame(), \
                pd_final['decision_tree_accuracy'].to_frame(), unimodal_result['lse_accuracy'].to_frame(), multimodal_result['lse_accuracy'].to_frame(),\
                    multimodal_result['anchor_accuracy'].to_frame(), multimodal_result['decision_tree_accuracy'].to_frame()
    
    lse_accuracy['text_method'], ls_accuracy['text_method'], lime_accuracy['text_method'], anchor_accuracy['text_method'], dt_accuracy['text_method'] = \
        pd_final['text_method'], pd_final['text_method'], pd_final['text_method'], pd_final['text_method'], pd_final['text_method']
    lse_unimodal_accuracy['text_method'] = unimodal_result['text_method']
    lse_multimodal_accuracy['text_method'], anchor_multimodal_accuracy['text_method'], dt_multimodal_accuracy['text_method'] =\
        multimodal_result['text_method'], multimodal_result['text_method'], multimodal_result['text_method']
    
    lse_accuracy.rename(columns={"lse_accuracy": "accuracy"}, inplace=True)
    ls_accuracy.rename(columns={"ls_accuracy": "accuracy"}, inplace=True)
    lime_accuracy.rename(columns={"lime_accuracy": "accuracy"}, inplace=True)
    anchor_accuracy.rename(columns={"anchor_accuracy": "accuracy"}, inplace=True)
    dt_accuracy.rename(columns={"decision_tree_accuracy": "accuracy"}, inplace=True)
    lse_unimodal_accuracy.rename(columns={"lse_accuracy": "accuracy"}, inplace=True)
    lse_multimodal_accuracy.rename(columns={"lse_accuracy": "accuracy"}, inplace=True) 
    anchor_multimodal_accuracy.rename(columns={"anchor_accuracy": "accuracy"}, inplace=True) 
    dt_multimodal_accuracy.rename(columns={"decision_tree_accuracy": "accuracy"}, inplace=True)
    
    ape_a = pd.concat([lse_unimodal_accuracy, anchor_multimodal_accuracy])
    ape_t = pd.concat([lse_unimodal_accuracy, dt_multimodal_accuracy])

    lse_accuracy['method'], ls_accuracy['method'], lime_accuracy['method'], anchor_accuracy['method'], dt_accuracy['method'], ape_a['method'], \
        ape_t['method'], lse_unimodal_accuracy['method'], lse_multimodal_accuracy['method'] = 'LSe', 'LS', 'LIME', 'Anchors', 'DT', 'APEa', 'APEt', 'LSe Uni', 'LSe Mul'
    
    new_pd_final = pd.concat([lse_accuracy, ls_accuracy, lime_accuracy, ape_a, ape_t])
    sns.boxplot(x="text_method", y="accuracy", hue="method", data=new_pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="method", ylabel="accuracy", filename="./results/accuracy_lse_ls_lime_anchor_dt.png")
    pyplot.figure(figsize=(22,10))
    pd_uni_multi = pd.concat([lse_unimodal_accuracy, lse_multimodal_accuracy])
    sns.boxplot(x="text_method", y="accuracy", hue="method", data=pd_uni_multi, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="method", ylabel="accuracy", filename="./results/accuracy_lse_uni_mul.png")

def generate_accuracy_boxplot_uni_mul():
    print("generate accuracy boxplot uni vs mul")
    sns.set(style="darkgrid")
    pyplot.figure(figsize=(22,10))
    pd_final = pd.read_csv("./results/pd_final_accuracy.csv")
    pd_final.drop(['Unnamed: 0'], axis=1, inplace=True)
    cf, fr, fr_pvalue, cf_pvalue = pd_final['explainer.counterfactual_folding_statistics'], pd_final['explainer.friends_folding_statistics'], \
        pd_final['explainer.friends_pvalue'], pd_final['explainer.counterfactual_pvalue']
    multimodal_index = list(
            set(np.where(cf.to_numpy() >= 1)[0].tolist()) |\
            set(np.where(fr_pvalue.to_numpy() > 0.5)[0].tolist()) |\
            set(np.where(cf_pvalue.to_numpy() > 0.5)[0].tolist()) |\
            set(np.where(fr.to_numpy() >= 1)[0].tolist()))
    unimodal_index = list(set(range(0, pd_final.shape[0])) - set(multimodal_index))

    unimodal_result, multimodal_result = pd_final.loc[unimodal_index], pd_final.loc[multimodal_index]
    unimodal_result['uni'], multimodal_result['uni'] = 'Mul', 'Uni'
    new_pd_final = pd.concat([unimodal_result, multimodal_result])
    sns.boxplot(x="text_method", y="lse_balanced_accuracy_score", hue="uni", data=new_pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="method", ylabel="accuracy", filename="./results/accuracy_uni_mul.png")

def generate_accuracy_boxplot_close_far():
    print("generate accuracy boxplot close far")
    sns.set(style="darkgrid")
    pyplot.figure(figsize=(22,10))
    pd_final = pd.read_csv("./results/pd_final_accuracy.csv")
    pd_final.drop(['Unnamed: 0'], axis=1, inplace=True)
    radius = pd_final['radius']
    close_index = np.where(radius.to_numpy() <= 0.2)[0].tolist()
    far_index = list(set(range(0, pd_final.shape[0])) - set(close_index))

    close_result, far_result = pd_final.loc[close_index], pd_final.loc[far_index]
    close_result['close'] = 'Close'
    far_result['close'] = 'Far'
    new_pd_final = pd.concat([close_result, far_result])
    sns.boxplot(x="text_method", y="lse_balanced_accuracy_score", hue="close", data=new_pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="method", ylabel="accuracy", filename="./results/accuracy_close_far.png")

def generate_accuracy_length_plot():
    print("generate accuracy length plot")
    sns.set(style="darkgrid")
    pyplot.figure(figsize=(22,10))
    pd_final = pd.read_csv("./results/pd_final_accuracy.csv")
    pd_final.drop(['Unnamed: 0'], axis=1, inplace=True)
    cf, fr, fr_pvalue, cf_pvalue = pd_final['explainer.counterfactual_folding_statistics'], pd_final['explainer.friends_folding_statistics'], \
        pd_final['explainer.friends_pvalue'], pd_final['explainer.counterfactual_pvalue']
    multimodal_index = list(
            set(np.where(cf.to_numpy() >= 1)[0].tolist()) |\
            set(np.where(fr_pvalue.to_numpy() > 0.5)[0].tolist()) |\
            set(np.where(cf_pvalue.to_numpy() > 0.5)[0].tolist()) |\
            set(np.where(fr.to_numpy() >= 1)[0].tolist()))
    unimodal_index = list(set(range(0, pd_final.shape[0])) - set(multimodal_index))

    unimodal_result, multimodal_result = pd_final.loc[unimodal_index], pd_final.loc[multimodal_index]
    unimodal_result['uni'], multimodal_result['uni'] = 'Uni', 'Mul'
    new_pd_final = pd.concat([unimodal_result, multimodal_result])

    sns.boxplot(x="text_method", y="size_of_linear_explanation", hue='uni', data=new_pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="text method", ylabel="size of linear explanation", filename="./results/size_linear_explanation.png", ylim=False)
    pyplot.figure(figsize=(22,10))
    sns.boxplot(x="text_method", y="size_of_anchor_explanation", hue='uni', data=new_pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="text method", ylabel="size of anchor explanation", filename="./results/size_anchor_explanation.png", ylim=False)

    pyplot.figure(figsize=(22,10))
    pyplot.scatter(unimodal_result['size_of_linear_explanation'], unimodal_result['lse_roc_auc_score'], linewidth=2.0, label="uni")
    pyplot.scatter(multimodal_result['size_of_linear_explanation'], multimodal_result['lse_roc_auc_score'], linewidth=2.0, label="mul")
    pyplot.ylim([0, 1.1])
    pyplot.xlabel("size of linear explanation", fontsize=25)
    pyplot.ylabel("lse accuracy", fontsize=25)
    pyplot.legend(fontsize=25)
    pyplot.savefig("./results/plot.png",bbox_inches='tight', pad_inches=0)
    pyplot.show(block=False)
    pyplot.pause(0.1)
    pyplot.close('all')
    
def generate_k_closest_boxplot():
    print("generate k closest boxplot")
    sns.set(style="darkgrid")
    pyplot.figure(figsize=(22,10))
    pd_final = pd.read_csv("./results/pd_final_k_closest.csv")
    pd_final.drop(['Unnamed: 0'], axis=1, inplace=True)
    sns.boxplot(x="text_method", y="time", hue="distance_metric", data=pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="method", ylabel="time", filename="./results/k_closest_time.png", ylim=False)
    pyplot.figure(figsize=(22,10))
    sns.boxplot(x="text_method", y="farthest_distance", hue="distance_metric", data=pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="method", ylabel="distance", filename="./results/k_closest_distance.png")
    
def generate_robustness_boxplot():
    print("generate robustness boxplot")
    sns.set(style="darkgrid")
    pyplot.figure(figsize=(22,10))
    pd_final = pd.read_csv("./results/pd_final_robustness.csv")
    pd_final.drop(['Unnamed: 0'], axis=1, inplace=True)
    sns.boxplot(x="text method", y="time needed", data=pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="method", ylabel="time", filename="./results/robustness_time.png", ylim=False)
    pyplot.figure(figsize=(22,10))
    sns.boxplot(x="text method", y="average distance", data=pd_final, palette="Set1", width=0.6)
    pyplot_display_information(xlabel="method", ylabel="distance", filename="./results/robustness_distance.png")

def pyplot_display_information(xlabel, ylabel, filename, ylim=True):
    if ylim: pyplot.ylim([0, 1.1])
    pyplot.legend(fontsize=12)
    pyplot.xlabel(xlabel, fontsize=25)
    pyplot.ylabel(ylabel, fontsize=25)
    pyplot.xticks(size = 25)
    pyplot.yticks(size = 25)
    pyplot.savefig(filename,bbox_inches='tight', pad_inches=0)
    pyplot.show(block=False)
    pyplot.pause(0.1)
    pyplot.close('all')

if __name__ == "__main__":
    store_all_data()
    generate_robustness_boxplot()
    generate_k_closest_boxplot()
    generate_accuracy_boxplot_uni_mul()
    """
    generate_accuracy_boxplot_ape()
    generate_accuracy_boxplot_linear()
    generate_accuracy_boxplot_uni_mul()
    generate_accuracy_boxplot_close_far()
    generate_accuracy_length_plot()"""
    """store_accuracy()
    accuracy_score_boxplot()
    #k_closest_boxplot()"""
