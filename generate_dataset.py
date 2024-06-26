from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import unidecode

def preparing_dataset(x, y, dataset_name, vectorizer):
    """
    Train test split the input dataset and vectorize the input dataset with the input vectorizer
    """
    print("Preparing dataset", dataset_name)
    x_train_vectorize, x_test_vectorize, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)
    vectorizer.fit(x_train_vectorize)
    x_train = vectorizer.transform(x_train_vectorize)
    x_test = vectorizer.transform(x_test_vectorize)
    return x_train, x_test, y_train, y_test, x_train_vectorize, x_test_vectorize, vectorizer

def generate_dataset(dataset_name):
    # Function used to get dataset depending on the dataset name
    if "polarity" in dataset_name:
        path='./dataset/rt-polaritydata'
        X = []
        y = []
        class_names = ['positive', 'negative']
        f_names = ['rt-polarity.neg', 'rt-polarity.pos']
        for (l, f) in enumerate(f_names):
            for line in open(os.path.join(path, f), 'rb'):
                try:
                    line.decode('utf8')
                    line = line.decode('ascii')
                except:
                    continue
                X.append(line.strip())
                y.append(l)
    
    elif 'spam' in dataset_name:
        spam = pd.read_csv("./dataset/spam.csv", encoding='latin-1')
        X, y = spam['message'].to_list(), spam['label'].to_list()
        class_names = ['ham', 'spam']

    elif "fake" in dataset_name:
        fake_news = pd.read_csv("./dataset/news.csv")
        X, y = fake_news['message'].to_list(), fake_news['label'].to_list()
        class_names = ['news', 'fake']

    return X, y, class_names

def prepare_dataset(dataset="spam"):
    """
    Prepare the dataset from the initial datasets to get the csv files used in the experiments
    """
    if dataset=="spam":
        sms = pd.read_csv("./dataset/spam/spam.csv", encoding='latin-1')
        sms.dropna(how="any", inplace=True, axis=1)
        for i, message in enumerate(sms['v2']):
            unaccented_string = unidecode.unidecode(message)
            sms['v2'][i] = unaccented_string
        sms.columns = ['label', 'message']
        print(sms.head)
        try:
            sms.drop(['Unnamed: 0'], axis=1, inplace=True)
        except:
            print("preparing spam dataset")
        # The following allows to balance the dataset of spam detection through oversampling
        sms['label'] = np.where((sms.label == 'ham'),0,1)
        index_spam = np.where(sms.label == 1)
        index_ham = np.where(sms.label == 0)
        balance_dataset = sms.loc[index_spam]
        print("balance", balance_dataset.head(10))
        sms = pd.concat([sms, balance_dataset, balance_dataset, balance_dataset, balance_dataset])
        index_spam = np.where(sms.label == 1)
        index_ham = np.where(sms.label == 0)
        print("spam", len(index_spam[0]))
        print("ham", len(index_ham[0]))
        sms = sms.sample(frac=1)
        sms.reset_index(drop=True, inplace=True)
        sms.to_csv("./dataset/spam.csv")

    elif dataset == "fake_news":
        fake_news = pd.read_csv("./dataset/fake_news/test.csv")
        fake_news.dropna(inplace=True)
        fake_news_title, fake_news_label = fake_news['title'], [1]*fake_news['title'].shape[0]
        fake_news = pd.DataFrame({'label': pd.Series(fake_news_label), 'message':fake_news_title})
        fake_news.dropna(inplace=True)
        fake_news.reset_index(drop=True, inplace=True)
        fake_news.to_csv("./dataset/fake_news.csv")
        
        news = pd.read_json("./dataset/fake_news/News_Category.json", lines=True)
        news.dropna(inplace=True)
        news_title, news_label = news['headline'], [0]*news['headline'].shape[0]
        news = pd.DataFrame({'label': pd.Series(news_label), 'message':news_title})
        news.dropna(inplace=True)
        news.reset_index(drop=True, inplace=True)
        news = news[:fake_news.shape[0]]
        
        news_dataset = pd.concat([news, fake_news], axis=0)
        news_dataset = news_dataset.sample(frac=1)
        news_dataset.reset_index(drop=True, inplace=True)
        news_dataset['label'] = news_dataset['label'].astype(int)
        print(news_dataset)
        news_dataset.to_csv("./dataset/news.csv")

def count_mean_std_in_corpus(dataset_name):
    """
    Measures the average length and standard deviation of the input dataset
    """
    dataset, _, _ = generate_dataset(dataset_name)
    my_counter, all_length = [], []
    for sentence in dataset:
        sentence_split = str(sentence).split(" ")
        all_length.append(len(sentence_split))
        for word in sentence_split:
            my_counter.append(word)
    print(dataset_name)
    print("mean", np.mean(all_length))
    print("std", np.std(all_length))
    tot_words = []
    for word in my_counter:
        tot_words.append(word.lower())
    my_counter = list(set(tot_words))
    print("nb total of words", len(my_counter))

if __name__ == "__main__":
    #prepare_dataset("fake_news")
    #prepare_dataset()
    count_mean_std_in_corpus("fake")
        