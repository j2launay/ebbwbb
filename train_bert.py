from torch.utils.data import Dataset
from torch import tensor
from transformers import DistilBertForSequenceClassification, AutoTokenizer, TextClassificationPipeline, Trainer, TrainingArguments
from generate_dataset import generate_dataset, preparing_dataset

training_args = TrainingArguments(
                output_dir='./model/bert/',          # output directory
                num_train_epochs=3,              # total number of training epochs
                per_device_train_batch_size=16,  # batch size per device during training
                per_device_eval_batch_size=64,   # batch size for evaluation
                warmup_steps=500,                # number of warmup steps for learning rate scheduler
                weight_decay=0.01,               # strength of weight decay
                logging_dir='./logs',            # directory for storing logs
                logging_steps=10,
            )

class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    dataset_names = ["fake", "polarity", "spam"] # polarity, spam, fake, religion, baseball, ag_news
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    for dataset_name in dataset_names:                                                                          
        x, y, class_names = generate_dataset(dataset_name)

        bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(set(y)))
        def tokenize_function(examples):
            return tokenizer(examples, padding="max_length", truncation=True)

        pipe = TextClassificationPipeline(model=bert, tokenizer=tokenizer, batch_size=100, return_all_scores=True)#batch_size=64
        x_train, x_test, y_train, y_test, x_train_vectorize, x_test_vectorize, vectorizer = preparing_dataset(x, y, dataset_name, vectorizer)
        train_encodings = tokenizer(x_train_vectorize, truncation=True, padding=True)
        test_encodings = tokenizer(x_test_vectorize, truncation=True, padding=True)
        train_dataset = IMDbDataset(train_encodings, y_train)
        val_dataset = IMDbDataset(test_encodings, y_test)
        trainer = Trainer(
            model=bert,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset             # evaluation dataset
        )
        trainer.train()
        if dataset_name == "polarity":
            trainer.save_model("./model/bert/") 
        elif "fake" in dataset_name:
            trainer.save_model("./model/fake/bert/")
        elif "spam" in dataset_name:
            trainer.save_model("./model/spam/bert/")
        else:
            trainer.save_model("./model/ag_news/bert/")