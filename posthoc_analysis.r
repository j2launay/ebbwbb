data <- read.csv(file = "./perplexity.csv")
head(data)
data$perplexity <- as.numeric(data$perplexity)
data$dataset.name <- as.factor(data$dataset.name)
data$model <- as.factor(data$model)
data$method <- as.factor(data$method)

data <- data[sample(nrow(data)), ]
head(data)

#temp_data <- subset(data, dataset.name == 'spam') 
temp_data <- subset(data, model == 'rf') 
tt <- pairwise.t.test(temp_data$perplexity, temp_data$method, p.adjust.method="bonferroni")
print(tt)

data <- read.csv(file = "./distance counterfactual target instance lev.csv")
head(data)
data$distance_lev <- as.numeric(data$distance_lev)
data$dataset <- as.factor(data$dataset)
data$model <- as.factor(data$model)
data$method <- as.factor(data$method)

data <- data[sample(nrow(data)), ]
head(data)

#temp_data <- subset(data, dataset.name == 'spam') 
temp_data <- subset(data, model == 'Random Forest') 
temp_data <- subset(temp_data, vectorizer == '_TfidfVectorizer') 
temp_data <- subset(temp_data, vectorizer == '') 
head(temp_data)
tt <- pairwise.t.test(temp_data$distance_lev, temp_data$method, p.adjust.method="bonferroni")
print(tt)


data <- read.csv(file = "./distance counterfactual target instance bert.csv")
head(data)
data$distance_bert <- as.numeric(data$distance_bert)
data$dataset <- as.factor(data$dataset)
data$model <- as.factor(data$model)
data$method <- as.factor(data$method)

data <- data[sample(nrow(data)), ]
head(data)

#temp_data <- subset(data, dataset.name == 'spam') 
temp_data <- subset(data, model == 'Neural Network') 
temp_data <- subset(temp_data, vectorizer == '_TfidfVectorizer') 
temp_data <- subset(temp_data, vectorizer == '') 
head(temp_data)
tt <- pairwise.t.test(temp_data$distance_bert, temp_data$method, p.adjust.method="bonferroni")
print(tt)
