import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import io

"""
The following code was partially adapted from this article (https://www.dominodatalab.com/blog/named-entity-recognition-ner-challenges-and-model)
"""

# read the annotated dataset
data_df = pd.read_csv("ner_datasetreference.csv", encoding="iso-8859-1", header=0)
data_df.head()

#replace zero values with last valid observation

data_df = data_df.fillna(method="ffill")

data_df.head()

# count number of sentences and number of words
#print("Total number of sentences in the dataset:", data_df["Sentence #"].nunique())
#print("Total words in the dataset:", (data_df.shape[0]))

# count labels
data_df[data_df["Tag"]!="O"]["Tag"].value_counts()

# visualising the tag frequency
data_df["Tag"].value_counts().plot(kind="bar", figsize=(4,2))

# count words for each sentences
word_counts = data_df.groupby("Sentence #")["Word"].agg(["count"])
#print(word_counts)

# define the longest sentence in the corpus
MAX_SENTENCE = word_counts.max()[0]
#print(MAX_SENTENCE)

# define number of unique words and unique tags
uniq_words = list(set(data_df["Word"].values))
uniq_tags = list(set(data_df["Tag"].values))
                  

#len_uniq_tag=len(uniq_tags)
#len_uniq_tag


# Implement the necessary feature engineering.

""" 
    build a dictionary (word2id) that assigns a unique integer value to every word from the corpus and 
    a reversed dictionary (id2word) that maps indices to words
    
"""

word2id = {word: idx + 2 for idx, word in enumerate(uniq_words)}

word2id["--UNKNOWN_WORD--"]=0

word2id["--PADDING--"]=1

id2word = {idx: word for word, idx in word2id.items()}
len_uniq_words = len(id2word)

print(len_uniq_words)

#print(word2id["Crumpton"])
#print(id2word[24640])


#  build a similar dictionary for the various tags
tag2id = {tag: idx + 1 for idx, tag in enumerate(uniq_tags)}
tag2id["--PADDING--"] = 0
id2tag = {idx: word for word, idx in tag2id.items()}

#print(id2tag)


def create_tuples(data):
    
    """ return a tuple containing of each token, the part of speech it represents, and its corresponding tag"""
    
    iterator = zip(data["Word"].values.tolist(),
                   data["POS"].values.tolist(),
                   data["Tag"].values.tolist())
    return [(word, pos, tag) for word, pos, tag in iterator]

# apply this function to the entire dataset
sentences = data_df.groupby("Sentence #").apply(create_tuples).tolist()

#print(sentences[0])


"""  
     extract the features (X) and labels (y) for the model 
     discard the part of speech data, as it is not needed for this implementation.
     
"""

X = [[word[0] for word in sentence] for sentence in sentences]
y = [[word[2] for word in sentence] for sentence in sentences]
#print("X[0]:", X[0])
#print("y[0]:", y[0])


# replace each word with its corresponding index from the dictionary

X = [[word2id[word] for word in sentence] for sentence in X]
y = [[tag2id[tag] for tag in sentence] for sentence in y]
#print("X[0]:", X[0])
#print("y[0]:", y[0])

# for the LSTM model to process input of consistent length, eauch sentence should be padded to match the longest sentence

X = [sentence + [word2id["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in X]
y = [sentence + [tag2id["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in y]
#print("X[0]:", X[0])
#print("y[0]:", y[0])

TAG_COUNT = len(tag2id)
print(TAG_COUNT)

# split data in test, development udn train 
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)
X_train,X_dev,y_train,y_dev=train_test_split(X_main, y_main, test_size=0.1, random_state=1234)

#print("Number of sentences in the training dataset: {}".format(len(X_train)))
#print("Number of sentences in the test dataset : {}".format(len(X_test)))

#print(X_train)
