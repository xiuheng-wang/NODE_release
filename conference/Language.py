import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from tqdm import tqdm 

data = pd.read_csv("./data/Language Detection.csv")
data.info()
print(data["Language"].value_counts())

X = data["Text"]
data_list = []

# convert text to vectors by Word2vec Word Embeddings
for text in X:         
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)      # removing the symbols and numbers
    text = re.sub(r'[[]]', ' ', text)   
    text = text.lower()          # converting the text to lower case
    data_list.append(text)       # appending to data_list
w2v_model = Word2Vec(sentences=data_list, vector_size=20, window=5, min_count=1, workers=4)
w2v_words = list(w2v_model.wv.index_to_key)

# compute average word2vec for each tweet.
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(data_list): # for each tweet/sentence
    sent_vec = np.zeros(20)
    cnt_words =0; # num of words with a valid vector in the sentence/tweet
    for word in sent: # for each word in a tweet/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vectors[0]))

# replace text with vector and split data
data['Text'] = sent_vectors
data.info()
print(data.head())
df = data.copy()
French = df.loc[df["Language"] == 'French']
French = French.reset_index()
Malayalam = df.loc[df["Language"] == 'Malayalam']
Malayalam = Malayalam.reset_index()
Arabic = df.loc[df["Language"] == 'Arabic']
Arabic = Arabic.reset_index()

# concatenate these three data
data = pd.concat([French, Malayalam, Arabic]).reset_index()
Y = data["Text"].tolist()
Y = np.array(Y)

np.save('./baselines/data/Language Detection.npy', Y.transpose())