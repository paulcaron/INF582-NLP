# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:30:32 2019

@author: paulc
"""

import numpy as np
from viterbi import viterbi

print("Buildind sets")

set_Q = {}
set_V = {}

from nltk.corpus import brown
corpus = brown.tagged_sents()
training = corpus[:-10]
testing = corpus[-10:]

for sentence in corpus:
    for word in sentence:
        if not word[0] in set_V:
            set_V[word[0]] = 0
        if not word[1] in set_Q:
            set_Q[word[1]] = 0
            
print("Buildind V")

V = []
dict_V = {}
i = 0
for v in set_V.keys():
    dict_V[v] = i
    i+=1
    V.append(v)

print("Buildind Q")

Q = []
dict_Q = {}
i = 0
for q in set_Q.keys():
    dict_Q[q] = i
    i+=1
    Q.append(q)
    
print("Splitting into training and testing")



print("Building pi, A and B")

pi = np.zeros(len(Q))
A = np.zeros((len(Q), len(Q)))
B = np.zeros((len(Q), len(V)))

for sentence in training:
    pi[dict_Q[sentence[0][1]]] += 1
    for i in range(1, len(sentence)):
        previous_word = sentence[i-1]
        current_word = sentence[i]
        A[dict_Q[previous_word[1]], dict_Q[current_word[1]]] += 1
        B[dict_Q[previous_word[1]], dict_V[current_word[0]]] += 1
        
alpha = 0.1
d_A = len(Q)
d_B = len(V)
 
for i in range(len(Q)):
    A[i,:] = (A[i, :] + alpha) / (np.sum(A[i, :]) + alpha * d_A)
    B[i,:] = (B[i, :] + alpha) / (np.sum(B[i, :]) + alpha * d_B)
pi /= np.sum(pi)

sent = testing[0]
processed_sent = [dict_V[word[0]] for word in sent]
predicted = viterbi((pi, A, B), processed_sent)


print('%20s\t%5s\t%5s' % ('TOKEN', 'TRUE', 'PRED'))
for (wi, ti), pi in zip(sent, predicted[0][1:]):
    print('%20s\t%5s\t%5s' % (wi, ti, Q[pi]))



# 3. NER using Conditional Random Fields
    
import pandas as pd
data = pd.read_csv("ner_dataset/ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill") #repeat sentence number on each row

words = list(set(data["Word"].values)) #vocabulary V
n_words = len(words)

tags = list(set(data["Tag"].values)) #tags Q
n_tags = len(tags)

from crf_helper import *


getter = SentenceGetter(data) #transform sentences into sequences of (Word, POS, Tag)
sentences = getter.sentences

X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

from sklearn_crfsuite import CRF


from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)

# Best algorithm for PER: 'ap'
# Best algorithm for GEO: 'ap'
# Best algorithm for ORG: 'pa'

crf_ap = CRF(algorithm='ap', max_iterations=100)
crf_pa = CRF(algorithm='pa', max_iterations=100)


print("Fitting...")
crf_ap.fit(X_train, y_train)
crf_pa.fit(X_train, y_train)
print("Done")

pred_ap=crf_ap.predict(X_test)
pred_pa=crf_pa.predict(X_test)
report_ap = flat_classification_report(y_pred=pred_ap, y_true=y_test)
print(report_ap)
report_pa = flat_classification_report(y_pred=pred_pa, y_true=y_test)
print(report_pa)

# 4. NER using Deep Learning + CRFs


word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

from keras.preprocessing.sequence import pad_sequences
max_len=75
X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post",value=n_words-1)
y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post",
value=tag2idx["O"])

from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=20, input_length=max_len, mask_zero=True)(input) # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model) # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)
crf = CRF(n_tags)
out = crf(model)
model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.33, verbose=1)
test_pred = model.predict(X_test, verbose=1)




glove_dir = 'C:\Users\paulc\Desktop\INF582\lab7' #substitute with the right directory for your installation
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=20, input_length=max_len, mask_zero=True)(input) # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model) # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)
crf = CRF(n_tags)
out = crf(model)
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.33, verbose=1)
test_pred = model.predict(X_test, verbose=1)








