# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:16:16 2019

@author: paulc
"""

import nltk
from nltk.corpus import state_union
from nltk.corpus import inaugural
from nltk.util import ngrams
from nltk.lm import NgramCounter
from nltk.lm import Vocabulary

from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

from nltk.lm import Lidstone
from nltk.lm import Laplace
from nltk.lm import KneserNeyInterpolated

# Exercise 1

president_unigrams = {}

for president in inaugural.fileids():
    text_unigrams = [ngrams(sent, 1) for sent in inaugural.sents(president)]
    ngram_counts=NgramCounter(text_unigrams)
    president_unigrams[president] = ngram_counts.N()
    
inverse_unigrams = [(value, key) for key, value in president_unigrams.items()]
print(max(inverse_unigrams)[1],max(inverse_unigrams)[0]) #longest discourse for Harrison in 1841
print(min(inverse_unigrams)[1],min(inverse_unigrams)[0]) #shortest discourse for Washington in 1793

president_vocabulary = {}

for president in inaugural.fileids():
    vocab = Vocabulary(inaugural.words(president), unk_cutoff=2)
    president_vocabulary[president] = len(vocab)

inverse_vocabulary = [(value, key) for key, value in president_vocabulary.items()]
print(max(inverse_vocabulary)[1],max(inverse_vocabulary)[0]) #richest vocabulary for Harrison in 1841
print(min(inverse_vocabulary)[1],min(inverse_vocabulary)[0]) #poorest vocabulary for Washington in 1793

president_vocabulary_state_union = {}

for president in state_union.fileids():
    vocab = Vocabulary(state_union.words(president), unk_cutoff=2)
    president_vocabulary_state_union[president] = len(vocab)

inverse_vocabulary_state_union = [(value, key) for key, value in president_vocabulary_state_union.items()]
print(max(inverse_vocabulary_state_union)[1],max(inverse_vocabulary_state_union)[0]) #richest vocabulary for Truman in 1946
print(min(inverse_vocabulary_state_union)[1],min(inverse_vocabulary_state_union)[0]) #poorest vocabulary for Johnson in 1963




# Exercise 2

train, vocab = padded_everygram_pipeline(2, state_union.sents())
lm = MLE(2)
lm.fit(train, vocab)
print(lm.counts['America'])
print(lm.counts[['bless']]['America'])
print(lm.score('the'))
print(lm.score("America", ["bless"]))

train, vocab = padded_everygram_pipeline(2, state_union.sents())
lm = Lidstone(2, 0.5)
lm.fit(train, vocab)
print(lm.counts['America'])
print(lm.counts[['bless']]['America'])
print(lm.score('the'))
print(lm.score("America", ["bless"]))

train, vocab = padded_everygram_pipeline(2, state_union.sents())
lm = Laplace(2)
lm.fit(train, vocab)
print(lm.counts['America'])
print(lm.counts[['bless']]['America'])
print(lm.score('the'))
print(lm.score("America", ["bless"]))

train, vocab = padded_everygram_pipeline(2, state_union.sents())
lm = KneserNeyInterpolated(2)
lm.fit(train, vocab)
print(lm.counts['America'])
print(lm.counts[['bless']]['America'])
print(lm.score('the'))
print(lm.score("America", ["bless"]))

#EXERCISE 3

train, vocab = padded_everygram_pipeline(2, state_union.sents('1945-Truman.txt'))
lm = MLE(2)
lm.fit(train, vocab)
print(lm.generate(100))


# Exercice 4

from neuralLG import dataset_preparation, create_model, generate_text
data = inaugural.raw()

X, Y, msl, total_words = dataset_preparation(data)
model = create_model(X, Y, msl, total_words)

text = generate_text("", 3, msl, model)
print(text)


























