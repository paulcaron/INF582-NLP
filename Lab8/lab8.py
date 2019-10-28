# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:36:13 2019

@author: paulc
"""

# Test 1
import nltk
from nltk.corpus import treebank
print(treebank.parsed_sents('wsj_0001.mrg')[0])
treebank.parsed_sents('wsj_0001.mrg')[0].draw()



# Test 2

from nltk import Tree
s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
t = Tree.fromstring(s)
t.draw() 
t.draw()
print(t)


# Test 3
t.chomsky_normal_form()
t.productions()

for p in t.productions():
    print (p.lhs()) #left part (nonterminal)
    print (p.rhs()) #right part of the rule (sequence)
    
# Test 4
from nltk import nonterminals, Nonterminal, Production
# Create some nonterminals
S, NP, VP, PP = nonterminals('S, NP, VP, PP')
N, V, P, Det = nonterminals('N, V, P, Det')
# Create a production rule ant print it
print(Production(S, [NP, PP]))


# Exercise 1

NP, PP, VP, VB = nonterminals('NP, PP, VP, VB')
production_NP_NPPP = Production(NP, [NP, PP])
production_VP_VBNP = Production(VP, [VB, NP])
productions=[]
for item in treebank.fileids():
    for tree in treebank.parsed_sents(item):
        # perform optional tree transformations, e.g.:
        tree.chomsky_normal_form() # Don’t forget to normalize in CNF the parse tree
        productions += tree.productions()

NP_C = 0
NP_NPPP = 0
VP_C = 0
VP_VBNP = 0

for production in productions:
    if production.lhs() == NP:
        NP_C+=1
        if production == production_NP_NPPP:
            NP_NPPP+=1
    if production.lhs() == VP:
        VP_C+=1
        if production == production_VP_VBNP:
            VP_VBNP+=1

prob_NP_NPPP = NP_NPPP / NP_C
prob_VP_VBNP = VP_VBNP / VP_C

print("The probability of NP -> NP PP is ", str(prob_NP_NPPP))
print("The probability of VP -> VB NP is ", str(prob_VP_VBNP))
        
# Exercise 2
from nltk import Nonterminal, induce_pcfg
NP = Nonterminal('NP')
grammar = induce_pcfg(NP, productions)
    
for production in grammar.productions():
    if production.lhs() == production_NP_NPPP.lhs() and production.rhs() == production_NP_NPPP.rhs():
        print("The probability of NP -> NP PP is ", str(production.prob()))
    elif production.lhs() == production_VP_VBNP.lhs() and production.rhs() == production_VP_VBNP.rhs():
        print("The probability of VP -> VB NP is ", str(production.prob()))


# Test 5
from nltk.parse import ViterbiParser
s="I saw John with my eyes"
tokens = s.split() #simplified – you can use a tokenizer
parser=ViterbiParser(grammar) #using the grammar induced by the Treebank
parses = parser.parse_all(tokens) #the resulting parse tree

"""
s2="I saw John with my telescope"
tokens2 = s2.split() #simplified – you can use a tokenizer
parses2 = parser.parse_all(tokens2) #the resulting parse tree        
"""


from nltk.corpus import comtrans
data = comtrans.aligned_sents("alignment-en-fr.txt") #this will load the alignments English-French
from nltk import IBMModel1
# model=IBMModel1(data, 30) #data is the parallel corpus, 30 is the number of iterations for the EM algorithm

plotting_x = [i for i in range(10, 110, 10)]
plotting_y = [] 

for size in range(10, 110, 10):
    model=IBMModel1(data[:size], 30)
    plotting_y.append(model.translation_table["the"]["le"])
    print("With a corpus of size ", str(size), ", the probability is ", model.translation_table["the"]["le"])       
        
import matplotlib.pyplot as plt
plt.plot(plotting_x, plotting_y)    


plotting_y = [] 

for iterations in range(10, 110, 10):
    model=IBMModel1(data[:30], iterations)
    plotting_y.append(model.translation_table["the"]["le"])
    print("With ", str(iterations), " iterations, the probability is ", model.translation_table["the"]["le"])       
        
plt.plot(plotting_x, plotting_y) 


model = IBMModel1(data[:100], 30)

print(model.translation_table["dog"]["chien"])

"""
The value is 1e-12 (corresponding to 0). The pair has never been seen (in fact, the word "dog" has never been seen.)
"""
        
        
        
        
        
        
        
        
        
        
        
        
        
        