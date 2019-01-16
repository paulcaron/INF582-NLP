#-*- coding: utf-8 -*-

import numpy
import pylab
import math
import matplotlib.pyplot as plt
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / float(len(blob.words))

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    ret = math.log(len(bloblist) / (1.0 + n_containing(word, bloblist)))
    if (ret < 0.0):
        return 0.0
    return ret

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)




            

doc1 = tb("""Python is a 2000 made-for-TV horror movie directed by Richard
Clabaugh. The film features several cult favorite actors, including William
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean
Whalen. The film concerns a genetically engineered snake, a python, that
escapes and unleashes itself on a small town. It includes the classic final
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles,
 California and Malibu, California. Python was followed by two sequels: Python
 II (2002) and Boa vs. Python (2004), both also made-for-TV films.""")

doc2 = tb(""" Python is a very nice programming programming programming language
language language language used by many researchers, engineers and data scientists.""")

doc3 = tb("""The Colt Python is a .357 Magnum caliber revolver formerly
manufactured by Colt's Manufacturing Company of Hartford, Connecticut.
It is sometimes referred to as a "Combat Magnum".[1] It was first introduced
in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued
Colt Python targeted the premium revolver market segment. Some firearm
collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy
Thompson, Renee Smeets and Martin Dougherty have described the Python as the
finest production revolver ever made.""")

doc4 = tb("""hello hello hello hello hello hello hello hello hello hello hello""")

bloblist = [doc1, doc2, doc3, doc4]



## TODO: write your own functions for tf and idf


doc1 = """Python is a 2000 made-for-TV horror movie directed by Richard
Clabaugh. The film features several cult favorite actors, including William
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean
Whalen. The film concerns a genetically engineered snake, a python, that
escapes and unleashes itself on a small town. It includes the classic final
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles,
 California and Malibu, California. Python was followed by two sequels: Python
 II (2002) and Boa vs. Python (2004), both also made-for-TV films."""

doc2 = """ Python is a very nice programming programming programming language
language language language used by many researchers, engineers and data scientists."""

doc3 = """The Colt Python is a .357 Magnum caliber revolver formerly
manufactured by Colt's Manufacturing Company of Hartford, Connecticut.
It is sometimes referred to as a "Combat Magnum".[1] It was first introduced
in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued
Colt Python targeted the premium revolver market segment. Some firearm
collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy
Thompson, Renee Smeets and Martin Dougherty have described the Python as the
finest production revolver ever made."""

doc4 = """hello hello hello hello hello hello hello hello hello hello hello"""

doclist = [doc1,doc2,doc3,doc4]

def my_tf(word, doc):
    n_occ=0
    n_words=0
    for w in doc.replace(',','').replace('.','').replace('(','').replace(')','').replace(':','').replace('"','').split():
        if w==word:
            n_occ+=1
        n_words+=1
    return n_occ/float(n_words)

def my_idf(word, doclist):
    N_t=0
    N=len(doclist)
    for doc in doclist:
        for w in doc.replace(',','').replace('.','').replace('(','').replace(')','').split():
            if w==word:
                N_t+=1
                break
    return math.log(N/(1+N_t))

def my_tfidf(word, doc, doclist):
    return my_tf(word, doc) * my_idf(word, doclist)



num_docs = len(bloblist)


## TODO: print top words per document
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i+1))
    scores = {word:tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    for word, score in sorted_words[:10]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score,5)))

for i, doc in enumerate(doclist):
    print("Top words in document {}".format(i+1))
    scores = {word:my_tfidf(word, doc, doclist) for word in doc.replace(',','').replace('.','').replace('(','').replace(')','').split()}
    sorted_words = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    for word, score in sorted_words[:10]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score,5))) 
    
    
    
    
    
    