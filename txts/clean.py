# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:30:36 2020

@author: hjbec
"""

import nltk
import re 


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def remove_www(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"www\S+", "", sample)

dict = set(nltk.corpus.words.words())

f = open("Appendix2019.txt", "r")
line = f.read()



"""
Bit of a Hack to get rid of the page headers and footers as they stick to the
text otherwise (Lancet report only)

"""

line = line.replace("www.thelancet.com Vol 387 May 14, 2016", " ")
line = line.replace("www.thelancet.com Vol 391 February 10, 2018", " ")
line = line.replace("www.thelancet.com Vol 392 December 8, 2018", " ")
line = line.replace("www.thelancet.com Vol 394 November 16, 2019", " ")


"""
End of Hannah's ridiculously stupid hack
"""

#How do I remove all hyperlinks from the text?
#Saved by a regex! You heard me!
line = remove_URL(line)
line = remove_www(line)

myarray = line.split("\n\n")
print(len(myarray))

o= open("full","a")

for i in myarray:
    #sent = " ".join(w for w in nltk.wordpunct_tokenize(i) \
    #     if w.lower() in dict or not w.isalpha())
    words = nltk.word_tokenize(i)
    #I don't want paragraphs with less than 20 words. Life has taught me these
    # sentences are junk. I list to life
    if len(words) >30:
     o.write(i.replace('\n', ' ') + "\t" + "1\n") # Set the lancet label to 1
    
  
  