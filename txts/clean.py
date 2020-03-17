# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:30:36 2020

@author: hjbec
"""

import nltk
import re 
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def remove_www(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"www\S+", "", sample)
def remove_bracket(sample):
    """Remove text between curly brackets"""
    return re.sub(r"\s*{.*}\s*", " ", sample)
def remove_parens(sample):
    """Remove text between parens"""
    return re.sub("[\(\[].*?[\)\]]", "",  sample)


f = open("ipcc_ar5.txt", "r")
line = f.read()

keywords = ["malaria", "diarrhoea", "infection", "disease", "sars", "measles", 
            "pneumonia", "epidemic", "pandemic", "public health", "health care", 
            "epidemiology", "healthcare", "health", "mortality", "morbidity", 
            "nutrition", "illness", "infectious", "ncd", "non-communicable disease", 
            "noncommunicable disease", "communicable disease", "air pollution", 
            "nutrition", "malnutrition", "mental disorder", "stunting"]

"""
Bit of a Hack to get rid of the page headers and footers as they stick to the
text otherwise (Lancet report only)



line = line.replace("www.thelancet.com Vol 387 May 14, 2016", " ")
line = line.replace("www.thelancet.com Vol 391 February 10, 2018", " ")
line = line.replace("www.thelancet.com Vol 392 December 8, 2018", " ")
line = line.replace("www.thelancet.com Vol 394 November 16, 2019", " ")



End of Hannah's ridiculously stupid hack
"""

#How do I remove all hyperlinks from the text?
#Saved by a regex! You heard me!
line = remove_URL(line)
line = remove_www(line)

#test = "(high confidence) {1.2.1, 1.2.2, Figure 1.1, Figure 1.3, 3.3.1, 3.3.2}"
#print(remove_bracket(test))

myarray = line.split("\n\n")
print(len(myarray))
count = 0
o = open("full","w")


for i in myarray:
    #sent = " ".join(w for w in nltk.wordpunct_tokenize(i) \
    #     if w.lower() in dict or not w.isalpha())
    cleaned = i.replace('\n', ' ')
    cleaned = remove_bracket(cleaned)
    cleaned = remove_parens(cleaned)
    words = nltk.word_tokenize(cleaned)
    #I don't want paragraphs with less than 30 words. Life has taught me these
    # sentences are junk. 
    label = 0
    if len(words) >100:
        for j in words:
            if lemmatizer.lemmatize(j) in keywords:
               label = 1 # is this paragraph linking health to climate change?
               count+= 1
        o.write(cleaned + "\t" + str(label) + "\n") # Set the label to 1

print("Labelled 1:" + str(count))
print("Labelled 0:" + str(len(myarray)-count))
                        
o.close()
f.close()  
  