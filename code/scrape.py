# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:19:53 2020

@author: Hannah
This code exists to convert the pdf into something readable
That is its sole purpose in life
I will help it fulfil its purpose
"""


import PyPDF2 
import textract 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
import re

filename = '2019.pdf'
pdfFileObj = open(filename,'rb') 
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

num_pages = pdfReader.numPages
count = 0
text = ""

while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()

#Test if it works   
#print(text)


#encoding from pdf
mystring = str(text.encode('utf-8'))

#fixing the broken things:
mystring = mystring.replace("\\xcb\\x9a", "ff")
mystring = mystring.replace("\\xcb\\x9c", "fi")
mystring = mystring.replace("\\xcb\\x9b", "ffi")
mystring = mystring.replace("\\xc3\\xb6", "o")

mystring = mystring.replace("\\n","")



#split into sentences
mylist=mystring.split(".")

#print(len(mylist))


f= open("encoded2019.txt","w+")
f.write(mystring)
f.close() 