# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:12:09 2020

@author: hjbec
"""


from urllib import request
from bs4 import BeautifulSoup
import re
import os
import urllib

# connect to website and get list of all pdfs
url="https://www.ipcc.ch/reports/"
response = request.urlopen(url).read()
soup= BeautifulSoup(response, "html.parser")     
links = soup.find_all('a', href=re.compile(r'(.pdf)'))




# clean the pdf link names
url_list = []
for el in links:
    if(el['href'].startswith('http')):
        url_list.append(el['href'])
    else:
        url_list.append("https://www.ipcc.ch/reports/" + el['href'])

print(url_list)


# download the pdfs to a specified location
for url in url_list:
    print(url)
    fullfilename = os.path.join('C:\\Users\\hjbec\\Documents\\work\\LancetProject', url.replace("https://www.ipcc.ch/reports/", ""))
    print(fullfilename)
    request.urlretrieve(url, fullfilename)