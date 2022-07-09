#!/usr/bin/env python
# coding: utf-8

# In[1]:


#installing required libraries
#!pip install requests
#!pip install BeautifulSoup4
#!pip install lxml


# In[2]:


#importing required libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os


# In[3]:


urls_df = pd.read_csv('input_url.csv')
urls_df


# In[4]:


title_text = []
post_content_text = []

for i in range(len(urls_df)): 
    url = urls_df.loc[i,'URL']
    #print(url)
    
    #getting the html_text from all the links
    html_text = requests.get(url, headers={"User-Agent": "XY"}).text
    #print(html_text) 
    #inspecting the page
    
    #extracting data
    soup = BeautifulSoup(html_text, 'lxml')
    
    article = soup.find('article')

    #Getting the article title text
    title_text.append(article.find('h1', class_ = 'entry-title').text)
    #print(title_text[i])

    #Getting the article content text
    post_content_text.append(article.find('div', class_ = 'td-post-content').text)
    #print(post_content_text[i])

    url_id = urls_df.loc[i,'URL_ID']
    text_file_name = "{}.txt".format(url_id)
    text_path = os.path.join(os.getcwd(), "Black_Coffer_Articles", text_file_name)
    with open(text_path, 'w') as f:
        f.write("{}\n".format(title_text[i]))
        f.write("{}".format(post_content_text[i]))
    print("File saved: {}".format(text_file_name))

