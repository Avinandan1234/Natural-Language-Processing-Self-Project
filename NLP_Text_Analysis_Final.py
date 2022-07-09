#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT THE TEXT FILES

import glob
import os

file_list = glob.glob(os.path.join(os.getcwd(), "Black_Coffer_Articles", "*.txt"))
corpus = []

for file_path in file_list:
    with open(file_path) as f_input:
        corpus.append(f_input.read())

#print(corpus)
#print(len(corpus))


# In[2]:


#REMOVE EXTRA WHITE SPACES
import string

def remove_whitespace(text): 
    return  " ".join(text.split()) 

raw_corpus = list(map(remove_whitespace, corpus))
#print(raw_corpus[0])
#print(len(raw_corpus))


# In[3]:


#REMOVE STOPWORDS

#Using nltk tokenization and nltk stopwords
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

def remove_stopwords(text): 
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words] 
    return ' '.join(filtered_text)

corpus_cleaned_stopwords = list(map(remove_stopwords, corpus))
#corpus_cleaned_stopwords[0]


# In[4]:


#Find the total number of sentences for each article
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

#Defining sentence counter of an article (.!?)
def sent_counter(article):
    sents = sent_tokenize(article)
    sent_count = (len(sents))
    return sent_count

#Counting the number of sentences of each article
sent_count = []
for article in corpus_cleaned_stopwords:
    sent_count.append(sent_counter(article))

#print(sent_count)   


# In[5]:


#Find the total number of words for each article
#Function to remove punctuations from the text
import string
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 

#Defining word counter of an article
def word_counter(article):
    article = remove_punctuation(article)
    word_count = len(article.split())
    return word_count

#Counting the number of words of each article
word_count = []
for article in corpus_cleaned_stopwords:
    word_count.append(word_counter(article))
#print(word_count)


# In[6]:


#Find the total number of characters in each article
#Defining character counter of an article
def char_counter(article):
    article = remove_punctuation(article)
    word_tokens = word_tokenize(article) #tokenizing & splitting gives the same words
    #print(word_tokens)
    char_count=[]
    for word in word_tokens:
        char_count.append(len(word))
    #print(char_count)
    total_char_count = sum(char_count)
    return total_char_count

#We can also use this other function for finding the number of characters
# def char_counter(article):
#     article = remove_punctuation(article)
#     char_count = map(len,article.split())
#     total = sum(char_count)
#     return total

#Counting the number of characters of each article
char_count = []
for article in corpus_cleaned_stopwords:
    char_count.append(char_counter(article))
#print(char_count)


# In[7]:


#Find the total number of syllables in each article
#Defining syllable counter of an article

# def syllable_count(word):
#     word = word.lower()
#     count = 0
#     vowels = "aeiouy"
#     if word[0] in vowels:
#         count += 1
#     for index in range(1, len(word)):
#         if word[index] in vowels and word[index - 1] not in vowels:
#             count += 1
#     if word.endswith("e"):
#         count -= 1
#     if count == 0:
#         count += 1
#     return count

# def syllable_count2(word):
#     syllable = 0
#     for vowel in ['a','e','i','o','u']:
#         syllable += word.count(vowel)
#     for ending in ['es','ed','e']:
#         if word.endswith(ending):
#             syllable -= 1
#     if word.endswith('le'):
#         syllable += 1
#     return syllable

#!pip install textstat
from textstat.textstat import textstatistics
def syllables_count(word):
    return textstatistics().syllable_count(word)


#Counting the number of syllables of each article
syllable_count = []
for article in corpus_cleaned_stopwords:
    article = remove_punctuation(article)
    count = 0
    words = article.split()
    for word in words:
        count += syllables_count(word) 
    syllable_count.append(count)
    
#print(syllable_count)


# In[8]:


#Find the total number of complex words in each article
#Defining complex word counter of an article
# Return total Difficult Words in a text

def complex_word_counter(article):
    article = remove_punctuation(article)
    words = article.split()
    # difficult words are those with syllables > 2
    diff_words_set = set()
    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count > 2:
            diff_words_set.add(word)
    return len(diff_words_set)


#Counting the number of complex words of each article
complex_words = []
for article in corpus_cleaned_stopwords:
    complex_words.append(complex_word_counter(article))


# In[9]:


#Calculating the Gunning Fog Index
Fog_Index = [0.4*(i+j) for i, j in zip([i / j for i, j in zip(word_count, sent_count)], [100*(i/j) for i, j in zip(complex_words, word_count)])]
#print(Fog_Index)


# In[10]:


#Find the total number of personal pronouns in each article
#Defining personal pronoun counter of an article
import re
def personal_pronoun_counter(word):
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
    pronouns = pronounRegex.findall(word)
    count = len(pronouns)
    return count

##Calculating the Personal pronoun count for each article
personal_pronoun_count = []
for article in raw_corpus:
    article = remove_punctuation(article)
    words = article.split()
    count = 0
    for word in words:
        count += personal_pronoun_counter(word)
    personal_pronoun_count.append(count)
#print(personal_pronoun_count)


# In[11]:


#Dictionary Based Sentiment Analysis
import nltk
from nltk.corpus import opinion_lexicon
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

#Download Opinion Dictionary
#nltk.download('opinion_lexicon') 

#Positive and Negative dictionary words
positive_wds = set(opinion_lexicon.positive())
negative_wds = set(opinion_lexicon.negative())

#Returns the positive and negative score of each sentence
def score_sent(sent):
    #"""Returns a score btw -1 and 1"""
    sent = [word.lower() for word in sent if word.isalnum()]
    total = len(sent)
    pos = len([word for word in sent if word in positive_wds])
    neg = len([word for word in sent if word in negative_wds])
    return pos,neg,total;

#Returns the positive and negative score of a whole article   
def score_article(article):
    pos_score, neg_score, total_words = 0, 0,0
    sents = sent_tokenize(article)
    for sent in sents:
        sent = remove_punctuation(sent)
        words = word_tokenize(sent)
        pos, neg, total = score_sent(words)
        total_words += total
        pos_score += pos
        neg_score += neg
    return pos_score, neg_score, total_words


##Calculating the score for each article
pos_score, neg_score, total_words = [], [], []
for article in corpus_cleaned_stopwords:
    pos, neg, total = score_article(article)
    total_words.append(total)
    pos_score.append(pos)
    neg_score.append(neg)


# In[12]:


#Calculating polarity score of each article
#defining the polarity score function

# def polarity_score(pos,neg):
#     score = [(pos-neg)/((pos+neg)+0.000001) for pos, neg in zip(pos, neg)]
#     return score
# polarity = polarity_score(pos_score,neg_score)
# print(polarity)

#Using textblob for Polarity score:
from textblob import TextBlob
def getPolarity(article):
    return TextBlob(article).sentiment.polarity

polarity_score = []
for article in corpus_cleaned_stopwords:
    article = remove_punctuation(article)
    polarity_score.append(getPolarity(article))
    
#print(polarity_score)


# In[13]:


#Calculating subjectivity score of each article
#defining the subjectivity score function

# def subjectivity_score(pos,neg,total):
#     score = [(pos+neg)/(total+0.000001) for pos,neg,total in zip(pos,neg,total)]
#     return score
# subjectivity = subjectivity_score(pos_score,neg_score, total_words)
# print(subjectivity)

#Using textblob for subjectivity score:
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

subjectivity_score = []
for article in corpus_cleaned_stopwords:
    article = remove_punctuation(article)
    subjectivity_score.append(getSubjectivity(article))
    
#print(subjectivity_score)


# In[14]:


#Saving the output data structure in the mentioned format
#Import the output file
import pandas as pd
output_df = pd.read_excel(os.path.join(os.getcwd(), "Downloads/Output_DS.xlsx"))

#1.a Saving the positive score of each article
output_df['POSITIVE SCORE'] = pos_score

#1.b Saving the negative score of each article
output_df['NEGATIVE SCORE'] = neg_score

#1.c Saving the polarity score
output_df['POLARITY SCORE'] = polarity_score

#1.d Saving the subjectivity score
output_df['SUBJECTIVITY SCORE'] = subjectivity_score

#2.a Saving the average sentence length
output_df['AVG SENTENCE LENGTH'] = [i / j for i, j in zip(word_count, sent_count)]

#2.b Saving the percentage of complex words
output_df['PERCENTAGE OF COMPLEX WORDS'] = [100*(i/j) for i, j in zip(complex_words, word_count)]

#2.c Saving the Fog index
output_df['FOG INDEX'] = Fog_Index

#3. Save the Average Number of Words Per Sentence
output_df['AVG NUMBER OF WORDS PER SENTENCE'] = [i / j for i, j in zip(word_count, sent_count)]

#4. Saving the Word Count
output_df['COMPLEX WORD COUNT'] = complex_words

#5. Saving the Word Count
output_df['WORD COUNT'] = word_count

#6. Saving the Syllable Count per word
output_df['SYLLABLE PER WORD'] = [i / j for i, j in zip(syllable_count, word_count)]

#7 Saving the personal pronoun count.
output_df['PERSONAL PRONOUNS'] = personal_pronoun_count

#8. Save the Average Word Length for each article
output_df['AVG WORD LENGTH'] = [i / j for i, j in zip(char_count, word_count)]

output_df


# In[15]:


file_name = 'Output_Data.xlsx'
output_df.to_excel(file_name)

