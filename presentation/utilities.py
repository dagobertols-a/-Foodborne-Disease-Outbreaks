# coding: utf-8
import math 
import pandas as pd
import numpy as np
import re, string, unicodedata
from numpy import array
from random import seed
import copy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Pandas columns processing

def convert_to_float(dataframe,column):
    #Convert a type int column to float type
    """
    Parameters: dataframe and column
    
    Return: column in float type
    """
    return dataframe[column].astype("float")


def text_to_list(dataframe,column,sep='; '):
    # Convert each text entry to a list of words
    
    # All words to lowercase
    lowercase_column=dataframe[column].str.lower()
    # List of words
    list_format_column=lowercase_column.str.split(sep)
    # Sorting alphabetically
    sorted_entries=list_format_column.apply(lambda locations: sorted(locations))
    
    return sorted_entries

def list_to_text(column_list):
    # Transform each element(list of words) of column_list to a #string
    return column_list.apply(lambda entry: '; '.join(entry))
    

def unique_labels(column_list):
    # Sorted list of unique labels
    all_locations=[]
    for entry in column_list:
        for item in entry:
            all_locations.append(item)
    return sorted(list(set(all_locations)))

########################################################################################################################

#  Text processing functions

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words



def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^A-Za-z]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_uninformative_words(words):
    """Remove uninformative words from list of tokenized words"""
    
    uninformative_words=['unspecified','other','unknown','source']
    new_words = []
    for word in words:
        if word not in uninformative_words:
            new_words.append(word)
    return new_words

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_stopwords(words)
    words = remove_punctuation(words)
    words=lemmatize_verbs(words)
    #words=remove_uninformative_words(words)
    return words



########################################################################################################################

# Dictionary that map state name to its abbreviation

state_abbreviations  = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    'Washington DC' : "DC",
    'Guam':'GU',
    'Puerto Rico' : "PR",
    'Republic of Palau' : "PW",
    'Multistate': "MUL_STATE"
}

