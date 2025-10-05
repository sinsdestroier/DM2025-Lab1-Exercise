# helpers/data_mining_helpers.py

import pandas as pd  # 新增：用來辨識 DataFrame / Series
import nltk
import numpy as np
"""
Helper functions for data mining lab session 2018 Fall Semester
Author: Elvis Saravia
Email: ellfae@gmail.com
"""

import pandas as pd 

def format_rows(docs):
    """Format the text field and strip special characters
    Args:
        docs: pandas DataFrame or object with data attribute
    Returns:
        list: List of formatted text strings
    """
    D = []
    
    # Handle pandas DataFrame
    if isinstance(docs, pd.DataFrame):
        if 'text' not in docs.columns:
            raise ValueError("DataFrame must contain 'text' column")
        
        for text in docs['text']:
            if pd.isna(text):
                continue
            temp_d = " ".join(str(text).split("\n")).strip()
            D.append([temp_d])
            
    # Handle original input format
    elif hasattr(docs, 'data'):
        for d in docs.data:
            temp_d = " ".join(d.split("\n")).strip()
            D.append([temp_d])
    
    else:
        raise ValueError("Input must be DataFrame with 'text' column or have 'data' attribute")
        
    return D

def format_labels(target, docs):
    """ format the labels（維持原樣；供 twenty_train 使用）"""
    return docs.target_names[target]

def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter += 1
    return ("The amoung of missing records is: ", counter)

def tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using the nltk library
    """
    tokens = []
    for d in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(d, language='english'):
            tokens.append(word)
    return tokens
