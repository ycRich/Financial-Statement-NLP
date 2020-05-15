import matplotlib.pyplot as plt
import requests
from ratelimit import limits, sleep_and_retry
import re
import pandas as pd
import os
from bs4 import BeautifulSoup
from datetime import date
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import pickle
import numpy as np
from wordcloud import WordCloud, STOPWORDS 


class SecAPI(object):
    SEC_CALL_LIMIT = {'calls': 10, 'seconds': 1}

    @staticmethod
    @sleep_and_retry
    # Dividing the call limit by half to avoid coming close to the limit
    @limits(calls=SEC_CALL_LIMIT['calls'] / 2, period=SEC_CALL_LIMIT['seconds'])
    def _call_sec(url):
        return requests.get(url)

    def get(self, url):
        return self._call_sec(url).text


def print_ten_k_data(ten_k_data, fields, field_length_limit=50):
    indentation = '  '

    print('[')
    for ten_k in ten_k_data:
        print_statement = '{}{{'.format(indentation)
        for field in fields:
            value = str(ten_k[field])

            # Show return lines in output
            if isinstance(value, str):
                value_str = '\'{}\''.format(value.replace('\n', '\\n'))
            else:
                value_str = str(value)

            # Cut off the string if it gets too long
            if len(value_str) > field_length_limit:
                value_str = value_str[:field_length_limit] + '...'

            print_statement += '\n{}{}: {}'.format(indentation * 2, field, value_str)

        print_statement += '},'
        print(print_statement)
    print(']')


def plot_similarities(similarities_list, dates, title, labels):
    assert len(similarities_list) == len(labels)

    plt.figure(1, figsize=(10, 7))
    for similarities, label in zip(similarities_list, labels):
        plt.title(title)
        plt.plot(dates, similarities, label=label)
        plt.legend()
        plt.xticks(rotation=90)

    plt.show()


def plot_sentiment_count(sentiment_count, dates):
    sentiment_count['date'] = dates
    sentiment_count.set_index('date', inplace=True)
    sentiment_count.div(sentiment_count.sum()).plot.bar()

    plt.show()


def plot_wordcloud(text, stopwords):
    wc = WordCloud(stopwords=stopwords)
    wc.generate(text)

    plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


def get_sec_data(cik, doc_type, sec_api, start=0, count=60):

    newest_pricing_data = date.today()
    rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
        '&CIK={}&type={}&start={}&count={}&owner=exclude&output=atom' \
        .format(cik, doc_type, start, count)
    sec_data = sec_api.get(rss_url)
    feed = BeautifulSoup(sec_data.encode('ascii'), 'xml').feed
    entries = [
        (
            entry.content.find('filing-href').getText(),
            entry.content.find('filing-type').getText(),
            entry.content.find('filing-date').getText())
        for entry in feed.find_all('entry', recursive=False)
        if pd.to_datetime(entry.content.find('filing-date').getText()) <= newest_pricing_data]

    return entries


def load_cik_lookup():
    try:
        with open('cik_lookup', 'rb') as file:
            cik_lookup = pickle.load(file)
    except:
        cwd = os.getcwd()
        df = pd.read_csv(os.path.join(cwd, 'cik_ticker.csv'), sep='|', dtype=str)
        df['CIK'] = df['CIK'].str.zfill(10)
        cik_lookup = {row['Ticker']: row['CIK'] for i, row in df.iterrows()}

        with open('cik_lookup', 'wb') as file:
            pickle.dump(cik_lookup, file)
    return cik_lookup


def get_documents(text):
    """
    Extract the documents from the text

    Parameters
    ----------
    text : str
        The text with the document strings inside

    Returns
    -------
    extracted_docs : list of str
        The document strings found in `text`
    """
    
    regex = re.compile(r'<DOCUMENT>((.|\n)*?)<\/DOCUMENT>')
    matches = regex.finditer(text)
    
    return [match[1] for match in matches]


def get_document_type(doc):
    """
    Return the document type lowercased

    Parameters
    ----------
    doc : str
        The document string

    Returns
    -------
    doc_type : str
        The document type lowercased
    """
    
    regex = re.compile(r'<TYPE>.*')
    matches = regex.finditer(doc)
    doc_type = [match[0].replace(r'<TYPE>','').lower() for match in matches][0]
    return doc_type


def clean_text(text):
    text = text.lower()
    text = BeautifulSoup(text, 'html.parser').get_text()
    return text


def lemmatize_words(words):
    """
    Lemmatize words 

    Parameters
    ----------
    words : list of str
        List of words

    Returns
    -------
    lemmatized_words : list of str
        List of lemmatized words
    """
    
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(word, pos='v') for word in words]


def get_sentiment_df():
    try:
        with open('sentiment_df', 'rb') as file:
            sentiment_df = pickle.load(file)
    except:
        sentiments = ['negative', 'positive', 'uncertainty', 'litigious', 'constraining', 'interesting']

        sentiment_df = pd.read_csv(os.path.join(os.getcwd(), 'LoughranMcDonald_MasterDictionary_2018.csv'))
        sentiment_df.columns = [column.lower() for column in sentiment_df.columns] # Lowercase the columns for ease of use

        # Remove unused information
        sentiment_df = sentiment_df[sentiments + ['word']]
        sentiment_df[sentiments] = sentiment_df[sentiments].astype(bool)
        sentiment_df = sentiment_df[(sentiment_df[sentiments]).any(1)]

        # Apply the same preprocessing to these words as the 10-k words
        sentiment_df['word'] = lemmatize_words(sentiment_df['word'].str.lower())
        sentiment_df = sentiment_df.drop_duplicates('word')

        with open('sentiment_df', 'wb') as file:
            pickle.dump(sentiment_df, file)
    return sentiment_df


def get_bag_of_words(sentiment_words, docs):
    """
    Generate a bag of words from documents for a certain sentiment

    Parameters
    ----------
    sentiment_words: Pandas Series
        Words that signify a certain sentiment
    docs : list of str
        List of documents used to generate bag of words

    Returns
    -------
    bag_of_words : 2-d Numpy Ndarray of int
        Bag of words sentiment for each document
        The first dimension is the document.
        The second dimension is the word.
    """
    
    bag_of_words = np.zeros((len(docs), len(sentiment_words)), dtype=np.int)
    # print(sentiment_words)
    for i, doc in enumerate(docs):
        for word in doc.split():
            if word in sentiment_words.values:
                bag_of_words[i, sentiment_words.index.get_loc(sentiment_words[sentiment_words==word].index[0])] += 1
    return bag_of_words


def get_jaccard_similarity(bag_of_words_matrix):
    """
    Get jaccard similarities for neighboring documents

    Parameters
    ----------
    bag_of_words : 2-d Numpy Ndarray of int
        Bag of words sentiment for each document
        The first dimension is the document.
        The second dimension is the word.

    Returns
    -------
    jaccard_similarities : list of float
        Jaccard similarities for neighboring documents
    """

    bag_of_words_matrix = bag_of_words_matrix.astype(bool)
    jaccard_similarities = []
    for i in range(bag_of_words_matrix.shape[0]-1):
        jaccard_similarities.append(jaccard_score(bag_of_words_matrix[i,:], bag_of_words_matrix[i+1,:]))
    return jaccard_similarities