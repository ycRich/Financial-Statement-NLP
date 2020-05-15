import nltk
import numpy as np
import pandas as pd
import pprint
from tqdm import tqdm
import re
from nltk.corpus import stopwords
import pickle
import utils


try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')


def get_raw_fillings(tickers, doc_type, cik_lookup, show_example=False):

    sec_api = utils.SecAPI()

    sec_data = {}
    for ticker in tickers:
        sec_data[ticker] = utils.get_sec_data(cik_lookup[ticker], doc_type.lower(), sec_api)

    raw_fillings_by_ticker = {}
    for ticker, data in sec_data.items():
        raw_fillings_by_ticker[ticker] = {}
        for index_url, file_type, file_date in tqdm(data, desc='Downloading {} Fillings'.format(ticker), unit='filling'):
            if (file_type.lower() == doc_type.lower()):
                file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')            
                
                raw_fillings_by_ticker[ticker][file_date] = sec_api.get(file_url)
    
    if show_example:
        print('Example Document:\n\n{}...'.format(next(iter(raw_fillings_by_ticker[tickers[0]].values()))[:1000]))
    
    return  raw_fillings_by_ticker


def extract_filling_document(raw_fillings_by_ticker, show_example=False):
    filling_documents_by_ticker = {}

    for ticker, raw_fillings in raw_fillings_by_ticker.items():
        filling_documents_by_ticker[ticker] = {}
        for file_date, filling in tqdm(raw_fillings.items(), desc='Getting Documents from {} Fillings'.format(ticker), unit='filling'):
            filling_documents_by_ticker[ticker][file_date] = utils.get_documents(filling)
    
    if show_example:
        print('\n\n'.join([
            'Document {} Filed on {}:\n{}...'.format(doc_i, file_date, doc[:200])
            for file_date, docs in filling_documents_by_ticker[ticker].items()
            for doc_i, doc in enumerate(docs)][:3]))

    return filling_documents_by_ticker


def preprocess_docs(filling_documents_by_ticker, doc_type, cik_lookup, lemma_english_stopwords):
    docs_by_ticker = {}
    word_pattern = re.compile(r'\w+')

    for ticker, filling_documents in filling_documents_by_ticker.items():
        
        docs_by_ticker[ticker] = []
        for file_date, documents in tqdm(filling_documents.items(), desc='Processing {} {}'.format(ticker, doc_type), unit=doc_type):
            for document in documents:
                if utils.get_document_type(document) == doc_type.lower():
                    # remove html tags
                    file_clean = utils.clean_text(document)
                    # lemmatize words
                    file_lemma = utils.lemmatize_words(word_pattern.findall(file_clean))
                    # remove stopwords
                    file_lemma = [word for word in file_lemma if word not in lemma_english_stopwords]

                    docs_by_ticker[ticker].append({
                        'cik': cik_lookup[ticker],
                        'file': document,
                        'file_date': file_date,
                        'file_clean': file_clean,
                        'file_lemma': file_lemma})
    return docs_by_ticker


def get_sentiment_bow(docs_by_ticker, sentiment_df, sentiments):
    sentiment_bow_by_ticker = {}

    for ticker, docs in docs_by_ticker.items():
        lemma_docs = [' '.join(doc['file_lemma']) for doc in docs]
        
        sentiment_bow_by_ticker[ticker] = {
            sentiment: utils.get_bag_of_words(sentiment_df[sentiment_df[sentiment]]['word'], lemma_docs)
            for sentiment in tqdm(sentiments, desc='Getting Sentiment BOW {}'.format(ticker), unit='sentiment')}

    return sentiment_bow_by_ticker


def get_sentiment_count(sentiment_bow_by_ticker):
    sentiment_count_by_ticker = {}
    for ticker, sentiment_bow in sentiment_bow_by_ticker.items():
        sentiment_count_by_ticker[ticker] = pd.DataFrame({
            sentiment: bow.sum(axis=1)
            for sentiment, bow in sentiment_bow.items()
        })

    return sentiment_count_by_ticker


# Get dates for the universe
def get_docs_jaccard_similarities(sentiment_bow):

    jaccard_similarities = {
        ticker: {
            sentiment_name: utils.get_jaccard_similarity(sentiment_values)
            for sentiment_name, sentiment_values in tqdm(docs_sentiments.items(), desc='Computing Similarity {}'.format(ticker), unit='sentiment')}
        for ticker, docs_sentiments in sentiment_bow.items()}
    
    return jaccard_similarities


if __name__ == "__main__":
    tickers = input('\nPlease enter tickers separated by space:\n').split()
    doc_type = input('\nPlease specify document type for analysis (e.g., 10-K, 10-Q):\n')

    # load cik_lookup dictionary and stopswords
    cik_lookup = utils.load_cik_lookup()
    lemma_english_stopwords = utils.lemmatize_words(stopwords.words('english'))

    # load sentiment_df
    sentiment_df = utils.get_sentiment_df()
    sentiments = [s for s in sentiment_df.columns if s.lower() != 'word']

    # get raw_fillings
    raw_fillings = get_raw_fillings(tickers, doc_type, cik_lookup)

    # extract documents from fillings
    filling_documents_by_ticker = extract_filling_document(raw_fillings)

    # preprocess documents to remove tags, lemmatize and remove stopwords
    docs_by_ticker = preprocess_docs(filling_documents_by_ticker, doc_type, cik_lookup, lemma_english_stopwords)
    # utils.print_ten_k_data(docs_by_ticker[tickers[0]][:5], ['cik', 'file_lemma', 'file_date'])

    # get bag of words for each document
    sentiment_bow = get_sentiment_bow(docs_by_ticker, sentiment_df, sentiments)
    sentiment_count = get_sentiment_count(sentiment_bow)

    jaccard_similarities = get_docs_jaccard_similarities(sentiment_bow)

    file_dates = {
        ticker: [pd.to_datetime(doc['file_date']) for doc in docs]
        for ticker, docs in docs_by_ticker.items()} 

    utils.plot_similarities(
        [jaccard_similarities[tickers[0]][sentiment] for sentiment in sentiments],
        file_dates[tickers[0]][0:-1],
        'Jaccard Similarities for {} Sentiment'.format(tickers[0]),
        sentiments)

    # wordcloud
    available_dates = ' | '.join([date.strftime('%Y-%m-%d') for date in file_dates[tickers[0]]])
    selected_date = input('\nPlease select a file date to show wordcloud:\n{}\n'.format(available_dates))
    selected_doc = [data['file_lemma'] for data in docs_by_ticker[tickers[0]] if data['file_date'] == selected_date]
    text = ' '.join([word for word in selected_doc[0] if word in sentiment_df['word'].values])
    utils.plot_wordcloud(text, lemma_english_stopwords)
    # utils.plot_sentiment_count(sentiment_count, file_dates[tickers[0]][:-1])