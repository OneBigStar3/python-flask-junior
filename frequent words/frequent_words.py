import nltk
import pandas as pd
import numpy as np
from PIL import Image
import sqlalchemy
engine = sqlalchemy.create_engine("mysql+pymysql://root:StreamDeck1!@127.0.0.1:3306/book_db")

from datetime import datetime

from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_sql_table('book_metadata',engine, columns=['published_date','title'])
df['published_date'] = pd.to_datetime(df['published_date'])
start_date = min(df['published_date'])

def get_corpus(since_date: datetime)->str:
    corpus = '. '.join(
        df.loc[(df['published_date'] > since_date) & (df['title'].notna())]['title'].to_list())
    return corpus

def get_word_freq_from_corpus(corpus:str):
    allWords = nltk.tokenize.word_tokenize(corpus)
    #print(allWords)
    allWordDist = nltk.FreqDist(w.lower() for w in allWords)

    stopwords = nltk.corpus.stopwords.words('english') + ['the','and','with','not','this','that','there','here','a','an','all']
    stopwords.extend(['.', ',', ':', ';', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'vol', 'i', "'", '(', ')', '{', '}', '[', ']', '!', '$', '_', "'S"])
    
    #print(stopwords)
    
    allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w.lower() not in stopwords and len(w) > 2)
    
    print(allWordExceptStopDist)

    return allWordExceptStopDist


def get_frequencies_since(since_date:datetime):
    return get_word_freq_from_corpus(get_corpus(since_date))

def get_frequencies_word_cloud_since(since_date:datetime, fig_size=(24,18), dpi=100):
    freqs = get_frequencies_since(since_date)
    mask = np.array(Image.open("/home/ec2-user/site/frequent words/color11.png"))
    image_colors = ImageColorGenerator(mask)
    wordcloud = WordCloud(background_color="white",colormap="tab10").generate_from_frequencies(freqs)
    #wordcloud.recolor(color_func=image_colors)
    plt.figure(figsize=fig_size, dpi=dpi)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('/home/ec2-user/site/static/images/one_year_titles.png', bbox_inches='tight')
    
def get_frequencies_word_cloud_since_start(fig_size=(24,18), dpi=100):
    since_date = start_date
    freqs = get_frequencies_since(since_date)
    mask = np.array(Image.open("/home/ec2-user/site/frequent words/color11.png"))
    image_colors = ImageColorGenerator(mask)
    wordcloud = WordCloud(background_color="white",colormap="tab10").generate_from_frequencies(freqs)
    #wordcloud.recolor(color_func=image_colors)
    plt.figure(figsize=fig_size, dpi=dpi)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('/home/ec2-user/site/static/images/all_titles.png', bbox_inches='tight')

def get_all_since(since_date:datetime, fig_size=(24,18), dpi=100):
    freqs = get_frequencies_since(since_date)
    wordcloud = WordCloud().generate_from_frequencies(freqs)
    plt.figure(figsize=fig_size, dpi=dpi)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('one_year_titles.png')

    return freqs


