import pandas as pd
import sqlalchemy
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
engine = sqlalchemy.create_engine("mysql+pymysql://root:StreamDeck11958$@127.0.0.1:3306/book_db")

title = input("Enter the title: ")

df = pd.read_sql_table('book_metadata',engine, columns=['title'])
# print(df.head())

uniques = df['title'].unique()

corpus = (". ".join(uniques))

# print("UNFILTERED:",corpus)

stop_words = stopwords.words('english')
stop_words.extend(['.', ',', ':', ';', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'vol', 'i', "'", '(', ')', '{', '}', '[', ']', '!', '$', '_', "'S"])

stop_words = set(stop_words)
  
word_tokens = word_tokenize(corpus)
  
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
# filtered_sentence = (' ').join(filtered_sentence)

# print("FILTERED:",filtered_sentence)

def CountFrequency(my_list):
 
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
 
    # for key, value in freq.items():
        # print (f"{key} : {value}")
    return freq

result = CountFrequency(filtered_sentence)

sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

# print(sorted_result.keys())

freq_title = {}
for word in title.split():
    # print(word.upper())
    if word.upper() in sorted_result.keys():
        freq_title[word] = sorted_result[word.upper()]
    
sorted_freq_title = dict(sorted(freq_title.items(), key=lambda item: item[1], reverse=True))

print("TITLE: ", title)
print()

print("****************TOP 3 WORDS FROM YOUR TITLE BY FREQUENCY****************")
first2pairs = {k: sorted_freq_title[k] for k in list(sorted_freq_title)[:3]}
for key, value in first2pairs.items():
    print (f"{key}: {value}")

print()

print("****************TOP 3 WORDS IN DATABASE BY FREQUENCY****************")
first2pairs = {k: sorted_result[k] for k in list(sorted_result)[1:4]}
for key, value in first2pairs.items():
    print (f"{key} : {value}")

