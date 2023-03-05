from pprint import pprint
import requests
from datetime import datetime
import time
import pandas as pd
from pandas import json_normalize
import json
import urllib.request
import sqlalchemy
import pymysql
# from tqdm import tqdm
from datetime import date, datetime, timedelta

def datespan(startDate, endDate, delta=timedelta(days=1)):
    currentDate = startDate
    while currentDate < endDate:
        yield currentDate
        currentDate += delta

BOOKS_ROOT = "https://api.nytimes.com/svc/books/v3/lists/overview.json"
API_KEY = "wnmbxexUpq5cbNvUzfTot1yrvRSXkxyc"

def execute(date, API_KEY):
  requestUrl = f"https://api.nytimes.com/svc/books/v3/lists/full-overview.json?published_date={date}&api-key={API_KEY}"
  requestHeaders = {
    "Accept": "application/json"
  }

  response = requests.get(requestUrl, headers=requestHeaders)

  return response

def parse_booklist(book_list):
    books = []
    for i in book_list:
        dic = {'published_date': i['created_date'],
               'title': i['title'].encode("utf8"),
               'description': i['description'].encode("utf8"),
               'book_image': i['book_image'].encode("utf8") if i['book_image'] != None else '',
               'amazon_product_url': i['amazon_product_url'].encode("utf8"),
               'age_group': i['age_group'],
               'author': i['author'].encode("utf8"),
               'book_review_link': i['book_review_link'].encode("utf8"),
               'contributor': i['contributor'].encode("utf8"),
               'first_chapter_link': i['first_chapter_link'].encode("utf8"),
               'price': i['price'],
               'primary_isbn10': i['primary_isbn10'].encode("utf8"),
               'primary_isbn13': i['primary_isbn13'].encode("utf8"),
               'publisher': i['publisher'].encode("utf8"),
               'sunday_review_link': i['sunday_review_link'].encode("utf8"),
               'weeks_on_list': i['weeks_on_list']
               }
        books.append(dic)
    return books


def parse_response(response):
    response = response.json()

    resultant_bookList = []
    if len(response['results']) < 1:
        return -1
    bestseller_date = response['results'].get('bestsellers_date', None)

    for list_ in response['results']['lists']:
        header_dict = {'list_name': list_['list_name'],
                       'list_name_encoded': list_['list_name_encoded'],
                       'display_name': list_['display_name']}

        books_list = parse_booklist(list_['books'])

        for book in books_list:
            book['bestseller_date'] = bestseller_date
            for k, v in header_dict.items():
                book[k] = v
            resultant_bookList.append(book)
    return resultant_bookList
    
def retrieve_images(booklist):
  x = 1
  for book in booklist:
    if type(book['book_image']) != str:
              url = book['book_image'].decode('UTF-8')  
              urllib.request.urlretrieve(url, f"/home/ec2-user/site/static/Book{x}.jpg")
    x = x+1
    
def create_df(booklist):
  published_date=[]
  title=[]
  description=[]
  book_image=[]
  amazon_product_url=[]
  age_group=[]
  author=[]
  book_review_link=[]
  contributor=[]
  first_chapter_link=[]
  price=[]
  primary_isbn10=[]
  primary_isbn13=[]
  publisher=[]
  sunday_review_link=[]
  weeks_on_list=[]
  bestseller_date=[]
  list_name=[]
  list_name_encoded=[]
  display_name=[]
  
  for book in booklist:
    published_date.append(book['published_date'])
    title.append(book['title'].decode('UTF-8'))
    description.append(book['description'].decode('UTF-8'))
    book_image.append(book['book_image'].decode('UTF-8'))
    amazon_product_url.append(book['amazon_product_url'].decode('UTF-8'))
    age_group.append(book['age_group'])
    author.append(book['author'].decode('UTF-8'))
    book_review_link.append(book['book_review_link'].decode('UTF-8'))
    contributor.append(book['contributor'].decode('UTF-8'))
    first_chapter_link.append(book['first_chapter_link'].decode('UTF-8'))
    price.append(book['price'])
    primary_isbn10.append(book['primary_isbn10'].decode('UTF-8'))
    primary_isbn13.append(book['primary_isbn13'].decode('UTF-8'))
    publisher.append(book['publisher'].decode('UTF-8'))
    sunday_review_link.append(book['sunday_review_link'].decode('UTF-8'))
    weeks_on_list.append(book['weeks_on_list'])
    bestseller_date.append(book['published_date'])
    list_name.append(book['list_name'])
    list_name_encoded.append(book['list_name_encoded'])
    display_name.append(book['display_name'])
    
  books_dic = {
    'published_date': published_date,
    'title': title,
    'description': description,
    'book_image': book_image,
    'amazon_product_url': amazon_product_url,
    'age_group': age_group,
    'author': author,
    'book_review_link': book_review_link,
    'contributor': contributor,
    'first_chapter_link': first_chapter_link,
    'price': price,
    'primary_isbn10': primary_isbn10,
    'primary_isbn13': primary_isbn13,
    'publisher': publisher,
    'sunday_review_link': sunday_review_link,
    'weeks_on_list': weeks_on_list,
    'bestseller_date': bestseller_date,
    'list_name': list_name,
    'list_name_encoded': list_name_encoded,
    'display_name': display_name
  }
  
  df = pd.DataFrame(books_dic)
  df.to_csv('bestseller.csv', index=False)
  return df
  
def add_to_db(df, date):
  engine = sqlalchemy.create_engine("mysql+pymysql://root:StreamDeck7692$@127.0.0.1:3306/book_db")
  dbConnection = engine.connect()
  try:
      df.to_sql('book_metadata', dbConnection, if_exists='append',index=False);
  except ValueError as vx:
      print(vx)
  except Exception as ex:   
      print(ex)
  else:
      # datetime object containing current date and time
      now = datetime.now()
      # mm/dd/YY H:M:S
      dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

      df2 = pd.read_sql_table('book_metadata',engine, columns=['title'])

      print(f"Old Books from {date} Added To Table On: ", dt_string, " Total Number of Titles In System: ", df2.shape[0]);   
  finally:
      dbConnection.close()

  
# function calls
for day in datespan(date(2020, 9, 9), date(2022, 6, 29), delta=timedelta(weeks=1)):
    date = day.strftime("%Y-%m-%d")
    response = execute(date, API_KEY)
    books = parse_response(response)
    retrieve_images(books)
    df = create_df(books)
    add_to_db(df, date)
