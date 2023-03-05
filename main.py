from flask import Flask, render_template, request, redirect, url_for, session, make_response, send_from_directory, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
s_words=set(stopwords.words('english'))
#Logging
import logging

from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from flask_mysqldb import MySQL
from flask_mail import Mail, Message
import MySQLdb.cursors, re, uuid, hashlib, datetime, os
# Start Of Tagging Modules
import spacy
import yake
import uuid
import emotions
from bookkeywords.bookwords import list_keywords

import book_cover.backHandler as BH
from book_cover.backHandler import get_heatmap
cover_compare = BH.BackendMobileNetv3()


from rake_nltk import Rake
from collections import Counter
from string import punctuation

import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy

import json

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Import Module With Name List As Python List
import names_list
# Import TextGenie For Rephrasing Using Spacy
# from textgenie import TextGenie

# import the Machine Learning Related Libraries
import torch
from nltk.tokenize import sent_tokenize
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
import sentencepiece
import pickle

from fastai.text.all import *
# from transformers import *

from blurr.text.data.all import *
from blurr.text.modeling.all import *

import nltk
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon')
# Import SYS To Add External Directories To Python PATH
import sys
# Import ReCaptcha
from flask_recaptcha import ReCaptcha
#from rephraser.parpaphrase_server import BackendT5

# loading Environemnt .env file
from dotenv import load_dotenv
load_dotenv()

# remove warnings
import warnings
warnings.warn('ignore')

# Import Dropzone Requirements
import os
from flask_dropzone import Dropzone
basedir = os.path.abspath(os.path.dirname(__file__))
# backT5 = BackendT5()
import requests
from datetime import date
import datetime

sys.path.append('/home/ec2-user/rephrase')
sys.path.append('/home/ec2-user/Book')
sys.path.append('/home/ec2-user/.local/lib/python3.8/site_packages')

#from rephrase import changer

# Import LIME Explainer
from lime_explainer import explainer, tokenizer, METHODS
from text2emotion import get_emotion

# Iport Email Sending Packages
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# spinrewriter Package
from spinrewriterapi import SpinRewriterAPI

#Logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

app = Flask(__name__, static_folder='static')

#Logging
logging.basicConfig(level=logging.DEBUG,
                    filename="app.log",
                    format='%(asctime)s %(message)s',
                    handlers=[logging.StreamHandler()])

import spacy
import mysql.connector
import pandas as pd
# thread Package
from threading import Thread


import spacy
from whoosh.fields import Schema, TEXT, ID
from whoosh import index as ind__
import os, os.path
from whoosh import qparser
import mysql.connector
from nltk.tokenize import word_tokenize


sp = spacy.load('en_core_web_lg')
all_stopwords = sp.Defaults.stop_words



nlp__ = spacy.load('en_core_web_lg',exclude=["tagger", "parser", "senter", "attribute_ruler", "lemmatizer", "ner"])


#cache_buster.init_app(app)
# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'Skittles'

# App Settings
app.config['threaded'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Date Time format
dt_format = "%Y-%m-%d %H:%M:%S"

# invoice Date time format
idt_format = "%Y%m%d%H%M%S"

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_PASSWORD'] = 'StreamDeck7692$'
app.config['MYSQL_DB'] = 'pythonlogin_advanced'

# Enter your email server details below, the following details uses the gmail smtp server (requires gmail account)
#app.config['MAIL_SERVER']= int(os.getenv('MAIL_SERVER'))
#app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT'))
#app.config['MAIL_USERNAME'] = int(os.getenv('MAIL_USERNAME'))
#app.config['MAIL_PASSWORD'] = int(os.getenv('MAIL_PASSWORD'))
#app.config['MAIL_USE_TLS'] = True
#app.config['MAIL_USE_SSL'] = False

# Enter your domain name below
app.config['DOMAIN'] = os.getenv('DOMAIN')

# ReCaptcha Configuration
app.config['RECAPTCHA_TYPE'] = 'image'
app.config['RECAPTCHA_ENABLED'] = os.getenv('RECAPTCHA_ENABLED')
app.config['RECAPTCHA_SITE_KEY'] = os.getenv('RECAPTCHA_SITE_KEY')
app.config['RECAPTCHA_SECRET_KEY'] = os.getenv('RECAPTCHA_SECRET_KEY')
recaptcha = ReCaptcha(app)

# PayPal SDK setting
app.config['PAYPAL_CLIENT_ID'] = os.getenv('PAYPAL_CLIENT_ID')
app.config['PAYPAL_CLIENT_SECRET'] = os.getenv('PAYPAL_CLIENT_SECRET')


# Create a secure SSL Context
context = ssl.create_default_context()


# DropZone Configuration
save_dropped = os.path.join(basedir) + '/static'
app.config.update(
    UPLOADED_PATH= os.path.join(save_dropped, 'uploads/cover_uploads/'),
    DROPZONE_MAX_FILE_SIZE = 20,
    DROPZONE_ALLOWED_FILE_TYPE = 'image',
    DROPZONE_MAX_FILES = 1,
    DROPZONE_DEFAULT_MESSAGE = 'Dragging You Cover Here Automaticallly Uploads And Starts Cover Compare.',
    DROPZONE_MAX_FILE_EXCEED = 'You can upload one file at a time. Refresh to upload another file.',
    DROPZONE_TIMEOUT = 5*60*1000)

main_array = []

# Intialize MySQL
mysql = MySQL(app)

# Initialize Smtp
mail = Mail(app)

# Create SQLAlchemy Engine to access the database
engine = sqlalchemy.create_engine("mysql+pymysql://root:StreamDeck7692$@127.0.0.1:3306/book_db")

def read_input():

    df = pd.read_sql_query('SELECT DISTINCT title , primary_isbn13 , description  FROM book_metadata;',con=engine)
    return df.values.tolist()

def read_input_for_title_ranking(algo):

    df = ''
    if (algo == "1"):
        df = pd.read_sql_query('SELECT DISTINCT   title,primary_isbn13,bestseller_date FROM book_metadata;',con=engine)
    else:
        df = pd.read_sql_query('SELECT title , primary_isbn13,bestseller_date FROM book_metadata;',con=engine)

    return df.values.tolist()

def multi_thread_similar(description,res):
    global main_array
    res_ = []
    sv_res = []
    for r in res:
        res_.append(str(r[2]))
    processed_docs_1 = nlp__.pipe(res_,disable=[ "parser"])
    processed_docs_2 = nlp__(description)
    print("start thraead....")
    for i in range(len(res_)):

        s_1 = next(processed_docs_1)
        s = s_1.similarity(processed_docs_2)
        c = [str(round(float(s * 100),1)),str(res[i][0]),str(res[i][1]),str(res[i][2])]
        main_array.append(c)

    print("Stop thread...")



def search_for_similar(description):
    global main_array
    main_array.clear()
    res = read_input()
    sv_res = []
    print("data_len: "+str(len(res)))
    first_half = res[:int(len(res) / 2)]
    second_half = res[int(len(res) / 2):]
    threads = []
    print("Start ...........")
    thread = Thread(target=multi_thread_similar,args=(description,first_half,))
    thread.start()

    thread_1 = Thread(target=multi_thread_similar,args=(description,second_half,))
    thread_1.start()


    thread.join()
    thread_1.join()

    print("stop ....................")

    df = pd.DataFrame(list(map(list,set(map(tuple,main_array)))),columns=["percentage","title","isbn","description"])
    df = df.drop_duplicates(subset=['description'])
    cc = df.sort_values(["percentage"], ascending=False).head(4)

    return cc.values.tolist()

def remove_stop_words(example_sent):
    word_tokens = word_tokenize(example_sent)

    filtered_sentence = [w for w in word_tokens if not w.lower() in s_words]

    filtered_sentence = []

    for w in word_tokens:
        if w not in s_words:
            filtered_sentence.append(w)


    return filtered_sentence

def index_search(schema,dirname, search_fields, search_query):

    ix = ind__.open_dir(dirname)
    schema = ix.schema
    og = qparser.OrGroup.factory(0.9)
    mp = qparser.MultifieldParser(search_fields, schema, group = og)


    q = mp.parse(search_query)

    r_list = []
    with ix.searcher() as s:
        results = s.search(q, terms=True)
        print("Search Results: ")


        for r in results:
            acc = round(r.score,2)
            acc = str(acc)
            id_ = str(r['path'])

            matches = []
            for c in r.matched_terms():
                matches.append(str(list(c)[1],'utf-8').lower())
            ls_ = [acc,id_,matches]
            r_list.append(ls_)
    return r_list

def get_similar_words(input_):

    dsc_ = read_input()
    schema = Schema( path=ID(stored=True), content=TEXT(stored = True))

    if not os.path.exists("index_dir"):
        os.mkdir("index_dir")

    ix = ind__.create_in("index_dir", schema)
    writer = ix.writer()
    for i in range(len(dsc_)):
        writer.add_document( content=str(dsc_[i][2]),
                        path=str(i))
    writer.commit()


    text_tokens = word_tokenize(input_)
    tokens_without_sw= [word for word in text_tokens if not word in all_stopwords]
    c = ' '.join(tokens_without_sw)
    print(c)
    results_dict = index_search(schema,"index_dir", ['title','content'], c)

    r_list = []
    if (len(results_dict) >= 4):
        for i in range(len(results_dict)):
            id_ = results_dict[i][1]

            acc = results_dict[i][0]
            title = list(dsc_[int(id_)])[0]
            isbn = list(dsc_[int(id_)])[1]
            description = list(dsc_[int(id_)])[2]


            matches = results_dict[i][2]

            out_ = ""
            for d in str(description).lower().split(" "):
                if ('-' not in d.lower()):
                    if (d.lower().replace("-","").replace("'s","").replace(',','').replace('"',"").replace("'","").replace("'s","").replace('.','') in matches):
                        out_ += "<strong>"+d+"</strong> "
                    else:
                        out_ += d + " "
                else:
                    for a in d.split('-'):
                        if (a.lower().replace("-","").replace("'s","").replace(',','').replace('"',"").replace("'","").replace("'s","").replace('.','') in matches):
                            out_ += "<strong>"+a+"</strong> "
                        else:
                            out_ += a + " "

            r_list.append([str(acc),str(title),str(isbn),str(out_)])
        df = pd.DataFrame(r_list,columns=['Percentage','title','isbn','descriptions'])
        df = df.drop_duplicates(subset=['descriptions']).head(4)

    return df.values.tolist()

def get_similar_ranking(description,res):

    main_array = []

    processed_docs_1 = nlp__.pipe(res)
    processed_docs_2 = nlp__(description)

    for i in range(len(res)):

        s_1 = next(processed_docs_1)
        s = s_1.similarity(processed_docs_2)
        if (round(float(s),2) > round(float(0.8),2) and round(float(s),2) < round(float(1.0),2)):

            c = [str(s),str(res[i])]
            main_array.append(c)

    df = pd.DataFrame(main_array,columns=['acc','word'])
    df = df.sort_values(["acc"], ascending=False).head(3)
    df = df.drop_duplicates(subset=['word'])
    return df.values.tolist()



def CountFrequency_(my_list):
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    return Counter(my_list)


def apply_top_ten(title,sorted_result,start,end):

    final_result = {}
    sorted_result = {k: sorted_result[k] for k in list(sorted_result)[1:]}
    counter = 0
    check = 0
    for key, value in sorted_result.items():
        for title_key in title.split():
            if (key != title_key.upper()):
                check = check + 1
        if (counter<10) and (check == len(title.split())) and (ord(key[0])>=65 and ord(key[0])<=90):
            counter = counter + 1
            final_result[key] = value
            print (f"{key} : {value}")
        elif counter == 10:
            break
        check = 0
    print()

    title_words = list(final_result.keys())
    title_words_freq = list(final_result.values())

    plt.rc('xtick', labelsize=6)
    end_ = str(end).split('-')[0]
    if ('2024' in end):
        end_ = "2022"
    plt.bar(title_words, title_words_freq,align='center',color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    plt.title("WORDS IN DATABASE WITH FREQUENCY (" + str(end_) + "-"+str(start).split('-')[0] + " )")

    strFile = "static/images/title_rank_plot.png"
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.close()


def apply_top_ten_for_year(title,res,start,end):
    if (os.path.exists("static/images/"+str(start).split('-')[0] + "-"+str(end).split('-')[0]+".png")):
        return
    df = pd.DataFrame(list(res),columns=['title','primary_isbn13','bestseller_date'])
    df['bestseller_date'] = pd.to_datetime(df['bestseller_date'],format='%Y-%m-%d')
    mask = (df['bestseller_date'] > start) & (df['bestseller_date'] <= end)
    df = df.loc[mask]
    uniques = df['title']
    corpus = (". ".join(uniques))

    stop_words = stopwords.words('english')
    stop_words.extend(['.', ',', ':', ';', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'vol', 'i', "\'RE", " ", '(', ')', '{', '}', '[', ']', '!', '$', '_', "\'S", "\'"])
    stop_words = set(stop_words)


    word_tokens = word_tokenize(corpus)
    text_tokens = word_tokenize(title)

    all_stopwords.add("this")
    all_stopwords.add("and")

    filtered_words = [w for w in word_tokens if not w.lower() in stop_words]


    result = CountFrequency_(filtered_words)
    sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    final_result = {}
    sorted_result = {k: sorted_result[k] for k in list(sorted_result)[1:]}
    counter = 0
    check = 0
    for key, value in sorted_result.items():
        for title_key in title.split():
            if (key != title_key.upper()):
                check = check + 1
        if (counter<10) and (check == len(title.split())) and (ord(key[0])>=65 and ord(key[0])<=90):
            counter = counter + 1
            final_result[key] = value
            print (f"{key} : {value}")
        elif counter == 10:
            break
        check = 0
    print()

    title_words = list(final_result.keys())
    title_words_freq = list(final_result.values())
    plt.rc('xtick', labelsize=6)
    plt.bar(title_words, title_words_freq,align='center',color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    plt.title("WORDS IN DATABASE WITH FREQUENCY ("+str(end).split('-')[0] + " - "+str(start).split('-')[0] + ")")

    strFile = "static/images/"+str(start).split('-')[0] + "-"+str(end).split('-')[0]+".png"
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.close()


def get_title_ranking(title,algo,start,end):
    res = read_input_for_title_ranking(algo)
    len_ = len(res)
    print(len_)
    df = pd.DataFrame(list(res),columns=['title','primary_isbn13','bestseller_date'])
    df['bestseller_date'] = pd.to_datetime(df['bestseller_date'],format='%Y-%m-%d')
    mask = (df['bestseller_date'] > start) & (df['bestseller_date'] <= end)
    df = df.loc[mask]
    uniques = df['title']
    corpus = (". ".join(uniques))

    stop_words = stopwords.words('english')
    stop_words.extend(['.', ',', ':', ';', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'vol', 'i', "\'RE", " ", '(', ')', '{', '}', '[', ']', '!', '$', '_', "\'S", "\'"])
    stop_words = set(stop_words)


    word_tokens = word_tokenize(corpus)
    text_tokens = word_tokenize(title)

    all_stopwords.add("this")
    all_stopwords.add("and")




    filtered_words = [w for w in word_tokens if not w.lower() in stop_words]


    result = CountFrequency_(filtered_words)

    sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


    freq_title = {}
    for word in title.split():
        if word.upper() in result.keys():
            freq_title[word] = result[word.upper()]
    sorted_freq_title = dict(sorted(freq_title.items(), key=lambda item: item[1], reverse=True))




    apply_top_ten(title,sorted_result,start,end)

    apply_top_ten_for_year(title,res,"2011-01-01","2022-01-01")
    apply_top_ten_for_year(title,res,"2012-01-01","2022-01-01")
    apply_top_ten_for_year(title,res,"2013-01-01","2022-01-01")
    apply_top_ten_for_year(title,res,"2014-01-01","2022-01-01")
    apply_top_ten_for_year(title,res,"2015-01-01","2022-01-01")
    apply_top_ten_for_year(title,res,"2016-01-01","2022-01-01")
    apply_top_ten_for_year(title,res,"2017-01-01","2022-01-01")
    apply_top_ten_for_year(title,res,"2018-01-01","2022-01-01")
    apply_top_ten_for_year(title,res,"2019-01-01","2022-01-01")
    apply_top_ten_for_year(title,res,"2020-01-01","2022-01-01")
    apply_top_ten_for_year(title,res,"2021-01-01","2022-01-01")

    apply_top_ten_for_year(title,res,"2011-01-01","2012-01-01")
    apply_top_ten_for_year(title,res,"2012-01-01","2013-01-01")
    apply_top_ten_for_year(title,res,"2013-01-01","2014-01-01")
    apply_top_ten_for_year(title,res,"2014-01-01","2015-01-01")
    apply_top_ten_for_year(title,res,"2015-01-01","2016-01-01")
    apply_top_ten_for_year(title,res,"2016-01-01","2017-01-01")
    apply_top_ten_for_year(title,res,"2017-01-01","2018-01-01")
    apply_top_ten_for_year(title,res,"2018-01-01","2019-01-01")
    apply_top_ten_for_year(title,res,"2019-01-01","2020-01-01")
    apply_top_ten_for_year(title,res,"2020-01-01","2021-01-01")


    return sorted_freq_title






# Function to create the Frequency map of the titles
def CountFrequency(my_list):
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    return freq
def is_membership_active():
        # get username from session
        if session.get('membership_expiry', None):
                expire_at = datetime.datetime.strptime(session['membership_expiry'], dt_format)
                if datetime.datetime.now() > expire_at:
                        return False
                else:
                        # need to check timeout
                        if not session.get('next_time_check', None):
                                return False

                        if datetime.datetime.strptime(session['next_time_check'], dt_format) > datetime.datetime.now():
                                        return True
                        else:
                                # check the expiry now
                                cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                                cursor.execute('SELECT * from membership where fk_user_account=%s', (session['id'],))
                                member  = cursor.fetchone()
                                if datetime.datetime.now() > member['expire_at']:
                                        # set the next time interval to 30 min
                                        session['next_time_check'] = (datetime.datetime.now() + datetime.timedelta(minutes=30)).strftime(dt_format)
                                        return True
                                else:
                                        # yes, the expire datetime reached
                                        return False
        else:
                # it seems the cookies not set, so logout
                False

# Check if logged in function, update session if cookie for "remember me" exists
def loggedin():
    # print(f"login check called {session}")
    if 'loggedin' in session:
                # check if membership is active
        if is_membership_active():
            return True
        else:
            return False
    elif 'rememberme' in request.cookies:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # check if remembered, cookie has to match the "rememberme" field
        cursor.execute('SELECT * FROM accounts WHERE rememberme = %s', (request.cookies['rememberme'],))
        account = cursor.fetchone()
        cursor.execute('SELECT * FROM membership WHERE fk_user_account=%s', (account['id'],))
        member = cursor.fetchone()
        if account:
            # update session variables
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            session['role'] = account['role']
            session['membership_expiry'] = member['expire_at'].strftime(dt_format)
            session['next_time_check'] = (datetime.datetime.now() + datetime.timedelta(minutes=30)).strftime(dt_format)
            return True
    # account not logged in return false
    return False

# Intialize Mail
mail = Mail(app)


# Enable account activation?
account_activation_required = True

# Enable CSRF Protection?
csrf_protection = False

# Initialize textginie object
# textgenie = TextGenie(paraphrase_model_name='hetpandya/t5-small-tapaco', mask_model_name='bert-base-uncased',)

# load fine-tuned model:
"""
    def get_model(file):
      return send_from_directory(app.static_folder, file

    with open(get_model('model1.pkl'), 'rb') as file:
      model = pickle.load(file)
"""


# Index page
# http://localhost:5000/
@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def index():
    # Redirect user to home page if logged-in
    if loggedin():
        return redirect(url_for('home'))

    # return Index page
    # planlist list all

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    message = '' # Create empty message

    # Need to fetch planlist for all POST methods also. Earliear its only doing for GET
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * from planlist;')
    planlist = cursor.fetchall()
    settings = get_settings()

    # crawler file
    crawler_file = os.path.join(basedir, 'crawler', 'last_week')

    if os.path.isfile(crawler_file):
        with open(crawler_file) as f:
            crawler_file_data = f.read().strip()

    else:
        crawler_file_data = ""

    if request.method == 'POST':
        if settings['recaptcha']['value'] == 'true':
            if 'g-recaptcha-response' not in request.form:
                message='Invalid captcha!'
                return render_template('index.html', settings=settings, message=message, planlist=planlist)
            req = urllib.request.Request('https://www.google.com/recaptcha/api/siteverify', urllib.parse.urlencode({ 'response': request.form['g-recaptcha-response'], 'secret': settings['recaptcha_secret_key']['value'] }).encode())
            response_json = json.loads(urllib.request.urlopen(req).read().decode())
            if not response_json['success']:
                message='Invalid captcha!'
                return render_template('index.html', settings=settings, message=message, planlist=planlist)
        # If account exists in accounts table in out database
        if recaptcha.verify():
            user_email = request.form['promo']  # user e-mail id
            title = request.form['title']   # user title (to be ranked)
            date_today = date.today()
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            query1 = "SELECT * FROM `front_page_promo` ORDER BY `id` DESC;"
            cursor.execute(query1)
            id_store = cursor.fetchone()
            last_id = int(id_store['id'])
            cursor1 = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            query = "SELECT * FROM `front_page_promo` WHERE `promo_email` = '{0}' ORDER BY `times_used` DESC;".format(user_email)
            cursor1.execute(query)
            promo_users = cursor1.fetchone()
            if promo_users:
                database_last_day = promo_users['promo_date']
                total_times_used = int(promo_users['times_used'])
                user_id = promo_users['id']
                if database_last_day == date_today:
                    message = 'Hello ' + user_email + '! You have already used ranking feature today. Please try again tomorrow.'
                    return render_template('index.html', settings=settings, message=message, planlist=planlist, last_week=crawler_file_data)
                else:

                    total_times_used += 1
                    #time.sleep(15)
                    thread = Thread(target=threaded_task, args=('Title Ranking', title, settings, user_email,))
                    thread.daemon = True
                    thread.start()
                    cursor1.execute("UPDATE `front_page_promo` SET `times_used` = {0}, `promo_date` = '{1}' WHERE id = {2}".format(total_times_used, date_today, user_id))
                    message = 'Thanks for filling out the form!'
                    mysql.connection.commit()
                    return render_template('index.html', settings=settings, message=message, planlist=planlist, last_week=crawler_file_data)
            else:
                total_times_used = 1
                thread = Thread(target=threaded_task, args=('Ranking Results', title, settings, user_email,))
                thread.daemon = True
                thread.start()
                new_id = last_id + 1
                cursor.execute("INSERT INTO `front_page_promo` (id, promo_email, promo_date, times_used) VALUES (%s, %s, %s, %s)",(new_id, user_email, date_today, total_times_used))
                message = 'Thanks for filling out the form!'
                mysql.connection.commit()
                return render_template('index.html', settings=settings, message=message, planlist=planlist, last_week=crawler_file_data)
        else:
            message = 'Please fill out the ReCaptcha!'
            return render_template('index.html', settings=settings, message=message, planlist=planlist, last_week=crawler_file_data)
    else:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * from planlist;')
        planlist = cursor.fetchall()
        token = uuid.uuid4()
        session['token'] = token

        # crawler file
        crawler_file = os.path.join(basedir, 'crawler', 'last_week')

        if os.path.isfile(crawler_file):
            with open(crawler_file) as f:
                crawler_file_data = f.read().strip()

        else:
            crawler_file_data = ""
        # Try to get unique key session token
        if not session.get('pagesession', None):
            # create a session and assign to user
            cursor.execute('SELECT uuid()')
            mysql_uuid = cursor.fetchall()[0]['uuid()']
            print('UUID GENERATED is ', mysql_uuid)
            cursor.execute('INSERT INTO pagevisit(`uid`, `url`, `is_login`, `created_at`, `updated_at`) VALUES(%s, %s, %s, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());',(mysql_uuid, '/', 0,))
            mysql.connection.commit()
            session['pagesession'] = mysql_uuid

        return render_template('index.html', settings=settings, sitekey=settings['recaptcha_site_key']['value'], planlist=planlist, token=token, last_week=crawler_file_data)

# background thread ranking and title email
def threaded_task(subject, title, settings, user_email):
    env = Environment(
      loader=FileSystemLoader("/home/ec2-user/site/templates/"),
      autoescape=select_autoescape(['html', 'xml'])
    )

    template = env.get_template('ranking-email-template.html')
    mail_server = settings['MAIL_SERVER']['value']
    sender_email = settings['MAIL_USERNAME_RANK']['value']
    receiver_email = user_email
    password = settings['MAIL_PASWORD_RANK']['value']
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    print("ranking start!!")
    sorted_freq_title1 = get_title_ranking(title,"1", '1999-01-01','2024-01-01')
    sorted_freq_title2 = get_title_ranking(title,"2", '1999-01-01','2024-01-01')
    print("ranking end!!")
    html = template.render(name=user_email, title=title, freq1=sorted_freq_title1,freq2=sorted_freq_title2)

    msgText = MIMEText(html, 'html')
    msg.attach(msgText)
    try:
        replyEmail = smtplib.SMTP(mail_server, 587)
        replyEmail.connect(mail_server, 587)
        replyEmail.ehlo()
        replyEmail.starttls(context=context)
        replyEmail.login(sender_email, password)
        replyEmail.sendmail(sender_email, receiver_email, msg.as_string())
        print("email sended!!")
    except Exception as e:
        print(e)


# Create Function To Compare Names Ins Users' Text And The Above Imported Name List
def check_names(user_text):
    names_found = ''
    for word in user_text.split():
        if word in names_list.names:
            names_found += word
    return names_found

# Declare Tagging Function That Can Be Called By Any Route

def ExtractKeywords(keys, ngram, duplicate, count):
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = int(ngram)
    deduplication_threshold = float(duplicate)
    numOfKeywords = int(count)
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    key_words = custom_kw_extractor.extract_keywords(keys)
    words_raw = ''
    for kw in key_words:
        words_raw += str(kw) + '\n'

    return words_raw

def spaCy_Extraction(msg, count):
    kwords = list_keywords(msg)
    output = [(x[0]) for x in Counter(kwords).most_common(count)]
    return output



def Rake_Words(msg, count, ngram):
    ngram_count = int(ngram)
    rake_nltk_var = Rake(min_length=0, max_length=ngram_count)
    num = int(count)
    rake_nltk_var.extract_keywords_from_text(msg)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()[:num]
    return(keyword_extracted)



def Spacy_News(msg, count):
    nlp = spacy.load("en_core_web_lg")

    def get_hotwords(text):
        result = []
        pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
        doc = nlp(text.lower())
        for token in doc:
            if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
                continue
            if(token.pos_ in pos_tag):
                result.append(token.text)

        return result

    output = get_hotwords(msg)
    k_words = [(x[0]) for x in Counter(output).most_common(count)]
    return k_words



# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
    # Redirect user to home page if logged-in
    if loggedin():
        return redirect(url_for('home'))
    # Output message if something goes wrong...
    msg = ''
    # Retrieve the settings
    settings = get_settings()
    # Check if "username" and "password" POST requests exist (user submitted form)
    # Also check recaptcha
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form and 'token' in request.form:
        # Create variables for easy access
        # username = request.form['username']
        if not recaptcha.verify():
            return "Please fill the captcha"

        email = request.form['email']
        password = request.form['password']
        token = request.form['token']
        # Retrieve the hashed password
        hash = password + app.secret_key
        hash = hashlib.sha1(hash.encode())
        password = hash.hexdigest();
        print('hashed password is', password)
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s AND password = %s', (email, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        # reCAPTCHA
        if settings['recaptcha']['value'] == 'true':
            if 'g-recaptcha-response' not in request.form:
                return 'Invalid captcha!'
            req = urllib.request.Request('https://www.google.com/recaptcha/api/siteverify', urllib.parse.urlencode({ 'response': request.form['g-recaptcha-response'], 'secret': settings['recaptcha_secret_key']['value'] }).encode())
            response_json = json.loads(urllib.request.urlopen(req).read().decode())
            if not response_json['success']:
                return 'Invalid captcha!'
        # If account exists in accounts table in out database
        if account:
            if account_activation_required and account['activation_code'] != 'activated' and account['activation_code'] != '':
                return 'Please activate your account to login!'
            if csrf_protection and str(token) != str(session['token']):
                return 'Invalid token!'
            # Create session data, we can access this data in other routes
            # check if membership valid
            cursor.execute('SELECT * from membership where fk_user_account=%s', (account['id'],))
            member = cursor.fetchone()
            if not member:
                return 'User is not a member'

            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            session['role'] = account['role']
            session['membership_expiry'] = member['expire_at'].strftime(dt_format)
            session['next_time_check'] = (datetime.datetime.now() + datetime.timedelta(minutes=30)).strftime(dt_format)
            if not is_membership_active():
                return "Membership Expired ! Please add a plan to your account"

            if 'rememberme' in request.form:
                # Create hash to store as cookie
                hash = account['username'] + request.form['password'] + app.secret_key
                hash = hashlib.sha1(hash.encode())
                hash = hash.hexdigest();
                # the cookie expires in 90 days
                expire_date = datetime.datetime.now() + datetime.timedelta(days=90)
                resp = make_response('Success', 200)
                resp.set_cookie('rememberme', hash, expires=expire_date)
                # Update rememberme in accounts table to the cookie hash
                cursor.execute('UPDATE accounts SET rememberme = %s WHERE id = %s', (hash, account['id'],))
                mysql.connection.commit()
                return resp

            print(f"login for {(session['role'])} was successful")
            mysql_uuid = session['pagesession']
            # update the table
            cursor.execute('UPDATE pagevisit set is_login=1 where uid=%s',(mysql_uuid,))
            mysql.connection.commit()
            return 'Success'
        else:
            # Account doesnt exist or username/password incorrect
            return 'Incorrect email/password!'
    # Generate random token that will prevent CSRF attacks
    return redirect('index')

# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests

@app.route('/pythonlogin/register/<int:planid>', methods=['GET', 'POST'])
def register(planid):
    # Redirect user to home page if logged-in
    if loggedin():
        return redirect(url_for('home'))
    # Output message if something goes wrong...
    msg = ''

    # Get the plan
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * from planlist where id=%s',(planid,))
    plan = cursor.fetchone()
    context = {
        'status': "Success",
        'msg': ''
    }

    # session must contain 'email'
    if not session.get('email', None):
        return redirect(url_for('home'))

    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if recaptcha.verify() and request.method == 'POST':
        if not all(input_var in request.form for input_var in ['username', 'password', 'cpassword', 'email', 'planid', 'token']):
            context['msg'] = 'please fill all details'
            context['status'] = "Error"

            return context

        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        cpassword = request.form['cpassword']
        email = request.form['email']
        # Hash the password
        hash = password + app.secret_key
        hash = hashlib.sha1(hash.encode())
        hashed_password = hash.hexdigest();
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor = mysql.get_db().cursor()
        cursor.execute('SELECT * FROM accounts WHERE email = %s', (email,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            context['msg'] = 'Account already exists!'
            context['status'] = "Error"
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            context['msg'] = 'Invalid email address!'
            context['status'] = "Error"
        elif not re.match(r'[A-Za-z0-9]+', username):
            context['msg'] = 'Username must contain only characters and numbers!'
            context['status'] = "Error"
        elif not username or not password or not cpassword or not email:
            context['msg'] = 'Please fill out the form!'
            context['status'] = "Error"
        elif password != cpassword:
            context['msg'] = 'Passwords do not match!'
            context['status'] = "Error"
        elif len(username) < 5 or len(username) > 20:
            context['msg'] = 'Username must be between 5 and 20 characters long!'
            context['status'] = "Error"
        elif len(password) < 5 or len(password) > 20:
            context['msg'] = 'Password must be between 5 and 20 characters long!'
            context['status'] = "Error"
        else:
            # No error found
            if account_activation_required:
                # Account activation enabled
                # Generate a random unique id for activation code
                activation_code = uuid.uuid4()
            else:
                activation_code = "activated"

        # check if there was any error or not
        if context['status'] == "Error":
            return context

        cursor.execute('INSERT INTO accounts (username, password, email, activation_code) VALUES (%s, %s, %s, %s)', (username, hashed_password, email, activation_code,))
        mysql.connection.commit()
        # create new invoice
        cursor.execute('INSERT INTO invoice(email, plan_id, price) VALUES (%s, %s, %s);',(email, plan['id'], plan['price'],))
        # now get the invoice
        mysql.connection.commit()
        cursor.execute('SELECT * from invoice where email=%s and status="unpaid";',(email, ))
        invoice = cursor.fetchone()

        context['status'] = "Success"
        # need to convert the created_date back to invoice format
        current_datetime = invoice['created_at']
        print('current_datetime', current_datetime)

        # current_datetime = datetime.datetime.strptime(current_datetime, dt_format)
        context['next_url'] = url_for('invoice', invoice_id="{}-{}".format(current_datetime.strftime(idt_format), invoice['id']) )

        return context


    elif request.method == 'POST':
        # Form is empty... (no POST data)
        return 'Please fill out the form!'
    # Show registration form with message (if any)

    token = uuid.uuid4()
    session['token'] = token
    return render_template('register.html', msg=msg, plan=plan, email=session.get('email'), token=token)


# http://localhost:5000/pythinlogin/activate/<email>/<code> - this page will activate a users account if the correct activation code and email are provided
@app.route('/pythonlogin/activate/<string:email>/<string:code>', methods=['GET'])
def activate(email, code):
    msg = 'Account doesn\'t exist with that email or the activation code is incorrect!'
    # Check if the email and code provided exist in the accounts table
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM accounts WHERE email = %s AND activation_code = %s', (email, code,))
    account = cursor.fetchone()
    if account:
        # account exists, update the activation code to "activated"
        cursor.execute('UPDATE accounts SET activation_code = "activated" WHERE email = %s AND activation_code = %s', (email, code,))
        mysql.connection.commit()
        # automatically log the user in and redirect to the home page
        session['loggedin'] = True
        session['id'] = account['id']
        session['username'] = account['username']
        session['role'] = account['role']
        return redirect(url_for('home'))
    return render_template('activate.html', msg=msg)

# http://localhost:5000/pythinlogin/contactus - this page will contact us message
@app.route('/pythonlogin/contactus' , methods=['GET', 'POST'])
def contactus():
    msg = ""
    settings = get_settings()
    if request.method == 'POST' and 'name' in request.form and 'email' in request.form and 'subject' in request.form and 'description' in request.form:
        username = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        description = request.form['description']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s', (email,))
        account = cursor.fetchone()

        #reCAPTCHA
        if not loggedin():
            if settings['recaptcha']['value'] == 'true':
                if 'g-recaptcha-response' not in request.form:
                    msg = 'Invalid captcha!'
                    return render_template('contactus.html', msg=msg)
                req = urllib.request.Request('https://www.google.com/recaptcha/api/siteverify', urllib.parse.urlencode({ 'response': request.form['g-recaptcha-response'], 'secret': settings['recaptcha_secret_key']['value'] }).encode())
                response_json = json.loads(urllib.request.urlopen(req).read().decode())
                if not response_json['success']:
                    msg = 'Invalid captcha!'
                    return render_template('contactus.html', msg=msg)
        #Validation
        if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif len(username) < 5 or len(username) > 20:
            msg = 'Username must be between 5 and 20 characters long!'
        elif not username or not email or not subject or not description:
            msg = 'Please fill out the form!'
        else:
            unread_email = True;
            if account:
                # Check if account is activated
                current_client = True;
                if settings['account_activation']['value'] == 'true' and account['activation_code'] != 'activated' and account['activation_code'] != '':
                    msg = 'Thanks for your inquiry, we will respond promptly.'
                else:
                    msg = 'Your email address shows you are a valued customer, and we will respond to your inquiry promptly.'
                cursor.execute('SELECT * FROM membership WHERE fk_user_account = %s', (account['id'],))
                member = cursor.fetchone()
                if member == None:
                    msg = 'Thanks for your inquiry, we will respond promptly.'
                elif (member != None) & (member['expire_at'] < datetime.datetime.now()):
                    unread_email = False;
            else:
                msg = 'Thanks for your inquiry, we will respond promptly.'
                current_client = False;

            # # Try to log in to server and send email (Author Email-Contact Us)
            env = Environment(
                        loader=FileSystemLoader("/home/ec2-user/site/templates/"),
                        autoescape=select_autoescape(['html', 'xml'])
                    )
            template = env.get_template('contact-email-template.html')
            sender_email = settings['MAIL_USERNAME']['value']
            receiver_email = sender_email
            password = settings['MAIL_PASSWORD']['value']
            mail_server = settings['MAIL_SERVER']['value']
            message = MIMEMultipart()
            message['Subject'] = subject
            message['From'] = sender_email
            message['To'] = receiver_email
            html = template.render(name=username, email=email, description=description)
            msgText = MIMEText(html, 'html')
            message.attach(msgText)
            try:
                authorEmail = smtplib.SMTP(mail_server, 587)
                authorEmail.connect(mail_server, 587)
                authorEmail.ehlo()
                authorEmail.starttls(context=context)
                authorEmail.login(sender_email, password)
                authorEmail.sendmail(sender_email, receiver_email, message.as_string())
                print("ContactUs smtp server run")
            except Exception as e:
                # print any error message to stdout
                print(e)
            cursor.execute('INSERT INTO contactus (username, email, subject, description, unread_email, current_client) VALUES (%s, %s, %s, %s, %s, %s)', (username, email, subject, description, unread_email, current_client))
            mysql.connection.commit()
    return render_template('contactus.html', msg=msg)


# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/pythonlogin/home')
def home():
    # Check if user is loggedin
    if loggedin():
        # User is loggedin show them the home page
        return render_template('home.html', selected="home", username=session['username'], role=session['role'])
    # User is not loggedin redirect to login page
    return redirect(url_for('index'))


# http://localhost:5000/pythinlogin/title - this will be the title page, only accessible for loggedin users
@app.route('/pythonlogin/title/', methods=['POST','GET'])
def title():
    # Check if user is loggedin
    if loggedin():
        # if request.method == 'POST':
        #     input__ = request.form['keywords']
        #     algo = request.form['myselect']
        #     d = request.form['demo']
        #     print(d + "-----------> range...")
        #     if (algo == "1"):
        #         out_list = get_title_ranking(str(input__),"1",str(d)+'-01-01','2022-01-01')
        #         return render_template('title.html', selected="title",username=session['username'],vis="show",inp=input__, words=str(out_list).replace("{","").replace("}",""), role=session['role'])
        #     elif (algo == "2"):
        #         out_list = get_title_ranking(str(input__),"2",str(d)+'-01-01','2022-01-01')
        #         return render_template('title.html', selected="title",username=session['username'],vis="show",inp=input__, words=str(out_list).replace("{","").replace("}",""), role=session['role'])
        #     else:
        #         return render_template('title.html', selected="title",username=session['username'],vis="show",inp=input__, words="Please Select the algorithm", role=session['role'])
        # else:
            # User is loggedin show them the home page
        return render_template('title.html', selected="title", username=session['username'], role=session['role'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


# http://localhost:5000/process_title - this will be the ajax link to send and fetch rewriting data and results
@app.route('/process_title', methods=['POST','GET'])
def title_results():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        d = data[2]["d"]
        if data[1]["Algo"] == "1":
            out_list = get_title_ranking(data[0]["Text"],"1",str(d)+'-01-01','2024-01-01')
        elif data[1]["Algo"] == "2":
            out_list = get_title_ranking(data[0]["Text"],"2",str(d)+'-01-01','2024-01-01')
        else:
            out_list = "Please Select the algorithm"
        returned = str(out_list).replace("{","").replace("}","")
    results = {'word': returned, "vis" : "show"}
    print(returned)
    print(results)
    return jsonify(results)


# http://localhost:5000/pythinlogin/descriptions - this will be the descriptions compare page, only accessible for loggedin users
@app.route('/pythonlogin/descriptions/', methods=['POST','GET'])
def descriptions():
    # Check if user is loggedin
    if loggedin():
        # if request.method == 'POST':
        #         input_text = request.form['n_samples']
        #         algo = request.form['algorithm']
        #         lists_ = None
        #         if (algo == '1'):
        #             lists_ = get_similar_words(input_text)
        #         elif (algo == '2'):
        #             lists_ = search_for_similar(input_text)
        #         if (lists_ != None):
        #             return render_template('descriptions.html',selected="descriptions",username=session['username'],txt=input_text,
        #             percentage_1=str(lists_[0][0]) + "%",
        #             title_1=lists_[0][1],
        #             isbn_1=lists_[0][2],
        #             description_1=lists_[0][3],

        #             percentage_2=str(lists_[1][0]) + "%",
        #             title_2=lists_[1][1],
        #             isbn_2=lists_[1][2],
        #             description_2=lists_[1][3],

        #             percentage_3=str(lists_[2][0]) + "%",
        #             title_3=lists_[2][1],
        #             isbn_3=lists_[2][2],
        #             description_3=lists_[2][3],

        #             percentage_4=str(lists_[3][0]) + "%",
        #             title_4=lists_[3][1],
        #             isbn_4=lists_[3][2],
        #             description_4=lists_[3][3],

        #             role=session['role'])
        #         else:
        #             return render_template('descriptions.html',selected="descriptions",username="Please select Algorithm",role=session['role'])
        return render_template('descriptions.html',selected="descriptions",username=session['username'],role=session['role'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/process_description - this will be the descriptions compare page, only accessible for loggedin users
@app.route('/process_description', methods=['POST','GET'])
def descriptions_result():
    settings = get_settings()
    if request.method == 'POST':
        data = request.get_json()
        print("descriptions_result")
        print(data)
        print(data[0]["Text"])
        if data[1]["Algo"] == '1':
            lists_ = get_similar_words(data[0]["Text"])
        elif data[1]["Algo"] == '2':
            lists_ = search_for_similar(data[0]["Text"])
        else:
            lists_ = "Please Select the algorithm"
        returned = lists_
    results = {'returned': lists_}
    return jsonify(results)

# http://localhost:5000/pythinlogin/rewrite - this will be the rewrite page, only accessible for loggedin users
@app.route('/pythonlogin/rewrite/', methods=['POST','GET'])
def rewrite():
    # Check if user is loggedin
    if loggedin():
        # User is loggedin show them the home page
        return render_template('rewrite.html', selected='rewrite', username=session['username'], role=session['role'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


# http://localhost:5000/process_rewrite - this will be the ajax link to send and fetch rewriting data and results
@app.route('/process_rewrite', methods=['POST','GET'])
def rewrite_results():
    settings = get_settings()
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        sentences = []
        if data[4]['Lang'] == 'books':
            sentence = data[0]['Text']
            length = int(data[1]['Max_Length'])
            returned_sent = int(data[2]['Returned_Sents'])
            change_rate = int(data[3]['Change_Rate']) / 10
            print('All Variables Stored.')
            print(sentence)
            print(length)
            print(returned_sent)
            # textgenie = TextGenie(paraphrase_model_name='hetpandya/t5-small-tapaco', mask_model_name='bert-base-uncased',)
            try:
              # define the Tokenizer:
              tokenizer = T5Tokenizer.from_pretrained('t5-base')
              # inputs for the paraphrasing task
              with open('/home/ec2-user/site/static/model1.pkl', 'rb') as file:
                paraphraser = pickle.load(file)
              text =  "paraphrase: " + sentence + ""
              encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
              print(encoding)
              input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
              # generating required outputs from the model
              """
              parameters:
              max_length - number of tokens appearing in the sentence
              top_p - the change rate captured during the paraphrasing task
              num_return_sequences - number of paraphrased sentence to return
              """
              if change_rate>0:
                change_rate = 1
              else:
                change_rate = 0

              beam_outputs = paraphraser.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                do_sample=True,
                max_length=length,
                top_k=120,
                top_p=change_rate,
                early_stopping=True,
                num_return_sequences=returned_sent
              )

              final_outputs =[]
              for beam_output in beam_outputs:
                sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                if sent.lower() != sentence.lower() and sent not in final_outputs:
                  final_outputs.append(sent)

              for i, final_output in enumerate(final_outputs):
                sentences.append(final_output)
                print("{}: {}".format(i, final_output))
            except Exception as e:
                sentences.append(e)
            print(sentences)
        elif data[4]['Lang'] == 'news':
            sentence = data[0]['Text']
            length = int(data[1]['Max_Length'])
            returned_sent = int(data[2]['Returned_Sents'])
            change_rate = int(data[3]['Change_Rate']) / 10

            sw_normal = data[5]['Sw_Normal']
            sw_entire = data[6]['Sw_Entire']
            sw_summary = data[7]['Sw_Summary']
            sw_phrase = data[8]['Sw_Phrase']
            sw_mode = int(data[9]['Sw_Mode'])

            print('All Variables Stored.')
            print(sentence)
            print(length)
            print(returned_sent)
            print('sw_normal:', sw_normal)
            print('sw_entire:', sw_entire)
            print('sw_summary:', sw_summary)
            print('sw_phrase:', sw_phrase)
            print('sw_mode:', sw_mode)

            print("News Model Chosen")
            # your Spin Rewriter email address goes here
            email_address = settings['SPINWRITER EMAIL']['value']

            # your unique Spin Rewriter API key goes here
            api_key = settings['SPINWRITER KEY']['value']

            # Spin Rewriter API settings - authentication:
            spinrewriter_api = SpinRewriterAPI(email_address, api_key)

            # (optional) Set the confidence level of the One-Click Rewrite process.
            spinrewriter_api.set_confidence_level("medium")

            # (optional) Set whether the One-Click Rewrite process automatically protects Capitalized Words outside the article's title.
            spinrewriter_api.set_auto_protected_terms(True)

            # (optional) Set the confidence level of the One-Click Rewrite process.
            spinrewriter_api.set_confidence_level("medium")

            # (optional) Set whether the One-Click Rewrite process uses nested spinning syntax (multi-level spinning) or not.
            spinrewriter_api.set_nested_spintax(True)

            # (optional) Set whether Spin Rewriter rewrites complete sentences on its own.
            spinrewriter_api.set_auto_sentences(sw_normal)

            # (optional) Set whether Spin Rewriter rewrites entire paragraphs on its own.
            spinrewriter_api.set_auto_paragraphs(sw_entire)

            # (optional) Set whether Spin Rewriter writes additional paragraphs on its own.
            spinrewriter_api.set_auto_new_paragraphs(sw_summary)

            # (optional) Set whether Spin Rewriter changes the entire structure of phrases and sentences.
            spinrewriter_api.set_auto_sentence_trees(sw_phrase)

            # (optional) Set the desired spintax format to be used with the returned spun text.
            spinrewriter_api.set_spintax_format("{|}")

            # (optional) Sets whether Spin Rewriter should only use synonyms (where available) when generating spun text.
            spinrewriter_api.set_use_only_synonyms(False)

            # (optional) Sets whether Spin Rewriter should intelligently randomize the order of paragraphs
            # and lists when generating spun text.
            spinrewriter_api.set_reorder_paragraphs(False)

            # # (optional) Sets whether Spin Rewriter should automatically enrich generated articles with headings, bulpoints, etc.
            # spinrewriter_api.set_add_html_markup(False)

            # # (optional) Sets whether Spin Rewriter should automatically convert line-breaks to HTML tags.
            # spinrewriter_api.set_use_html_linebreaks(False)

            if sw_mode == 0:
                # Make the actual API request and save the response as a native JSON dictionary or False on error
                response = spinrewriter_api.get_text_with_spintax(sentence)
            elif sw_mode == 1:
                # Make the actual API request and save the response as a native JSON dictionary or False on error
                response = spinrewriter_api.get_unique_variation(sentence)
            else:
                # Make the actual API request and save the response as a native JSON dictionary or False on error
                response = spinrewriter_api.get_unique_variation_from_spintax(sentence)

            sentences.append(response['response'])
            print(sentences)
        else:
            print("Use Other Library")
            sentences.append('Pick Other Library')
        returned = []
        loop_count = 1
        for sentence in sentences:
            if loop_count == 1:
                result = str(loop_count) + '. ' + str(sentence)
                returned.append(result)
                loop_count += 1
            else:
                result = str('&#10;') + str(loop_count) + '. ' + str(sentence)
                returned.append(result)
                loop_count += 1
    results = {'returned': returned}
    print(returned)
    print(results)
    return jsonify(results)

# http://localhost:5000/pythinlogin/covercompare - this will be the cover-compare page, only accessible for loggedin users
dropzone = Dropzone(app)
@app.route('/pythonlogin/covercompare')
def covercompare():
    # Check if user is loggedin
    if loggedin():
        return render_template('covercompare.html', selected='covercompare', username=session['username'], role=session['role'], tester='no')
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/pythinlogin/covercompare - this will be the cover-compare page, only accessible for loggedin users
@app.route('/pythonlogin/covercompare', methods=['POST'])
def covercompareresults():
    # Check if user is loggedin
    if loggedin():
      if 'file' not in request.files:
        return redirect(url_for('covercompare'))
      f= request.files['file']
      if len(f.filename)==0:
        return redirect(url_for('covercompare'))
      if f:
        f.save(os.path.join(app.config['UPLOADED_PATH'],f.filename))
        f_name = f.filename
        img = '/uploads/cover_uploads/' + f_name
        get_heatmap('/home/ec2-user/site/static/uploads/cover_uploads/'+f.filename)
        heatmap = f.filename.split('.')[0]+'_heatmap.jpg'
        titles, images = cover_compare.query('/home/ec2-user/site/static/uploads/cover_uploads/'+f.filename)
        print(titles[0][1])
        isbn1 = titles[0][1]
        title1 = titles[0][0]
        img1 = images[isbn1]
        isbn2 = titles[1][1]
        title2 = titles[1][0]
        img2 = images[isbn2]
        isbn3 = titles[2][1]
        title3 = titles[2][0]
        img3 = images[isbn3]
        isbn4 = titles[3][1]
        title4 = titles[3][0]
        img4 = images[isbn4]
        print(images[isbn1])
        file_present = True
        tester = 'yes'
        return render_template('covercompare.html', selected='covercompare', username=session['username'], role=session['role'], img=f.filename, tester=tester, heatmap=heatmap, isbn1=isbn1, title1 =title1, img1 = img1, isbn2=isbn2, title2 =title2, img2 = img2, isbn3=isbn3, title3 =title3, img3 = img3, isbn4=isbn4, title4 =title4, img4 = img4)

      else:
        return redirect(url_for('covercompare'))
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/uploads/cover_uploads/<filename>')
def send_uploaded_file(filename=''):
  #from flask import send_from_directory
  return redirect(url_for('static',filename='uploads/cover_uploads/'+filename), code=301)

# http://localhost:5000/pythinlogin/testimonial - this will be the cover-compare page, only accessible for loggedin users
#dropzone = Dropzone(app)
@app.route('/pythonlogin/testimonial ')
def testimonial ():
    # Check if user is loggedin
    if loggedin():
        return render_template('testimonial.html', selected='testimonial ', username=session['username'], role=session['role'], tester='no')
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


# http://localhost:5000/pythonlogin/about - view the about page
@app.route('/pythonlogin/about', methods=['GET', 'POST'])
def about():
    settings = get_settings()
    # Check if admin is logged-in
    if loggedin():
        # Render the about template
        return render_template('about.html', selected='about', settings=settings, sitekey=settings['recaptcha_site_key']['value'], username=session['username'], role=session['role'])
    return redirect(url_for('login'))


# http://localhost:5000/pythinlogin/tagging - this will be the tagging page, only accessible for loggedin users
@app.route('/pythonlogin/tagging')
def tagging():
    # Check if user is loggedin
    if loggedin():
        # User is loggedin show them the tagging page
        return render_template('tagging.html', selected='tagging', username=session['username'], role=session['role'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/process_keywords - this will be the ajax link to send and fetch tagging data and results
@app.route('/process_keywords', methods=['POST','GET'])
def tagging_results():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        if data[5]['Lib'] == 'yake-submit':
            # words = ['New', 'Old']
            text = data[0]['Text']
            ngram = data[1]['Ngram']
            duplicate = int(data[2]['Duplicate'])
            count = int(data[3]['Count'])
            unchanged = data[4]['Separate']
            print(text, ngram, duplicate, count, unchanged)
            ngram /= 10
            words = ExtractKeywords(text, ngram, duplicate, count)
            print(words)
        elif data[5]['Lib'] == 'default':
            words = ''

        elif data[5]['Lib'] == 'spacy-submit':
            text = data[0]['Text']
            count = data[3]['Count']
            words = spaCy_Extraction(text, count)
            print(words)

        elif data[5]['Lib'] == 'rake-submit':
            text = data[0]['Text']
            ngram = data[1]['Ngram']
            duplicate = int(data[2]['Duplicate'])
            count = data[3]['Count']
            unchanged = str(data[4]['Separate'])
            print("rake-submit")
            print(text, ngram, count, unchanged)
            # ngram %= 10
            words = Rake_Words(text, count, ngram)
            print('The Words Are', words)

        else:
            words = ['Not At All']

    results = {'words': words}
    return jsonify(results)

# http://localhost:5000/pythinlogin/sentimentanalysis - this will be the preview page, only accessible for loggedin users
@app.route('/pythonlogin/sentimentanalysis', methods=['POST', 'GET'])
def sentimentanalysis():
    # Check if user is loggedin
    if loggedin():
        # User is loggedin show them the home page
        exp = ""
        emotion = None
        if request.method == 'POST':
            text = tokenizer(request.form['entry'])
            method = request.form['classifier']
            n_samples = request.form['n_samples']
            if any(not v for v in [text, n_samples]):
                raise ValueError("Please do not leave text fields blank.")

            if method != "base":
                exp = explainer(method,
                    path_to_file=METHODS[method]['file'],
                    text=text,
                    lowercase=METHODS[method]['lowercase'],
                    num_samples=int(n_samples))
                exp = exp.as_html()
            text = text.replace("\\n", " ")
            emotion = get_emotion(text)
            return render_template('sentimentanalysis.html', selected='sentimentanalysis', username=session['username'], role=session['role'], exp=exp, entry=text, n_samples=n_samples, classifier=method, emotion=emotion)
        return render_template('sentimentanalysis.html', selected='sentimentanalysis', username=session['username'], role=session['role'], exp=exp, emotion=emotion)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))



# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/pythonlogin/preference')
def profile():
    # Check if user is loggedin
    if loggedin():
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', selected="profile", account=account, role=session['role'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/pythinlogin/profile/edit - user can edit their existing details
@app.route('/pythonlogin/preference/edit', methods=['GET', 'POST'])
def edit_profile():
    # Check if user is loggedin
    if loggedin():
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # Output message
        msg = ''
        # Check if "username", "password" and "email" POST requests exist (user submitted form)
        if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
            # Create variables for easy access
            username = request.form['username']
            password = request.form['password']
            email = request.form['email']
            # Retrieve account by the username
            cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
            account = cursor.fetchone()
            # validation check
            if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                msg = 'Invalid email address!'
            elif not re.match(r'[A-Za-z0-9]+', username):
                msg = 'Username must contain only characters and numbers!'
            elif not username or not email:
                msg = 'Please fill out the form!'
            elif session['username'] != username and account:
                msg = 'Username already exists!'
            elif len(username) < 5 or len(username) > 20:
                return 'Username must be between 5 and 20 characters long!'
            elif len(password) < 5 or len(password) > 20:
                return 'Password must be between 5 and 20 characters long!'
            else:
                cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
                account = cursor.fetchone()
                current_password = account['password']
                if password:
                    # Hash the password
                    hash = password + app.secret_key
                    hash = hashlib.sha1(hash.encode())
                    current_password = hash.hexdigest();
                # update account with the new details
                cursor.execute('UPDATE accounts SET username = %s, password = %s, email = %s WHERE id = %s', (username, current_password, email, session['id'],))
                mysql.connection.commit()
                msg = 'Updated!'
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile-edit.html', selected="profile", account=account, role=session['role'], msg=msg)
    return redirect(url_for('login'))

# http://localhost:5000/pythinlogin/forgotpassword - user can use this page if they have forgotten their password
@app.route('/pythonlogin/forgotpassword', methods=['GET', 'POST'])
def forgotpassword():
    settings = get_settings()
    msg = ''
    if request.method == 'POST' and 'email' in request.form:
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s', (email,))
        account = cursor.fetchone()
        if account:
            # Generate unique ID
            reset_code = uuid.uuid4()
            # Update the reset column in the accounts table to reflect the generated ID
            cursor.execute('UPDATE accounts SET reset = %s WHERE email = %s', (reset_code, email,))
            mysql.connection.commit()
            # Generate reset password link
            reset_link = settings['DOMAIN']['value'] + url_for('resetpassword', email = email, code = str(reset_code))
            # change the email body below
            sender_email = settings['MAIL_USERNAME']['value']
            receiver_email = email
            password = settings['MAIL_PASSWORD']['value']
            mail_server = settings['MAIL_SERVER']['value']
            message = MIMEMultipart()
            message['Subject'] = "Forgot Passwords"
            message['From'] = sender_email
            message['To'] = email
            html = "<html><head></head><body>" + '<p>Please click the following link to reset your password: <a href="' + str(reset_link) + '">' + str(reset_link) + '</a></p>'+'</body></html>'
            email_info = MIMEText(html, 'html')
            # email_info.body = 'Please click the following link to reset your password: ' + str(reset_link)
            # email_info.html = '<p>Please click the following link to reset your password: <a href="' + str(reset_link) + '">' + str(reset_link) + '</a></p>'
            message.attach(email_info)
            try:
                resetEmail = smtplib.SMTP(mail_server, 587)
                resetEmail.connect(mail_server, 587)
                resetEmail.ehlo()
                resetEmail.starttls(context=context)
                resetEmail.login(sender_email, password)
                resetEmail.sendmail(sender_email, receiver_email, message.as_string())
                print("Reset Password smtp server run")
            except Exception as e:
                # print any error message to stdout
                print(e)
            msg = 'Reset password link has been sent to your email!'
        else:
            msg = 'An account with that email does not exist!'
    return render_template('forgotpassword.html', msg=msg)

# http://localhost:5000/pythinlogin/resetpassword/EMAIL/CODE - proceed to reset the user's password
@app.route('/pythonlogin/resetpassword/<string:email>/<string:code>', methods=['GET', 'POST'])
def resetpassword(email, code):
    msg = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # Retrieve the account with the email and reset code provided from the GET request
    cursor.execute('SELECT * FROM accounts WHERE email = %s AND reset = %s', (email, code,))
    account = cursor.fetchone()
    # If account exists
    if account:
        # Check if the new password fields were submitted
        if request.method == 'POST' and 'npassword' in request.form and 'cpassword' in request.form:
            npassword = request.form['npassword']
            cpassword = request.form['cpassword']
            # Password fields must match
            if npassword == cpassword and npassword != "":
                # Hash new password
                hash = npassword + app.secret_key
                hash = hashlib.sha1(hash.encode())
                npassword = hash.hexdigest();
                # Update the user's password
                cursor.execute('UPDATE accounts SET password = %s, reset = "" WHERE email = %s', (npassword, email,))
                mysql.connection.commit()
                msg = 'Your password has been reset, you can now <a href="https://bestsellercreator.com">login</a>!'
            else:
                msg = 'Passwords must match and must not be empty!'
        return render_template('resetpassword.html', msg=msg, email=email, code=code)
    return 'Invalid email and/or code!'

# http://localhost:5000/pythinlogin/logout - this will be the logout page
@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out

    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    session.pop('role', None)
    # Remove cookie data "remember me"
    resp = make_response(redirect(url_for('index')))
    resp.set_cookie('rememberme', expires=0)
    return resp

def is_membership_active():
    # get username from session
    if session.get('membership_expiry', None):
        expire_at = datetime.datetime.strptime(session['membership_expiry'], dt_format)
        if datetime.datetime.now() > expire_at:
            return False
        else:
            # need to check timeout
            if not session.get('next_time_check', None):
                return False

            if datetime.datetime.strptime(session['next_time_check'], dt_format) > datetime.datetime.now():
                    return True
            else:
                # check the expiry now
                cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cursor.execute('SELECT * from membership where fk_user_account=%s', (session['id'],))
                member  = cursor.fetchone()
                if datetime.datetime.now() > member['expire_at']:
                    # set the next time interval to 30 min
                    session['next_time_check'] = (datetime.datetime.now() + datetime.timedelta(minutes=30)).strftime(dt_format)
                    return True
                else:
                    # yes, the expire datetime reached
                    return False
    else:
        # it seems the cookies not set, so logout
        False

# Check if logged in function, update session if cookie for "remember me" exists
def loggedin():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # print(f"login check called {session}")
    if 'loggedin' in session:
        # check if membership is active
        if is_membership_active():
            cursor.execute('UPDATE accounts SET last_seen = NOW() WHERE id = %s', (session['id'],))
            mysql.connection.commit()
            return True
        else:
            return False
    elif 'rememberme' in request.cookies:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # check if remembered, cookie has to match the "rememberme" field
        cursor.execute('SELECT * FROM accounts WHERE rememberme = %s', (request.cookies['rememberme'],))
        account = cursor.fetchone()
        cursor.execute('SELECT * FROM membership WHERE fk_user_account=%s', (account['id'],))
        member = cursor.fetchone()
        if account:
            # update session variables
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            session['role'] = account['role']
            session['membership_expiry'] = member['expire_at'].strftime(dt_format)
            session['next_time_check'] = (datetime.datetime.now() + datetime.timedelta(minutes=30)).strftime(dt_format)
            return True
    # account not logged in return false
    return False


@app.route('/<path:filename>')
def send_file(filename):
    return send_from_directory(app.static_folder, filename)


# payment accept
@app.route('/checkout/<int:plan_id>', methods=['GET', 'POST'])
def checkout_page(plan_id):
    planid = plan_id if plan_id else 1

    # Redirect user to home page if logged-in
    if loggedin():
        return redirect(url_for('home'))

    # Return Context
    context = {
        'status': 'Error',
        'msg' : ''
    }

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * from planlist where id=%s', (planid, ))
    plan = cursor.fetchone()
    if not plan:
        context['msg'] = "Invalid Plan Selected"
        return context
    # token
    if request.method == "POST":
        if 'email' in request.form and 'token' in request.form and recaptcha.verify():
            email = request.form['email']

            # try to get fetch user detail
            cursor.execute('SELECT * from accounts where email=%s', (email,))
            user = cursor.fetchone()
            if user:
                # update the session to current email
                session['email'] = email
                # Then we will redirect to payment html
                # Check if plan id is free 7 days
                if plan_id == 1:
                    # return the User an Error Message
                    context['msg'] = "User already exist. Please choose different plan"
                    return context
                else:
                    # create an invoice for the user
                    # check if there is any invoice unpaid with that user
                    #
                    cursor.execute('SELECT * from invoice where email=%s and status="unpaid";',(email,))
                    invoice = cursor.fetchone()
                    current_datetime = datetime.datetime.now().strftime(dt_format)
                    if invoice:
                        # we need to update the invoice with new plan
                        cursor.execute('UPDATE invoice set plan_id=%s, price=%s, created_at=%s where id=%s', (
                                        plan['id'],
                                        plan['price'],
                                        current_datetime,
                                        invoice['id']
                                        ))


                        mysql.connection.commit()
                        # now get the updated invoice
                        cursor.execute('SELECT * from invoice where id=%s;', (invoice['id'],))
                        invoice = cursor.fetchone()
                    else:
                        # create new invoice
                        cursor.execute('INSERT INTO invoice(email, plan_id, price) VALUES (%s, %s, %s);',(email, plan['id'], plan['price'],))
                        # now get the invoice
                        mysql.connection.commit()
                        cursor.execute('SELECT * from invoice where email=%s and status="unpaid";',(email, ))
                        invoice = cursor.fetchone()

                    context['status'] = "Success"
                    current_datetime = datetime.datetime.strptime(current_datetime, dt_format)
                    context['next_url'] = url_for('invoice', invoice_id="{}-{}".format(current_datetime.strftime(idt_format), invoice['id']) )

                    return context
            else:
                # Redirect to Registration page
                session['email'] = email
                context['status'] = "Success"
                context['next_url'] = url_for('register', planid=plan['id'])
            return context

    else:

        token = uuid.uuid4()
        session['token'] = token
        # create a form to collect email
        return render_template('checkout.html', plan=plan, token=token)


def create_update_membership(cursor, payment_id, invoice, plan):

    cursor.execute("UPDATE invoice set payment_id=%s, status='paid' where id=%s", (
                   payment_id, invoice['id']))
    mysql.connection.commit()

    # Create or update membership
    today = datetime.datetime.now()
    expire_date = today + datetime.timedelta(days=plan['validity'])

    # find account with email
    cursor.execute('SELECT * from accounts where email=%s', (
            invoice['email'],

            ))
    user = cursor.fetchone()

    # find membership
    cursor.execute('SELECT * from membership where fk_user_account=%s', (user['id'],))
    membership = cursor.fetchone()

    if membership:
        # update the validity
        if membership['is_active'] != 1:

            # need to send activation mail
            sendmail = True
        else:
            # no need to send
            sendmail = False
        cursor.execute('UPDATE membership set plan_detail=%s, is_active=%s, expire_at=%s, start_at=%s where id=%s', (
                "{} days Membership Plan".format(plan['validity']),
                1,
                expire_date.strftime(dt_format),
                today.strftime(dt_format),
                membership['id'],
                ))
    else:
        cursor.execute('INSERT INTO membership(plan_detail, is_active, expire_at, start_at, fk_user_account) VALUES (%s, %s, %s, %s, %s);',(
               "{} days Membership Plan".format(plan['validity']),
                1,
                expire_date.strftime(dt_format),
                today.strftime(dt_format),
                user['id'],
               ))
        sendmail = True

    mysql.connection.commit()

    # Now send email
    if sendmail:
        # Create new message
        email_info = Message('Account Activation Required', sender = app.config['MAIL_USERNAME'], recipients = [user['email']])
        # Activate Link URL
        activate_link = app.config['DOMAIN'] + url_for('activate', email=user['email'], code=str(user['activation_code']))
        # Define and render the activation email template
        email_info.body = render_template('activation-email-template.html', link=activate_link)
        email_info.html = render_template('activation-email-template.html', link=activate_link)
        # send activation email to user
        mail.send(email_info)
        # ID format BSC-<min 6 and max 20> increment number
        return "Please check your email for activation link"
    else:
        return "Thanks for your payment"


@app.route('/invoice/<string:invoice_id>', methods=['GET', 'POST'])
def invoice(invoice_id):
    # invoice_id format : YYYYMMDDHHMMSS-<primary key>
    print(invoice_id)
    if len(invoice_id.split('-')) == 2:
        # valid now need to check the invoice data
        invoice_date, inv_id = invoice_id.split('-')
        try:
            invoice_date = datetime.datetime.strptime(invoice_date, idt_format)
        except ValueError:
            return "<h1> Invalid Invoice </h1>"

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * from invoice where id=%s and created_at=%s', (inv_id, invoice_date.strftime(dt_format),))
        invoice = cursor.fetchone()
        if invoice:
            # get plan
            cursor.execute('SELECT * from planlist where id=%s', (invoice['plan_id'], ))
            plan = cursor.fetchone()
            # check if current user equals invoice user
            if session['email'] == invoice['email']:
                # apply logic check here
                if request.method == "POST":
                    # get the payment id
                    payment_id = request.form['payment_id']

                    status = create_update_membership(cursor, payment_id, invoice, plan)
                    # Update the invoice status
                    return {'msg': status}

                else:
                    return render_template(
                        'payment.html',
                         invoice=invoice,
                         plan=plan,
                         paypal_client_id=app.config['PAYPAL_CLIENT_ID']
                        )
            else:
                return "<h1> You are not authorized to view this </h1>"

        else:
            return "<h1> Invoice not found</h1>"

    else:
        return "<h1> Invoice doesnot exist </h1>"

# ADMIN PANEL
# http://localhost:5000/pythonlogin/admin/ - admin dashboard, view new accounts, active accounts, statistics
@app.route('/pythonlogin/admin/', methods=['GET', 'POST'])
def admin():
        # Check if admin is logged-in
        if not admin_loggedin():
                return redirect(url_for('login'))
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # Retrieve new accounts for the current date
        cursor.execute('SELECT * FROM accounts WHERE cast(registered as DATE) = cast(now() as DATE) ORDER BY registered DESC')
        accounts = cursor.fetchall()
        # Get the total number of accounts
        cursor.execute('SELECT COUNT(*) AS total FROM accounts')
        accounts_total = cursor.fetchone()
        # Get the total number of active accounts (<1 month)
        cursor.execute('SELECT COUNT(*) AS total FROM accounts WHERE last_seen < date_sub(now(), interval 1 month)')
        inactive_accounts = cursor.fetchone()
        # Retrieve accounts created within 1 day from the current date
        cursor.execute('SELECT * FROM accounts WHERE last_seen > date_sub(now(), interval 1 day) ORDER BY last_seen DESC')
        active_accounts = cursor.fetchall()
        # Get the total number of inactive accounts
        cursor.execute('SELECT COUNT(*) AS total FROM accounts WHERE last_seen > date_sub(now(), interval 1 month)')
        active_accounts2 = cursor.fetchone()
        # Render admin dashboard template
        return render_template('admin/dashboard.html', accounts=accounts, selected='dashboard', selected_child='view', accounts_total=accounts_total['total'], inactive_accounts=inactive_accounts['total'], active_accounts=active_accounts, active_accounts2=active_accounts2['total'], time_elapsed_string=time_elapsed_string)

# http://localhost:5000/pythonlogin/admin/accounts - view all accounts
@app.route('/pythonlogin/admin/accounts/<string:msg>/<string:search>/<string:status>/<string:activation>/<string:role>/<string:order>/<string:order_by>/<int:page>', methods=['GET', 'POST'])
@app.route('/pythonlogin/admin/accounts', methods=['GET', 'POST'], defaults={'msg': '', 'search' : '', 'status': '', 'activation': '', 'role': '', 'order': 'DESC', 'order_by': '', 'page': 1})
def admin_accounts(msg, search, status, activation, role, order, order_by, page):
    # Check if admin is logged-in
        if not admin_loggedin():
                return redirect(url_for('login'))
        # Params validation
        msg = '' if msg == 'n0' else msg
        search = '' if search == 'n0' else search
        status = '' if status == 'n0' else status
        activation = '' if activation == 'n0' else activation
        role = '' if role == 'n0' else role
        order = 'DESC' if order == 'DESC' else 'ASC'
        order_by_whitelist = ['id','username','email','activation_code','role','registered','last_seen']
        order_by = order_by if order_by in order_by_whitelist else 'id'
        results_per_page = 20
        param1 = (page - 1) * results_per_page
        param2 = results_per_page
        param3 = '%' + search + '%'
        # SQL where clause
        where = '';
        where += 'WHERE (username LIKE %s OR email LIKE %s) ' if search else ''
        # Add filters
        if status == 'active':
                where += 'AND last_seen > date_sub(now(), interval 1 month) ' if where else 'WHERE last_seen > date_sub(now(), interval 1 month) '
        if status == 'inactive':
                where += 'AND last_seen < date_sub(now(), interval 1 month) ' if where else 'WHERE last_seen < date_sub(now(), interval 1 month) '
        if activation == 'pending':
                where += 'AND activation_code != "activated" ' if where else 'WHERE activation_code != "activated" '
        if role:
                where += 'AND role = %s ' if where else 'WHERE role = %s '
        # Params array and append specified params
        params = []
        if search:
                params.append(param3)
                params.append(param3)
        if role:
                params.append(role)
        # Fetch the total number of accounts
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT COUNT(*) AS total FROM accounts ' + where, params)
        accounts_total = cursor.fetchone()
        # Append params to array
        params.append(param1)
        params.append(param2)
    # Retrieve all accounts from the database
        cursor.execute('SELECT * FROM accounts ' + where + ' ORDER BY ' + order_by + ' ' + order + ' LIMIT %s,%s', params)
        accounts = cursor.fetchall()
        # Determine the URL
        url = url_for('admin_accounts') + '/n0/' + (search if search else 'n0') + '/' + (status if status else 'n0') + '/' + (activation if activation else 'n0') + '/' + (role if role else 'n0')
        # Handle output messages
        if msg:
                if msg == 'msg1':
                        msg = 'Account created successfully!';
                if msg == 'msg2':
                        msg = 'Account updated successfully!';
                if msg == 'msg3':
                        msg = 'Account deleted successfully!'
        # Render the accounts template
        return render_template('admin/accounts.html', accounts=accounts, selected='accounts', selected_child='view', msg=msg, page=page, search=search, status=status, activation=activation, role=role, order=order, order_by=order_by, results_per_page=results_per_page, accounts_total=accounts_total['total'], math=math, url=url, time_elapsed_string=time_elapsed_string)

# http://localhost:5000/pythonlogin/admin/inquiries - view all inquires
@app.route('/pythonlogin/admin/inquiries', methods=['GET', 'POST'], defaults={'search' : '', 'status': '', 'order': 'DESC', 'order_by': '', 'page': 1})
def admin_inquiries(search, status, order, order_by, page):
    # Check if admin is logged-in
        if not admin_loggedin():
                return redirect(url_for('login'))
        # Params validation
        search = '' if search == 'n0' else search
        status = '' if status == 'n0' else status
        order = 'DESC' if order == 'DESC' else 'ASC'
        order_by_whitelist = ['id','username','email','subject', 'unread_email','current_client']
        order_by = order_by if order_by in order_by_whitelist else 'id'
        results_per_page = 20
        param1 = (page - 1) * results_per_page
        param2 = results_per_page
        param3 = '%' + search + '%'
        # SQL where clause
        where = '';
        where += 'WHERE (username LIKE %s OR email LIKE %s) ' if search else ''
        # Add filters
        # if status == 'active':
        #       where += 'AND last_seen > date_sub(now(), interval 1 month) ' if where else 'WHERE last_seen > date_sub(now(), interval 1 month) '
        # if status == 'inactive':
        #       where += 'AND last_seen < date_sub(now(), interval 1 month) ' if where else 'WHERE last_seen < date_sub(now(), interval 1 month) '
        # if activation == 'pending':
        #       where += 'AND activation_code != "activated" ' if where else 'WHERE activation_code != "activated" '
        # if role:
        #       where += 'AND role = %s ' if where else 'WHERE role = %s '
        # Params array and append specified params
        params = []
        if search:
                params.append(param3)
                params.append(param3)
        # if role:
        #       params.append(role)
        # Fetch the total number of inquiries
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # unread email check
        data = request.form

        if request.method == 'POST' and request.form:
                cursor.execute('UPDATE contactus SET unread_email = False')
                mysql.connection.commit()
                for key, value in data.items():
                        print(f'key : {key} value:{value}')
                        if 'on' in request.form.getlist(key):
                                value = True
                                cursor.execute('UPDATE contactus SET unread_email = %s WHERE id = %s', (value, key))
                                mysql.connection.commit()

        cursor.execute('SELECT COUNT(*) AS total FROM contactus ' + where, params)
        inquiries_total = cursor.fetchone()

        # Append params to array
        params.append(param1)
        params.append(param2)
    # Retrieve all inquiries from the database
        cursor.execute('SELECT * FROM contactus ' + where + ' ORDER BY ' + order_by + ' ' + order + ' LIMIT %s,%s', params)
        inquiries = cursor.fetchall()

        # Determine the URL
        url = url_for('admin_inquiries') + '/n0/' + (search if search else 'n0') + '/' + (status if status else 'n0')
        # # Handle output messages

        return render_template('admin/inquiries.html', inquiries=inquiries, selected='inquiries', selected_child='view', search=search, status=status, order=order, order_by=order_by, page=page, results_per_page=results_per_page, inquiries_total=inquiries_total['total'], math=math)


# http://localhost:5000/pythonlogin/admin/accounts/delete/<id> - delete account
@app.route('/pythonlogin/admin/inquiries/delete/<int:id>', methods=['GET', 'POST'])
@app.route('/pythonlogin/admin/inquiries/delete', methods=['GET', 'POST'], defaults={'id': None})
def admin_delete_inquiries(id):
        # Check if admin is logged-in
        if not admin_loggedin():
                return redirect(url_for('login'))
        # Set the database connection cursor
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # Delete account from database by the id get request param
        cursor.execute('DELETE FROM contactus WHERE id = %s', (id,))
        mysql.connection.commit()
        # Redirect to accounts page and output message
        return redirect(url_for('admin_inquiries'))

# http://localhost:5000/pythonlogin/admin/roles - view account roles
@app.route('/pythonlogin/admin/roles', methods=['GET', 'POST'])
def admin_roles():
        # Check if admin is logged-in
        if not admin_loggedin():
                return redirect(url_for('login'))
        # Set the connection cursor
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # Select and group roles from the accounts table
        cursor.execute('SELECT role, COUNT(*) as total FROM accounts GROUP BY role')
        roles = cursor.fetchall()
        new_roles = {}
        # Update the structure
        for role in roles:
                new_roles[role['role']] = role['total']
        for role in roles_list:
                if not new_roles[role]:
                        new_roles[role] = 0
        # Get the total number of active roles
        cursor.execute('SELECT role, COUNT(*) as total FROM accounts WHERE last_seen > date_sub(now(), interval 1 month) GROUP BY role')
        roles_active = cursor.fetchall()
        new_roles_active = {}
        for role in roles_active:
                new_roles_active[role['role']] = role['total']
        # Get the total number of inactive roles
        cursor.execute('SELECT role, COUNT(*) as total FROM accounts WHERE last_seen < date_sub(now(), interval 1 month) GROUP BY role')
        roles_inactive = cursor.fetchall()
        new_roles_inactive = {}
        for role in roles_inactive:
                new_roles_inactive[role['role']] = role['total']
        # Render he roles template
        return render_template('admin/roles.html', selected='roles', selected_child='', enumerate=enumerate, roles=new_roles, roles_active=new_roles_active, roles_inactive=new_roles_inactive)


# Dashboard Report page
@app.route('/pythonlogin/admin/sitereport')
def admin_sitereport():
    if not admin_loggedin():
        return redirect(url_for('logout'))

    settings = get_settings()
    # your Spin Rewriter email address goes here
    email_address = settings['SPINWRITER EMAIL']['value']

    # your unique Spin Rewriter API key goes here
    api_key = settings['SPINWRITER KEY']['value']

    # Spin Rewriter API settings - authentication:
    spinrewriter_api = SpinRewriterAPI(email_address, api_key)
    spinrider = spinrewriter_api.get_quota()
    # create context
    # try to get the year
    current_year = request.args.get('year', None)
    # Reset current year to present if year is None or not a 4 digit number
    if not current_year or not current_year.isdigit() or len(current_year.strip()) != 4:
        current_year = datetime.datetime.now().year
    else:
        current_year = int(current_year)

    context = {}
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # User who bought the book
    cursor.execute("""SELECT p.price AS plan, COUNT(i.price) AS total FROM planlist as p, invoice as i WHERE p.price=i.price AND i.status='paid' AND YEAR(i.updated_at)=%s GROUP BY plan""", (current_year,))
    today_result = cursor.fetchall()
    today_result = {p['plan']: p['total'] for p in today_result }

    # last month result
    last_month = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime(dt_format)
    cursor.execute("""SELECT p.price AS plan, COUNT(i.price) AS total FROM planlist as p, invoice as i WHERE p.price=i.price AND i.created_at < %s AND i.status='paid' GROUP BY plan""", (last_month,))
    last_month_result = cursor.fetchall()
    last_month_result = { p['plan']: p['total'] for p in last_month_result}

    totalplan = [35, 45, 55, 65]

    plan_status = {}

    for p in totalplan:
        today_total = today_result[p] if p in today_result else 0
        lastmonth_total = last_month_result[p] if p in last_month_result else 0
        if today_total >= lastmonth_total:
            textcolor = 'success'
            if today_total == 0:
                percent = 0
            else:
                if lastmonth_total == 0:
                    percent = today_total
                else:
                    percent = (today_total - lastmonth_total) / lastmonth_total

        else:
            textcolor = 'warning'
            percent = 0


        plan_status[p] = {
            'total' : today_total,
            'textcolor': textcolor,
            'percent' : percent
        }

    # total sale
    context['plan_status'] = plan_status
    total_sale = 0
    for p in plan_status:
        total_sale += p * plan_status[p]['total']


    context['total_sale'] = total_sale

    # total active user the membership is already purchased
    today = datetime.datetime.now().strftime(dt_format)
    cursor.execute('SELECT COUNT(*) FROM membership WHERE expire_at > %s', (today,))
    total_active = cursor.fetchone()['COUNT(*)']
    context['total_active'] = total_active

    # total sale group by month
    cursor.execute('''SELECT s.smonth AS smonth, COUNT(s.smonth) AS total FROM (
        SELECT MONTHNAME(created_at) AS smonth FROM invoice WHERE status=%s AND YEAR(created_at)=%s )
         AS s GROUP BY s.smonth''', ('paid',current_year,))
    sales_numbers = cursor.fetchall()
    sales_numbers = { p['smonth'][:3]: p['total'] for p in sales_numbers}
    # total emails group by month
    cursor.execute('''SELECT e.emonth AS emonth, COUNT(e.emonth) AS total FROM (
        SELECT MONTHNAME(promo_date) AS emonth from front_page_promo WHERE YEAR(promo_date)=%s)
        AS e GROUP BY e.emonth''',(current_year,))
    totals_email = cursor.fetchall()

    totals_email = {t['emonth'][:3]: t['total'] for t in totals_email}

    # Total Number of Users Active

    monthlist = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_salelist = []
    monthly_emaillist =[]
    for m in monthlist:
        if m in totals_email:
            monthly_emaillist.append(totals_email[m])
        else:
            monthly_emaillist.append(0)

        if m[:3] in sales_numbers:
            monthly_salelist.append(sales_numbers[m])
        else:
            monthly_salelist.append(0)

    context['monthlist'] = monthlist
    context['monthly_salelist'] = monthly_salelist
    context['monthly_emaillist'] = monthly_emaillist
    context['current_year'] = current_year
    # year list queyr
    cursor.execute('''SELECT YEAR(updated_at) as year from invoice group by YEAR(updated_at)''')
    yearlist = cursor.fetchall()

    year_list = []
    for i in yearlist:
        year_list.append(i['year'])

    context['year_list'] = year_list
    print(context)

    # total number of guest user
    guest_qs = """SELECT g.gmonth as gmonth, COUNT(g.gmonth) as total FROM (SELECT MONTHNAME(updated_at) AS gmonth from pagevisit where is_login=%s AND YEAR(updated_at)=%s) AS g GROUP BY g.gmonth """


    # query the not login user
    cursor.execute(guest_qs, (0, current_year,))
    total_guest  = cursor.fetchall()
    total_guest = {g['gmonth'][:3]:g['total'] for g in total_guest }

    # an empty array
    monthly_guestlist = []
    for m in monthlist:
        if m in total_guest:
            monthly_guestlist.append(total_guest[m])
        else:
            monthly_guestlist.append(0)

    context['monthly_guestlist'] = monthly_guestlist

    return render_template('admin/reports.html', selected='reports', segment='index', context=context, spinrider=spinrider)

# http://localhost:5000/pythonlogin/admin/settings - manage settings
@app.route('/pythonlogin/admin/settings/<string:msg>', methods=['GET', 'POST'])
@app.route('/pythonlogin/admin/settings', methods=['GET', 'POST'], defaults={'msg': ''})
def admin_settings(msg):
        # Check if admin is logged-in
        if not admin_loggedin():
                return redirect(url_for('login'))
        # Get settings
        settings = get_settings()
        # Set the connection cursor
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # If user submitted the form
        if request.method == 'POST' and request.form:
                # Retrieve the form data
                data = request.form
                # Iterate the form data
                for key, value in data.items():
                        if 'true' in request.form.getlist(key):
                                value = 'true'
                        # Convert boolean values to lowercase
                        value = value.lower() if value.lower() in ['true', 'false'] else value
                        # Update setting
                        cursor.execute('UPDATE settings SET setting_value = %s WHERE setting_key = %s', (value,key,))
                        mysql.connection.commit()
                # Redirect and output message
                return redirect(url_for('admin_settings', msg='msg1'))
        # Handle output messages
        if msg and msg == 'msg1':
                msg = 'Settings updated successfully!';
        else:
                msg = ''
        # Render the settings template
        return render_template('admin/settings.html', selected='settings', selected_child='', msg=msg, settings=settings, settings_format_tabs=settings_format_tabs, settings_format_form=settings_format_form)


# http://localhost:5000/pythonlogin/admin/accounts/delete/<id> - delete account
@app.route('/pythonlogin/admin/accounts/delete/<int:id>', methods=['GET', 'POST'])
@app.route('/pythonlogin/admin/accounts/delete', methods=['GET', 'POST'], defaults={'id': None})
def admin_delete_account(id):
        # Check if admin is logged-in
        if not admin_loggedin():
                return redirect(url_for('login'))
        # Set the database connection cursor
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # Delete account from database by the id get request param
        cursor.execute('DELETE FROM accounts WHERE id = %s', (id,))
        mysql.connection.commit()
        # Redirect to accounts page and output message
        return redirect(url_for('admin_accounts', msg='msg3', activation='n0', order='id', order_by='DESC', page=1, role='n0', search='n0', status='n0'))

# http://localhost:5000/pythonlogin/admin/account/<optional:id> - create or edit account
@app.route('/pythonlogin/admin/account/<int:id>', methods=['GET', 'POST'])
@app.route('/pythonlogin/admin/account', methods=['GET', 'POST'], defaults={'id': None})
def admin_account(id):
    # Check if admin is logged-in
    if not admin_loggedin():
        return redirect(url_for('login'))
        # Default page (Create/Edit)
    page = 'Create'
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # Default input account values
    account = {
        'username': '',
        'password': '',
        'email': '',
        'activation_code': '',
        'rememberme': '',
        'role': 'Member',
                'registered': str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'last_seen': str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    }
    roles = ['Member', 'Admin']
    # GET request ID exists, edit account
    if id:
        # Edit an existing account
        page = 'Edit'
        # Retrieve account by ID with the GET request ID
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (id,))
        account = cursor.fetchone()
                # If user submitted the form
        if request.method == 'POST' and 'submit' in request.form:
            # update account
            password = account['password']
                        # If password exists in POST request
            if request.form['password']:
                 hash = request.form['password'] + app.secret_key
                 hash = hashlib.sha1(hash.encode())
                 password = hash.hexdigest();
                        # Update account details
            cursor.execute('UPDATE accounts SET username = %s, password = %s, email = %s, activation_code = %s, rememberme = %s, role = %s, registered = %s, last_seen = %s WHERE id = %s', (request.form['username'],password,request.form['email'],request.form['activation_code'],request.form['rememberme'],request.form['role'],request.form['registered'],request.form['last_seen'],id,))
            mysql.connection.commit()
                        # Redirect to admin accounts page
            return redirect(url_for('admin_accounts', msg='msg2', activation='n0', order='id', order_by='DESC', page=1, role='n0', search='n0', status='n0'))
        if request.method == 'POST' and 'delete' in request.form:
            # delete account
            return redirect(url_for('admin_delete_account', id=id))
    if request.method == 'POST' and request.form['submit']:
        # Create new account, hash password
        hash = request.form['password'] + app.secret_key
        hash = hashlib.sha1(hash.encode())
        password = hash.hexdigest();
                # Insert account into database
        cursor.execute('INSERT INTO accounts (username,password,email,activation_code,rememberme,role,registered,last_seen) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)', (request.form['username'],password,request.form['email'],request.form['activation_code'],request.form['rememberme'],request.form['role'],request.form['registered'],request.form['last_seen'],))
        mysql.connection.commit()
                # Redirect to the admin accounts page and output message
        return redirect(url_for('admin_accounts', msg='msg1', activation='n0', order='id', order_by='DESC', page=1, role='n0', search='n0', status='n0'))
        # Render the admin account template
    return render_template('admin/account.html', account=account, selected='accounts', selected_child='manage', page=page, roles=roles, datetime=datetime.datetime, str=str)

# http://localhost:5000/pythonlogin/admin/emailtemplate - admin email templates page, manage email templates
@app.route('/pythonlogin/admin/emailtemplate/<string:msg>', methods=['GET', 'POST'])
@app.route('/pythonlogin/admin/emailtemplate', methods=['GET', 'POST'], defaults={'msg': ''})
def admin_emailtemplate(msg):
        # Check if admin is logged-in
        if not admin_loggedin():
                return redirect(url_for('login'))
        # Get the template directory path
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        print(f"template_dir : {template_dir}")
        # Update the template file on save
        if request.method == 'POST':
                # Update activation template
                activation_email_template = request.form['activation_email_template'].replace('\r', '')
                open(template_dir + '/activation-email-template.html', mode='w', encoding='utf-8').write(activation_email_template)
                # Update twofactor template
                twofactor_email_template = request.form['twofactor_email_template'].replace('\r', '')
                open(template_dir + '/twofactor-email-template.html', mode='w', encoding='utf-8').write(twofactor_email_template)
                # Update Rankgin Email template
                ranking_email_template = request.form['ranking_email_template'].replace('\r', '')
                open(template_dir + '/ranking-email-template.html', mode='w', encoding='utf-8').write(ranking_email_template)
                # Redirect and output success message
                return redirect(url_for('admin_emailtemplate', msg='msg1'))
        # Read the activation email template
        activation_email_template = open(template_dir + '/activation-email-template.html', mode='r', encoding='utf-8').read()
        # Read the twofactor email template
        twofactor_email_template = open(template_dir + '/twofactor-email-template.html', mode='r', encoding='utf-8').read()
        # Read the twofactor email template
        ranking_email_template = open(template_dir + '/ranking-email-template.html', mode='r', encoding='utf-8').read()
        # Handle output messages
        if msg and msg == 'msg1':
                msg = 'Email templates updated successfully!';
        else:
                msg = ''
        # Render template
        return render_template('admin/emailtemplates.html', selected='emailtemplate', selected_child='', msg=msg, activation_email_template=activation_email_template, twofactor_email_template=twofactor_email_template, ranking_email_template=ranking_email_template)

# Admin logged-in check function
def admin_loggedin():
    if loggedin() and session['role'] == 'Admin':
        # admin logged-in
        return True
    # admin not logged-in return false
    return False

# format settings key
def settings_format_key(key):
    key = key.lower().replace('_', ' ').replace('url', 'URL').replace('db ', 'Database ').replace(' password', ' Password').replace(' username', ' Username')
    return key.title()


# Format settings variables in HTML format
def settings_format_var_html(key, value):
        html = ''
        type = 'text'
        type = 'password' if 'pass' in key else type
        type = 'checkbox' if value.lower() in ['true', 'false'] else type
        checked = ' checked' if value.lower() == 'true' else ''
        html += '<label for="' + key + '">' + settings_format_key(key) + '</label>'
        if (type == 'checkbox'):
                html += '<input type="hidden" name="' + key + '" value="false">'
        html += '<input type="' + type + '" name="' + key + '" id="' + key + '" value="' + value + '" placeholder="' + settings_format_key(key) + '"' + checked + '>'
        return html

# Format settings tabs
def settings_format_tabs(tabs):
        html = ''
        html += '<div class="tabs">'
        html += '<a href="#" class="active">General</a>'
        for tab in tabs:
                html += '<a href="#">' + tab + '</a>'
        html += '</div>'
        return html

# Format settings form
def settings_format_form(settings):
        html = ''
        html += '<div class="tab-content active">'
        category = ''
        for setting in settings:
                if category != '' and category != settings[setting]['category']:
                        html += '</div><div class="tab-content">'
                category = settings[setting]['category']
                html += settings_format_var_html(settings[setting]['key'], settings[setting]['value'])
        html += '</div>'
        return html

# Get settings from database
def get_settings():
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM settings ORDER BY id')
        settings = cursor.fetchall()
        settings2 = {}
        for setting in settings:
                settings2[setting['setting_key']] = { 'key': setting['setting_key'], 'value': setting['setting_value'], 'category': setting['category'] }
        return settings2

# Format datetime
def time_elapsed_string(dt):
        d = datetime.datetime.strptime(str(dt), '%Y-%m-%d %H:%M:%S')
        dd = datetime.datetime.now()
        d = d.timestamp() - dd.timestamp()
        d = datetime.timedelta(seconds=d)
        timeDelta = abs(d)
        if timeDelta.days > 0:
                if timeDelta.days == 1:
                        return '1 day ago'
                else:
                        return '%s days ago' % timeDelta.days
        elif round(timeDelta.seconds / 3600) > 0:
                if round(timeDelta.seconds / 3600) == 1:
                        return '1 hour ago'
                else:
                        return '%s hours ago' % round(timeDelta.seconds / 3600)
        elif round(timeDelta.seconds / 60) < 2:
                return '1 minute ago'
        else:
                return '%s minutes ago' % round(timeDelta.seconds / 60)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

#Logging
@app.route('/')
def main():
        # showing different logging levels
        app.logger.debug("debug log info")
        app.logger.info("Info log information")
        app.logger.warning("Warning log info")
        app.logger.error("Error log info")
        app.logger.critical("Critical log info")
        print("logging success")
        return "testing logging levels."

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=2430)
    #app.run(host='172.31.40.33', debug=False, port=2430)
    #app.run()
