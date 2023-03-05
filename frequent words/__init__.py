import frequent_words
import transparent
from datetime import datetime
from dateutil.relativedelta import relativedelta

one_yr_ago = datetime.now() - relativedelta(years=1)

frequent_words.get_frequencies_word_cloud_since(one_yr_ago) # get frequencies of last one year's books
frequent_words.get_frequencies_word_cloud_since_start() # get frequencies of all books in the database

PATH1 = '/home/ec2-user/site/static/images/one_year_titles.png'
PATH2 = '/home/ec2-user/site/static/images/all_titles.png'

#transparent.convertImage(PATH1)
#transparent.convertImage(PATH2)