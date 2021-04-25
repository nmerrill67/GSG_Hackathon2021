#!/usr/bin/env python3
from textblob import TextBlob
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from spamdetector import CommentClassifier
# install textblob https://textblob.readthedocs.io/en/dev/install.html


input_file = 'data/reddit_data/CryptoCurrency_Reddit042421_9pm_BTC.csv'
output_file = 'data/reddit_data/Cleaned_CryptoCurrency_Reddit042421_9pm_BTC.csv'

# input_file = 'data/reddit_data/CryptoCurrency_Reddit042421_9pm_ETH.csv'
# output_file = 'data/reddit_data/Cleaned_CryptoCurrency_Reddit042421_9pm_ETH.csv'

# input_file = 'data/reddit_data/CryptoCurrency_Reddit042421_9pm_DOGE.csv'
# output_file = 'data/reddit_data/Cleaned_CryptoCurrency_Reddit042421_9pm_DOGE.csv'

# input_file = 'CryptoCurrency_Reddit042421_9pm_BTC.csv'
# output_file = 'Cleaned_CryptoCurrency_Reddit042421_9pm_BTC_w_spam_feature.csv'
#
# # input_file = 'CryptoCurrency_Reddit042421_9pm_ETH.csv'
# # output_file = 'Cleaned_CryptoCurrency_Reddit042421_9pm_ETH_w_spam_feature.csv'
#
# # input_file = 'CryptoCurrency_Reddit042421_9pm_DOGE.csv'
# # output_file = 'Cleaned_CryptoCurrency_Reddit042421_9pm_DOGE_w_spam_feature.csv'


input_file = 'CryptoCurrency_Reddit042421_9pm_BTC.csv'
output_file = 'Cleaned_CryptoCurrency_Reddit042421_9pm_BTC_w_spam_feature.csv'

# input_file = 'CryptoCurrency_Reddit042421_9pm_ETH.csv'
# output_file = 'Cleaned_CryptoCurrency_Reddit042421_9pm_ETH_w_spam_feature.csv'

# input_file = 'CryptoCurrency_Reddit042421_9pm_DOGE.csv'
# output_file = 'Cleaned_CryptoCurrency_Reddit042421_9pm_DOGE_w_spam_feature.csv'


md = pd.read_csv(input_file)
emoji_pattern = re.compile("["u"\U000000A0-\U0001FA90""]+", flags=re.UNICODE)

def rm_punc(text):
    return ' '.join([w for w in word_tokenize(text) if w not in string.punctuation + '“”’‘-——…'])

def rm_stops(text):
    return ' '.join([w for w in word_tokenize(text) if w not in set(stopwords.words('english'))])

def clean_text(text):

    # remove url
    clean_post = re.sub(r'http\S+', '', text)

    # remove emojis
    clean_post = emoji_pattern.sub(r'', clean_post)

    # convert to lowercase
    clean_post = clean_post.lower()

    # remove punctuation
    clean_post = clean_post.translate(str.maketrans('', '', string.punctuation))
    clean_post = rm_punc(clean_post)

    word_count = len(word_tokenize(clean_post))

    # remove stop words
    clean_post = rm_stops(clean_post)

    return word_count, clean_post


polarity_list = []
subjectivity_list = []
word_count_list = []
drop_row_list = []
is_spam_list = []

check_spam = CommentClassifier()
check_spam.load('nb') # use Naive Bayes BTC 1354 spam
#check_spam.load('logit') # 1354
#check_spam.load('rf') # 1354

for ind in md.index:
    post=md['Text'][ind]
    word_count, clean_post = clean_text(post)

    # remove if less than 3 words
    if word_count < 3:
        drop_row_list.append(ind)



    md['Text'][ind] = clean_post
    blob = TextBlob(post)
    Sentiment = blob.sentiment
    # seperate polarity and subjectivity in to two variables
    polarity_list.append(Sentiment.polarity)
    subjectivity_list.append(Sentiment.subjectivity)
    word_count_list.append(word_count)

    # classify if its spam
    is_spam_list.append(int(check_spam.is_spam(clean_post)))  # True (is spam) - 1, False (not spam) - 0

# append to data frame
md['polarity'] = polarity_list
md['subjectivity'] = subjectivity_list
md['word_count'] = word_count_list
md['is_spam'] = is_spam_list

# drop row if word less than 5
clean_md = md.drop(drop_row_list)

# save pd as csv
#clean_md.to_csv(output_file)

print("total spam", sum(is_spam_list))

# BTC (2989, 9), 1354 Spam - 45.30%, 1635 non spam
# ETH (3382, 9), 1563 spam - 46.21%, 1819 non spam
# DOGE (3270, 9), 1312 spam - 40.12%, 1958 non spam



