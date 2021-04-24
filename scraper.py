#!/usr/bin/env python3

from argparse import ArgumentParser
import praw
import json
import datetime
import csv
import time
filename = 'CryptoCurrency_Reddit042421.csv'
start = time.time()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-f', '--in-file', default='client_info.json', type=str, metavar='FILE',
            help='Json file with keys ID and SECRET which contain the necessary strings. '
            'The file should contain {"ID": "CLIENT_ID", "SECRET": "CLIENT_SECRET"}')
    args = parser.parse_args()

    with open(args.in_file, 'r') as f:
        data = json.load(f)

    assert data.get('SECRET',None) is not None, \
            "Please provide client secret as key: SECRET in json file!" + \
            " See https://towardsdatascience.com/scraping-reddit-data-1c0af3040768"
    assert data.get('ID',None) is not None, \
            "Please provide client secret as key: ID in json file!" + \
            " See https://towardsdatascience.com/scraping-reddit-data-1c0af3040768"

    # See: https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
    # NOTE: No username/password for public comments
    reddit = praw.Reddit(
        user_agent="CommentScraper",
        client_id=data['ID'],
        client_secret=data['SECRET'],
    )

    with open(filename, mode='w') as cryoto_file:
        cryoto_writer = csv.writer(cryoto_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        hot_posts = reddit.subreddit('CryptoCurrency').hot(limit=1000)
        i = 0
        for post in hot_posts:
        #print(post.title)
        #print("Attrs:")
        # for attr in dir(post):
        #    print(attr)
            i += 1
            post_time = datetime.datetime.fromtimestamp(post.created_utc)
            utc = post.created_utc
            text = post.title
            num_comments = post.num_comments
            upvote_ratio = post.upvote_ratio
            cryoto_writer.writerow([utc, post_time, num_comments, upvote_ratio, text])

    print("Number of posts:", i)

print("TIME SPENT: "+ str(time.time()-start))
