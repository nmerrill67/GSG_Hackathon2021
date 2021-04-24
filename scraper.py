#!/usr/bin/env python3

from argparse import ArgumentParser
import praw
import json
import datetime

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

    hot_posts = reddit.subreddit('CryptoCurrency').hot(limit=2000)
    i = 0
    for post in hot_posts:
        #print(post.title)
        #print("Attrs:")
        #for attr in dir(post):
        #    print(attr)
        print(datetime.datetime.fromtimestamp(post.created_utc))
        i += 1
    print("Number of posts:", i)
