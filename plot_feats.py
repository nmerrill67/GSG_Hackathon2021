#!/usr/bin/env python3
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import *
import pandas as pd

if __name__ == '__main__':

    # Read crypto stock data
    crypto_data_dir = 'data/hist_data'
    btc_usd_fl = os.path.join(crypto_data_dir, 'Poloniex_BTCUSDT_1h.csv')
    doge_btc_fl = os.path.join(crypto_data_dir, 'Poloniex_DOGEBTC_1h.csv')
    eth_btc_fl = os.path.join(crypto_data_dir, 'Poloniex_ETHBTC_1h.csv')

    not_before = 1618963200 # 4/20/2021 midnight

    btc_usd = read_crypto_csv(btc_usd_fl, not_before=not_before)
    doge_usd = read_crypto_csv(doge_btc_fl, btc_to_usd=btc_usd, not_before=not_before)
    eth_usd = read_crypto_csv(eth_btc_fl, btc_to_usd=btc_usd, not_before=not_before)
    
    
    # Read the reddit data
    reddit_data_dir = 'data/reddit_data'
    btc_r_fl = os.path.join(reddit_data_dir, 'Cleaned_CryptoCurrency_Reddit042421_9pm_BTC.csv')
    doge_r_fl = os.path.join(reddit_data_dir, 'Cleaned_CryptoCurrency_Reddit042421_9pm_DOGE.csv')
    eth_r_fl = os.path.join(reddit_data_dir, 'Cleaned_CryptoCurrency_Reddit042421_9pm_ETH.csv')

    btc_comments = pd.read_csv(btc_r_fl, delimiter=',')
    doge_comments = pd.read_csv(doge_r_fl, delimiter=',')
    eth_comments = pd.read_csv(eth_r_fl, delimiter=',')

    ## PLOT NUM COMMENTS ########################################################
    # Get the number of coments for every 30 min, corresponding to the 
    # crypto market times
    btc_num_comments = np.zeros((len(btc_usd["unix_time"])), dtype=np.int)
    doge_num_comments = np.zeros((len(doge_usd["unix_time"])), dtype=np.int)
    eth_num_comments = np.zeros((len(eth_usd["unix_time"])), dtype=np.int)

    for i in range(len(btc_usd["unix_time"])):
        t_market = btc_usd["unix_time"][i]
        ts_comments = np.array(btc_comments['UTC']).astype(np.int)
        btc_num_comments[i] = np.count_nonzero(np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60))
    
    for i in range(len(doge_usd["unix_time"])):
        t_market = doge_usd["unix_time"][i]
        ts_comments = np.array(doge_comments['UTC']).astype(np.int)
        doge_num_comments[i] = np.count_nonzero(np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60))

    for i in range(len(eth_usd["unix_time"])):
        t_market = eth_usd["unix_time"][i]
        ts_comments = np.array(eth_comments['UTC']).astype(np.int)
        eth_num_comments[i] = np.count_nonzero(np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60))


    # Plot Num comments vs market price
    plt.subplot(311)
    plt.scatter(btc_usd["close"], btc_num_comments,label='BTC', color='r')
    plt.legend()

    plt.subplot(312)
    plt.scatter(doge_usd["close"], doge_num_comments, label='DOGE', color='g')
    plt.legend()
    plt.ylabel('Number of Comments')

    plt.subplot(313)
    plt.scatter(eth_usd["close"], eth_num_comments, label='ETH', color='b')
    plt.legend()

    plt.xlabel('Closing Price (USD)')
    
    plt.savefig('plots/num_comments_vs_price.png')


    ##############################################################################
    ## Plot number of comments with >0 vote score ############################### 
    ##############################################################################
    btc_num_comments = np.zeros((len(btc_usd["unix_time"])), dtype=np.int)
    doge_num_comments = np.zeros((len(doge_usd["unix_time"])), dtype=np.int)
    eth_num_comments = np.zeros((len(eth_usd["unix_time"])), dtype=np.int)

    for i in range(len(btc_usd["unix_time"])):
        t_market = btc_usd["unix_time"][i]
        ts_comments = np.array(btc_comments['UTC']).astype(np.int)
        scores = np.array(btc_comments['Upvote Ratio'])
        btc_num_comments[i] = np.count_nonzero(
                scores[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)] > 0)
    
    for i in range(len(doge_usd["unix_time"])):
        t_market = doge_usd["unix_time"][i]
        ts_comments = np.array(doge_comments['UTC']).astype(np.int)
        scores = np.array(doge_comments['Upvote Ratio'])
        doge_num_comments[i] = np.count_nonzero(
                scores[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)] > 0)

    for i in range(len(eth_usd["unix_time"])):
        t_market = eth_usd["unix_time"][i]
        ts_comments = np.array(eth_comments['UTC']).astype(np.int)
        scores = np.array(eth_comments['Upvote Ratio'])
        eth_num_comments[i] = np.count_nonzero(
                scores[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)] > 0)

    plt.close()

    # Plot Num comments vs market price
    plt.subplot(311)
    plt.scatter(btc_usd["close"], btc_num_comments,label='BTC', color='r')
    plt.legend()

    plt.subplot(312)
    plt.scatter(doge_usd["close"], doge_num_comments, label='DOGE', color='g')
    plt.legend()
    plt.ylabel('Number of Comments with >0 Score')

    plt.subplot(313)
    plt.scatter(eth_usd["close"], eth_num_comments, label='ETH', color='b')
    plt.legend()

    plt.xlabel('Closing Price (USD)')
    
    plt.savefig('plots/num_pos_score_vs_price.png')

    ##############################################################################
    ## Plot number of comments with <=0 vote score ############################### 
    ##############################################################################
    btc_num_comments = np.zeros((len(btc_usd["unix_time"])), dtype=np.int)
    doge_num_comments = np.zeros((len(doge_usd["unix_time"])), dtype=np.int)
    eth_num_comments = np.zeros((len(eth_usd["unix_time"])), dtype=np.int)

    for i in range(len(btc_usd["unix_time"])):
        t_market = btc_usd["unix_time"][i]
        ts_comments = np.array(btc_comments['UTC']).astype(np.int)
        scores = np.array(btc_comments['Upvote Ratio'])
        btc_num_comments[i] = np.count_nonzero(
                scores[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)] <= 0)
    
    for i in range(len(doge_usd["unix_time"])):
        t_market = doge_usd["unix_time"][i]
        ts_comments = np.array(doge_comments['UTC']).astype(np.int)
        scores = np.array(doge_comments['Upvote Ratio'])
        doge_num_comments[i] = np.count_nonzero(
                scores[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)] <= 0)

    for i in range(len(eth_usd["unix_time"])):
        t_market = eth_usd["unix_time"][i]
        ts_comments = np.array(eth_comments['UTC']).astype(np.int)
        scores = np.array(eth_comments['Upvote Ratio'])
        eth_num_comments[i] = np.count_nonzero(
                scores[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)] <= 0)

    plt.close()

    # Plot Num comments vs market price
    plt.subplot(311)
    plt.scatter(btc_usd["close"], btc_num_comments,label='BTC', color='r')
    plt.legend()

    plt.subplot(312)
    plt.scatter(doge_usd["close"], doge_num_comments, label='DOGE', color='g')
    plt.legend()
    plt.ylabel('Number of Comments with <=0 Score')

    plt.subplot(313)
    plt.scatter(eth_usd["close"], eth_num_comments, label='ETH', color='b')
    plt.legend()

    plt.xlabel('Closing Price (USD)')
    
    plt.savefig('plots/num_neg_score_vs_price.png')


    ##############################################################################
    ## Plot number of comments with >0.5 subjectivity ############################ 
    ##############################################################################
    btc_num_comments = np.zeros((len(btc_usd["unix_time"])), dtype=np.int)
    doge_num_comments = np.zeros((len(doge_usd["unix_time"])), dtype=np.int)
    eth_num_comments = np.zeros((len(eth_usd["unix_time"])), dtype=np.int)

    for i in range(len(btc_usd["unix_time"])):
        t_market = btc_usd["unix_time"][i]
        ts_comments = np.array(btc_comments['UTC']).astype(np.int)
        scores = np.array(btc_comments['subjectivity'])
        btc_num_comments[i] = np.count_nonzero(
                scores[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)] > 0.5)
    
    for i in range(len(doge_usd["unix_time"])):
        t_market = doge_usd["unix_time"][i]
        ts_comments = np.array(doge_comments['UTC']).astype(np.int)
        scores = np.array(doge_comments['subjectivity'])
        doge_num_comments[i] = np.count_nonzero(
                scores[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)] > 0.5)

    for i in range(len(eth_usd["unix_time"])):
        t_market = eth_usd["unix_time"][i]
        ts_comments = np.array(eth_comments['UTC']).astype(np.int)
        scores = np.array(eth_comments['subjectivity'])
        eth_num_comments[i] = np.count_nonzero(
                scores[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)] > 0.5)

    plt.close()

    # Plot Num comments vs market price
    plt.subplot(311)
    plt.scatter(btc_usd["close"], btc_num_comments,label='BTC', color='r')
    plt.legend()

    plt.subplot(312)
    plt.scatter(doge_usd["close"], doge_num_comments, label='DOGE', color='g')
    plt.legend()
    plt.ylabel('Number of Comments with >0.5 Subjectivity')

    plt.subplot(313)
    plt.scatter(eth_usd["close"], eth_num_comments, label='ETH', color='b')
    plt.legend()

    plt.xlabel('Closing Price (USD)')
    
    plt.savefig('plots/num_subj_vs_price.png')


    ##############################################################################
    ## Plot number of comments mentioning "buy"       ############################ 
    ##############################################################################
    btc_num_comments = np.zeros((len(btc_usd["unix_time"])), dtype=np.int)
    doge_num_comments = np.zeros((len(doge_usd["unix_time"])), dtype=np.int)
    eth_num_comments = np.zeros((len(eth_usd["unix_time"])), dtype=np.int)

    for i in range(len(btc_usd["unix_time"])):
        t_market = btc_usd["unix_time"][i]
        ts_comments = np.array(btc_comments['UTC']).astype(np.int)
        comments = np.array(btc_comments['Text'])[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)]
        for j in range(len(comments)):
            if 'buy' in str(comments[j]).lower() or 'bought' in str(comments[j]).lower():
                btc_num_comments[i] += 1
    
    for i in range(len(doge_usd["unix_time"])):
        t_market = doge_usd["unix_time"][i]
        ts_comments = np.array(doge_comments['UTC']).astype(np.int)
        comments = np.array(doge_comments['Text'])[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)]
        for j in range(len(comments)):
            if 'buy' in str(comments[j]).lower() or 'bought' in str(comments[j]).lower():
                doge_num_comments[i] += 1

    for i in range(len(eth_usd["unix_time"])):
        t_market = eth_usd["unix_time"][i]
        ts_comments = np.array(eth_comments['UTC']).astype(np.int)
        comments = np.array(eth_comments['Text'])[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)]
        for j in range(len(comments)):
            if 'buy' in str(comments[j]).lower() or 'bought' in str(comments[j]).lower():
                eth_num_comments[i] += 1

    plt.close()

    # Plot Num comments vs market price
    plt.subplot(311)
    plt.scatter(btc_usd["close"], btc_num_comments,label='BTC', color='r')
    plt.legend()

    plt.subplot(312)
    plt.scatter(doge_usd["close"], doge_num_comments, label='DOGE', color='g')
    plt.legend()
    plt.ylabel('Number of Comments Mentioning "buy", "bought"')

    plt.subplot(313)
    plt.scatter(eth_usd["close"], eth_num_comments, label='ETH', color='b')
    plt.legend()

    plt.xlabel('Closing Price (USD)')
    
    plt.savefig('plots/num_buys_vs_price.png')
    
    
    ##############################################################################
    ## Plot number of comments mentioning "sell"       ###########################
    ##############################################################################
    btc_num_comments = np.zeros((len(btc_usd["unix_time"])), dtype=np.int)
    doge_num_comments = np.zeros((len(doge_usd["unix_time"])), dtype=np.int)
    eth_num_comments = np.zeros((len(eth_usd["unix_time"])), dtype=np.int)

    for i in range(len(btc_usd["unix_time"])):
        t_market = btc_usd["unix_time"][i]
        ts_comments = np.array(btc_comments['UTC']).astype(np.int)
        comments = np.array(btc_comments['Text'])[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)]
        for j in range(len(comments)):
            if 'sell' in str(comments[j]).lower() or 'sold' in str(comments[j]).lower():
                btc_num_comments[i] += 1
    
    for i in range(len(doge_usd["unix_time"])):
        t_market = doge_usd["unix_time"][i]
        ts_comments = np.array(doge_comments['UTC']).astype(np.int)
        comments = np.array(doge_comments['Text'])[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)]
        for j in range(len(comments)):
            if 'sell' in str(comments[j]).lower() or 'sold' in str(comments[j]).lower():
                doge_num_comments[i] += 1

    for i in range(len(eth_usd["unix_time"])):
        t_market = eth_usd["unix_time"][i]
        ts_comments = np.array(eth_comments['UTC']).astype(np.int)
        comments = np.array(eth_comments['Text'])[np.logical_and(ts_comments <= t_market,
                ts_comments > t_market-30*60)]
        for j in range(len(comments)):
            if 'sell' in str(comments[j]).lower() or 'sold' in str(comments[j]).lower():
                eth_num_comments[i] += 1

    plt.close()

    # Plot Num comments vs market price
    plt.subplot(311)
    plt.scatter(btc_usd["close"], btc_num_comments,label='BTC', color='r')
    plt.legend()

    plt.subplot(312)
    plt.scatter(doge_usd["close"], doge_num_comments, label='DOGE', color='g')
    plt.legend()
    plt.ylabel('Number of Comments Mentioning "sell", "sold"')

    plt.subplot(313)
    plt.scatter(eth_usd["close"], eth_num_comments, label='ETH', color='b')
    plt.legend()

    plt.xlabel('Closing Price (USD)')
    
    plt.savefig('plots/num_sells_vs_price.png')
