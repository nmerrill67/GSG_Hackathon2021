#!/usr/bin/env python3
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import *
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import copy
seed = 666
np.random.seed(seed)
torch.random.manual_seed(seed)

class CryptoNN(nn.Module):
    def __init__(self, num_feats):
        super().__init__()
        self.fc1 = nn.Linear(num_feats, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        if self.training:
            x = F.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        if self.training:
            x = F.dropout(x)
        x = self.fc3(x)
        return x

class CryptoDataset(data.Dataset):

    def __init__(self, crypto_data, reddit_data, split):
        self.split = split
        assert split in ['train', 'test']
        self.reddit_data = reddit_data
        split_point = int(round(.75 * len(crypto_data['unix_time'])))
        self.crypto_data = {}
        for k,v in crypto_data.items():
            self.crypto_data[k] = v[:split_point] if self.split=='train' else v[split_point:]
        assert self.__len__() > 0

    # Length of dataset. -1 since we need the future data as label
    def __len__(self):
        return len(self.crypto_data['unix_time'])-1

    def __getitem__(self, i):
        
        # Construct the feature vector from aggregate comment data
        
        t_market = self.crypto_data["unix_time"][i]
        ts_comments = np.array(self.reddit_data['UTC']).astype(np.int)
        scores = np.array(eth_comments['Upvote Ratio'])
        # Boolean indexing of comment data rows. The data within the 
        # 30-min block
        comment_inds = np.logical_and(ts_comments <= t_market,
                                      ts_comments > t_market-30*60)
        
        # If training, take a random subset for data augmentation
        #if self.split == 'train':
        #    keep_prob = np.random.uniform(.1, 1)
        #    keep = np.random.rand(*comment_inds.shape) <= keep_prob
        #    comment_inds = np.logical_and(comment_inds, keep)

        feats = []
        # Now construct the feature vector
        # Total num comments
        num_comments = np.count_nonzero(comment_inds)
        #feats.append(num_comments)

        # Number of comments with >0 vote score
        scores = np.array(self.reddit_data['Upvote Ratio'])
        num_score = np.count_nonzero(scores[comment_inds] > 0)
        feats.append(num_score)

        # Number of comments with >0.0 polarity
        scores = np.array(self.reddit_data['polarity'])
        num_pol = np.count_nonzero(scores[comment_inds] > 0.0)
        feats.append(num_pol)
        
        # Number of comments with >0.5 subjectivity
        scores = np.array(self.reddit_data['subjectivity'])
        num_subj = np.count_nonzero(scores[comment_inds] > 0.5)
        feats.append(num_subj)
    
        # Number of comments mentioning "buy"
        comments = np.array(self.reddit_data['Text'])[comment_inds]
        num_buy = 0
        for j in range(len(comments)):
            if 'buy' in str(comments[j]).lower() or 'bought' in str(comments[j]).lower():
                num_buy += 1
        feats.append(num_buy)
        
        # Number of comments mentioning "sell"
        comments = np.array(self.reddit_data['Text'])[comment_inds]
        num_sell = 0
        for j in range(len(comments)):
            if 'sell' in str(comments[j]).lower() or 'sold' in str(comments[j]).lower():
                num_buy += 1
        feats.append(num_sell)

        # The historical data for this timeslot as input too
        start = self.crypto_data["open"][i]
        close = self.crypto_data["close"][i]
        high = self.crypto_data["high"][i]
        low = self.crypto_data["low"][i]
        feats += [start, close, high, low]

        # Num feats is 9

        # Now, get the target by seeing if the next closing price is lower
        target = int(self.crypto_data["close"][i+1] < close)

        # Now stack the feature vector, normalize and return
        feats = np.array(feats).astype(np.float32)

        # If training, apply a random mult to do data augmentation.
        if self.split == 'train':
            feats *= np.random.uniform(0.9, 1.1, size=feats.shape)
        
        # Normalize the comment numbers to percents and return
        feats[:-4] = np.minimum(feats[:-4], num_comments*np.ones_like(feats[:-4]))
        feats[:-4] = feats[:-4] / max(1,num_comments)
        #print(feats)
        return feats, target

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

    ##############################################################################
    ## Setup the dataset and net and train #######################################
    ##############################################################################
        
    dataset_train = CryptoDataset(btc_usd, btc_comments, 'train')
    dataset_test = CryptoDataset(btc_usd, btc_comments, 'test')
    
    #dataset_train = CryptoDataset(doge_usd, doge_comments, 'train')
    #dataset_test = CryptoDataset(doge_usd, doge_comments, 'test')
    
    #dataset_train = CryptoDataset(eth_usd, eth_comments, 'train')
    #dataset_test = CryptoDataset(eth_usd, eth_comments, 'test')
 
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

    model = CryptoNN(9) # See CryptoDataset for num_feats
    if torch.cuda.is_available():
        model = model.cuda()

    opt = torch.optim.Adam(model.parameters(), 1e-3)

    num_epochs = 1000
    model_best = None
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for i, (x, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                x, target = x.cuda(), target.cuda()
            
            # Zero grad and forward pass
            opt.zero_grad()
            logits = model(x)

            # Calculate loss
            loss = F.cross_entropy(logits, target)
            #loss = F.binary_cross_entropy_with_logits(logits, 
            #        target[:,None].float())
            loss.backward() # compute gradient and do optim step
            opt.step()

            print(f"Epoch {epoch} step {i}: loss={loss:.3f}")
        
        model.eval()
        print("Epoch finished. Testing model")
        # Evaluate
        acc = 0.0
        for i, (x, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                x, target = x.cuda(), target.cuda()
            #print(x)
            logits = model(x)
            prob = F.softmax(logits) # [p_rise_or_stay_same, p_fall]
            pred = torch.argmax(prob) # 0 or 1 for rise/fall
            #pred = F.sigmoid(model(x))
            acc += float(int(round(pred.cpu().item())) == target.cpu().item())

        acc = acc / len(test_loader)
        print(f"Test accuracy = {acc:.3f}\n")
        # Save the best model
        if acc > best_acc:
            best_acc = acc
            model_best = copy.deepcopy(model)

    print("====== DONE ==================")
    print(f"Best test accuracy = {best_acc:.3f}\n")





