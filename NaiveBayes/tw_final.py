import os
from numpy import *
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy import ndimage
from pandas import *
import numpy
# from sklearn.linear_model import LogisticRegression
import operator
from scipy.sparse import *

# os.chdir("c:/Users/Guerzhoy/Desktop/twitter_project")

filename = "training.1600000.processed.noemoticon.csv"

a = read_csv(filename)

a.columns = ["1", "2", "3", "4", "5", "tweet"]


tweets = a["tweet"]
tweets = [t for t in tweets if '#' in t]


numpy.random.seed(20150819)
idx = numpy.random.permutation(len(tweets))
tweets_test = array(tweets)[idx[len(idx)/2:]]
tweets_train = array(tweets)[idx[:len(idx)/2]]



h_counts = {}
w_counts = {}



i = 0
for t in tweets_train:
    if i % 100 == 0:
        print i
    i += 1
    words = [w.lower().strip('!,.') for w in t.split()]
    hashtags = [w.strip('#') for w in words if '#' in w]
    hashtags = [w for w in hashtags if len(w)>1]

    for h in unique(hashtags):
        if h not in h_counts:
            h_counts[h] = 1
        else:
            h_counts[h] += 1

    for w in unique(words):
        if '#' in w:
            continue
        if '@' in w:
            continue
        if w not in w_counts:
            w_counts[w] = 1
        else:
            w_counts[w] += 1



wh_counts = {}
for t in tweets_train:
    if i % 100 == 0:
        print i
    i += 1

    words = [w.lower().strip('!,.') for w in t.split()]
    hashtags = [w.strip('#') for w in words if '#' in w]
    hashtags = [w for w in hashtags if len(w)>1]

    for h in unique(hashtags):
        if h not in wh_counts:
            wh_counts[h] = {}
        for w in unique(words):
            if w not in wh_counts[h]:
                wh_counts[h][w] = 1
            else:
                wh_counts[h][w] += 1


print(len(wh_counts.keys()))



to_consider = 56
to_suggest = 20


selected_hashtags = map(operator.itemgetter(0), sorted(h_counts.items(), key=operator.itemgetter(1))[-to_consider:])
print(selected_hashtags)
exit()
tophashtags = map(operator.itemgetter(0), sorted(h_counts.items(), key=operator.itemgetter(1))[-to_suggest:])

intop = 0
i = 0
for t in tweets_test:
    if i % 100 == 0:
        print i, "intop=", intop


    words = [w.lower().strip('!,.') for w in t.split() if '#' not in w]
    hashtags = [w.strip('#!,.') for w in t.lower().split() if '#' in w]
    if len(set(hashtags).intersection(selected_hashtags)) == 0:
        continue

    i += 1


    selected_probs = {}
    for h in selected_hashtags:
        prob = 0
        for w in words:
            prob += log(0.000000001+wh_counts[h].get(w, 0))-log(h_counts[h])

        selected_probs[h] = prob+5*log(h_counts[h])-log(len(tweets))

    #print "Tweet:", t
    #print "Probs:", sorted(selected_probs.items(), key=operator.itemgetter(1))[-10:]

    topmostprob = map(operator.itemgetter(0), sorted(selected_probs.items(), key=operator.itemgetter(1))[-to_suggest:])

    #intop += len(set(hashtags).intersection(topmostprob)) > 0


    #print t
    #print hashtags, ':', topmostprob

    intop += len(set(hashtags).intersection(topmostprob)) > 0

    # before value 6500 intop= 5125
    # after value 6500 intop= 5478
    # with alpha = 5: 6500 intop= 5648




to_consider = 1000


words = map(operator.itemgetter(0), sorted(w_counts.items(), key=operator.itemgetter(1))[-22000:])

hashtags = map(operator.itemgetter(0), sorted(h_counts.items(), key=operator.itemgetter(1))[6000:])


words_d = {}
i = 0
for w in words:
    words_d[w] = i
    i += 1

hashtags_d = {}
i = 0
for h in hashtags:
    hashtags_d[h] = i
    i += 1




tweets = tweets_train
feat = dok_matrix((len(tweets), len(words)))
outputs = dok_matrix((len(tweets), len(hashtags)))
i = 0
for t in tweets_train:
    if i % 100 == 0:
        print i
    words = [w.lower().strip('!,.') for w in t.split() if w in words_d]
    hashtags = [w.lower().strip('!,.') for w in t.split() if w in hashtags_d]
    words = [w for w in words if '#' not in w and '@' not in w]


    for w in words:
        feat[i,words_d[w]] = 1


    for h in hashtags:
        outputs[i, hashtags_d[h]] = 1
        last_i, last_h = i, h

    i += 1


logreg = sklearn.linear_model.LogisticRegression(C=1e5)

fits = []
for i in range(outputs.shape[1]):
    if sum(outputs[:,i].toarray().T[0]) > 0:
        fits.append(logreg.fit(feat, outputs[:,i].toarray().T[0]))
    else:
        fits.append(None)
