import optparse, sys, os, logging
from surprise import SVD
from surprise import dataset
from surprise import Dataset
from surprise import evaluate, print_perf,Reader
import pandas as pd
import csv

# Loading the trainset
df = pd.read_csv('train_rating.txt')
# The Reader class is used to parse a file containing ratings - one rating per line with a fixed structure
reader = Reader(rating_scale=(0, 5),sep=',')
df.drop('train_id',1)
df.drop('date',1)
data = Dataset.load_from_df(df[['user_id', 'business_id', 'rating']], reader)
# Building the trainset
trainset=data.build_full_trainset()
# Training the algorithm: We'll use the famous SVD algorithm: This is where the various parameters need to be passed
algo = SVD(n_factors=101,biased=True)
algo.train(trainset)
# Building the test-set. In this case, it is the train_set itself, with the predictions removed
testset=trainset.build_testset()
# Making the predictions
predictions=algo.test(testset)
# Printing the predictions into a file with submission format
with open('predicted_rating.csv', 'w') as csvfile:
    fieldnames = ['test_id', 'rating']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')
    writer.writeheader()
    for idx, row in enumerate(predictions):
        print(row[3])
        writer.writerow({'test_id': idx, 'rating': row[3]})








