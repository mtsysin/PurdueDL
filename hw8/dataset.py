# from ComputationalGraphPrimer import *
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torchvision.transforms as tvt
from gensim.models import KeyedVectors 
import gensim.downloader as gen_api
import gzip
import sys
import pickle

PATH_TO_EMBEDDINGS = "./data/word2vec/"
ROOT = "./data/"
TRAIN_DATASET = "sentiment_dataset_train_400.tar.gz"
TEST_DATASET = "sentiment_dataset_test_400.tar.gz"

class SentimentAnalysisDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_file, path_to_embeddings=PATH_TO_EMBEDDINGS):
        super().__init__()
        self.path_to_saved_embeddings = path_to_embeddings
        root_dir = ROOT
        f = gzip.open(root_dir + dataset_file, 'rb')
        dataset = f.read()

        if path_to_embeddings is not None:
            if os.path.exists(path_to_embeddings + 'vectors.kv'):
                self.word_vectors = KeyedVectors.load(path_to_embeddings + 'vectors.kv')
            else:
                raise ValueError("No embeddings downloaded")


        self.positive_reviews, self.negative_reviews, self.vocab = pickle.loads(dataset, encoding='latin1')
        self.categories = sorted(list(self.positive_reviews.keys()))
        self.category_sizes_pos = {category : len(self.positive_reviews[category]) for category in self.categories}
        self.category_sizes_neg = {category : len(self.positive_reviews[category]) for category in self.categories}
        self.indexed_dataset = []
        for category in self.positive_reviews:
            for review in self.positive_reviews[category]:
                self.indexed_dataset.append([review, category, 1])
        for category in self.negative_reviews:
            for review in self.negative_reviews[category]:
                self.indexed_dataset.append([review, category, 0])
        # random.shuffle(self.indexed_dataset)

    def review_to_tensor(self, review):
        list_of_embeddings = []
        for i,word in enumerate(review):
            if word in self.word_vectors.key_to_index:
                # print(word)
                embedding = self.word_vectors[word]
                # print(embedding[0:4])
                list_of_embeddings.append(np.array(embedding))
            else:
                pass
                # print("bad +++", word.replace(' ', 'HUI'))
                # next
        review_tensor = torch.FloatTensor(np.array(list_of_embeddings))
        return review_tensor

    def sentiment_to_tensor(self, sentiment):
        """
        Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
        sentiment and 1 for positive sentiment.  We need to pack this value in a
        two-element tensor.
        """        
        sentiment_tensor = torch.zeros(2)
        if sentiment == 1:
            sentiment_tensor[1] = 1
        elif sentiment == 0: 
            sentiment_tensor[0] = 1
        sentiment_tensor = sentiment_tensor.type(torch.long)
        return sentiment_tensor

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, idx):
        sample = self.indexed_dataset[idx]
        review = sample[0]
        review_sentiment = sample[2]
        review_sentiment = self.sentiment_to_tensor(review_sentiment)
        review_tensor = self.review_to_tensor(review)

        return review_tensor, review_sentiment
    

if __name__=="__main__":
    dataset_train = SentimentAnalysisDataset(
                                 dataset_file = TRAIN_DATASET,
                   )
    
    review, sent = dataset_train[0]
    print(review)
    print(sent)