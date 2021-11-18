import numpy as np
import pandas as pd

from word2vec import *

def get_features(words, model, num_features):
    # 출력 벡터 초기화
    feature_vector = np.zeros((num_features), dtype=np.float32)
    num_words = 0
    # 어휘 사전 준비
    index2word_set = set(model.wv.index2word)

    for w in words:
        if w in index2word_set:
            num_words += 1
            # 사전에 해당하는 단어에 대해 단어 벡터를 더함
            feature_vector = np.add(feature_vector, model[w])

    # 문장의 단어 수만큼 나누어 단어 벡터의 평균값을 문장 벡터로 함
    feature_vector = np.divide(feature_vector, num_words)
    return feature_vector


def get_dataset(model, reviews, num_features):
    dataset = list()

    for s in reviews:
        dataset.append(get_features(s, model, num_features))

    reviewFeatureVecs = np.stack(dataset)
    return reviewFeatureVecs

def load_review(df, **param):
    df = df[df['tokenized_review'].str.len() != 0] # tokenized_review 빈 리스트 제거
    tokenized_review = df['tokenized_review']
    model = Word2Vec.load('word2vec_model_150_5_5.bin')

    x = get_dataset(model, list(tokenized_review), param['size'])
    y = df['score'].to_numpy()

    return x.astype(float), y

