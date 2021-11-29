# 토큰화 + 워드 임베딩
# import pandas as pd
from gensim.models import Word2Vec
from konlpy.tag import Okt
from tqdm import tqdm
from preprocessing import *
from csv_handler import *
import jpype
import tweepy
# import sys
# sys.version
# tweepy.__version__
# 토큰화 (Okt 이용)
def text_to_token(df, column):
    okt = Okt()

    # 품사 종류 = [Adjective, Adverb, Alpha, Conjunction, Determiner, Eomi, Exclamation, Foreign, Hashtag, Josa,
    #           KoreanParticle, Noun, Number, PreEomi, Punctuation, ScreenName, Suffix, Unknown, Verb]

    tokenized_review = []
    for review in tqdm(df[column]):
        review = str(review)
        tmp = []
        if review:
            for j in okt.pos(review, norm=True, stem=True):
                if j[1] not in ['Alpha', 'Conjunction', 'Determiner', 'Eomi', 'Josa', 'PreEomi', 'Suffix']: # 알파벳, 접속사, 관형사, 어미, 조사, 선어말어미, 접미사 제외
                    tmp.append(j[0])
        tokenized_review.append(tmp)

    df['tokenized_review'] = tokenized_review
    return df

def word2vec(df, **param): # 모델 저장 # **param은 word2vec내에 정의되어있는 Word2Vec함수의 파라미터
    # df = text_to_token(df, column) # 토큰화
    df = remove_nan(df, ['tokenized_review', 'score', 'review'])
    df = df.reset_index(drop=True)
    # review_data = review_data[review_data['tokenized_review'].str.len() != 0] # tokenized_review 빈 리스트 제거

    texts = []
    for i in tqdm(df['tokenized_review']):
        texts.append(i)

    embedding_model = Word2Vec(texts, size=param['size'], window=param['window'], min_count=param['min_count'], workers=4, sg=1)
                    # size = 임베딩된 벡터의 차원, window = 컨텍스트 윈도우 크기, min_count = 단어 최소 빈도수 제한(빈도적으면학습X)
    a = []
    for j in range(len(texts)):
        try:
            a.append(embedding_model[texts[j]])
        except:
            a.append(0)
    df['word2vec'] = a
    save_csv(df, '민정3워드투벡.csv')

    embedding_model.save('C:/Users/alti1/PycharmProjects/r_web_crawling/data/word2vec_models/complete_duplicated.bin')

    return df

###########################################
# import pandas as pd
# df = pd.read_csv('data/naver_total(700000~750000).csv')
# df = df[:100]
# df = text_to_token(df=df, column='preprocessed_review')

