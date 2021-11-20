import pandas as pd
from gensim.models import Word2Vec
from konlpy.tag import Okt

# 토큰화 (Okt 이용)
def text_to_token(df, column):
    okt = Okt()

    # 품사 종류 = [Adjective, Adverb, Alpha, Conjunction, Determiner, Eomi, Exclamation, Foreign, Hashtag, Josa,
    #           KoreanParticle, Noun, Number, PreEomi, Punctuation, ScreenName, Suffix, Unknown, Verb]

    tokenized_review = []
    for review in df[column]:
        tmp = []
        if review:
            for j in okt.pos(review, norm=True, stem=True):
                if j[1] not in ['Alpha', 'Conjunction', 'Determiner', 'Eomi', 'Josa', 'PreEomi', 'Suffix']: # 알파벳, 접속사, 관형사, 어미, 조사, 선어말어미, 접미사 제외
                    tmp.append(j[0])
        tokenized_review.append(tmp)

    df['tokenized_review'] = tokenized_review
    return df

def word2vec(df, **param):
    # df = text_to_token(df, column) # 토큰화

    review_data = df.reset_index(drop=True) # 토큰화 한 후 인덱스 재배열
    review_data = review_data[review_data['tokenized_review'].str.len() != 0] # tokenized_review 빈 리스트 제거

    texts = []
    for i in review_data['tokenized_review']: # 빈 리스트 제거후 texts 리스트에 넣기
        texts.append(i)

    # texts = sum(texts, []) # 이중리스트 해결

    # texts : 결측값 없는 토큰화된 리뷰
    embedding_model = Word2Vec(texts, size=param['size'], window=param['window'], min_count=param['min_count'], workers=4, sg=1)
    embedding_model.save('weights/word2vec_test_jeong1120.bin')

    return df

df = pd.read_csv('data/complete_duplicated_tokenized_review.csv')


param = {'size':100, 'window':4, 'min_count':3}
df = word2vec(df,  **param)
