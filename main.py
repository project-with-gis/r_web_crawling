from kobert.kobert_model import kobert_train, kobert_predict

from csv_handler import *
from preprocessing import *
from site_crawling import *
import kss
from hanspell import spell_checker
import pandas as pd
from word2vec import *
from DEC import *

from bert import *
from preprocessing import *

# 띄어쓰기, 맞춤법 검사
# !pip install git+https://github.com/ssut/py-hanspell.git


def main():

    # 1. 사이트별 크롤링 함수 실행 - 전체 사이트 크롤링부터 시작할 때 (최종 project_ver)
    # review_df_di, review_df_go, review_df_na, review_df_si = site_crawling(path) # path는 store_info 파일 경로를 의미
    # concat_review = concat_df(review_df_di,review_df_go,review_df_na,review_df_si) # 4사이트 합침
    # print(concat_review)

    # 중간 test용(데이터전처리_ver) - 크롤링 된 데이터
    # concat_review =pd.read_csv('data/google_total_reviews_1105.csv')
    # concat_review = concat_review[8240:8260].reset_index(drop=True)

    # 2. 데이터 전처리 - 클리닝과정
    # total_review = remove_nan(concat_review, ['review', 'score']) # 데이터의 'review', 'score' null일 경우 해당 행 삭제
    # pre_review = pre_review.reset_index(drop=True)
    # print(total_review)
    # pre_review = prepro(total_review) # 전처리 함수 실행
    # print(pre_review)

    # # 중간 test용(워드임베딩_ver) - 전처리 된 데이터
    # pre_review = pd.read_csv("data/text_to_token_list_del_verb_noun.csv")

    # # 3. DEC 모델의 input 만들기 - 토큰화 컬럼 추가/Word2vec모델 학습시켜서 저장
    # ## 중복 데이터 제거
    # print("중복제거 전 pre_review 개수 : ", len(pre_review))
    # pre_review = remove_duple(pre_review, 5, '너무 맛있어요')
    # pre_review = remove_duple(pre_review, 5, '아주 맛있어요')
    # pre_review = remove_duple(pre_review, 4, '맛있습니다')
    # pre_review = remove_duple(pre_review, 4, '맛나요')
    # pre_review = remove_duple(pre_review, 4, '맛집')
    # pre_review = remove_duple(pre_review, 4, '굿')
    # pre_review = remove_duple(pre_review, 2, '별로')
    # pre_review = remove_duple(pre_review, 4, '맛있어요')
    # pre_review = remove_duple(pre_review, 4, '맛있어요')
    # pre_review = remove_duple(pre_review, 4, '좋아요')
    # pre_review = remove_duple(pre_review, 4, '맛있음')
    #
    # pre_review = pre_review.drop_duplicates(['review'], keep='first') # 나머지는 첫번째 값만
    # print("중복제거 후 pre_review 개수 : ", len(pre_review))
    # pre_review = pre_review.reset_index(drop=True)
    #
    # score1 = pre_review[pre_review['score']==1][:59304] # 2점짜리 리뷰 갯수에 맞춤
    # score2 = pre_review[pre_review['score']==2][:59304]
    # score3 = pre_review[pre_review['score']==3][:59304]
    # score4 = pre_review[pre_review['score']==4][:59304]
    # score5 = pre_review[pre_review['score']==5][:59304]
    # pre_review = concat_df(score1,score2,score3,score4,score5)
    # pre_review['score'] = pre_review['score'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4})
    # pre_review = remove_nan(pre_review, ['preprocessed_review', 'score', 'review', 'tokenized_review'])
    # pre_review = pre_review.reset_index(drop=True)

    # ## 리뷰 데이터 워드임베딩
    # param = {'size': 100, 'window':5, 'min_count':1}
    # token_review = word2vec(df=pre_review, column='review', **param)
    # print(token_review[['review']])

    # # 중간 test용(DEC모델_ver) - 토큰화 된 데이터
    # token_review = pd.read_csv("data/민정6워드투벡.csv")
    #
    # # 4. DEC 모델 돌리기 (학습된 Word2vec 모델 사용해서 워드임베딩 + autoencoder로 weight학습 + 클러스터링 -> dec_y 컬럼 추가)
    # token_review = remove_nan(token_review, ['preprocessed_review', 'score', 'review', 'tokenized_review'])
    # token_review = token_review.reset_index()
    # review_after_dec = dec_play(token_review)
    # save_csv(review_after_dec, "dec_y저장2.csv")

    # 중간 test용(BERT모델_ver) - 토큰화 된 데이터
    review_after_dec = pd.read_csv("data/dec_y저장2.csv")
    review_after_dec['DEC_y'] = review_after_dec['DEC_y'].map({0: 3, 1: 5, 2: 1, 3: 2, 4: 4}) # relabeling

    # # 5. BERT 분류모델 - original 라벨
    # pre_review = remove_nan(pre_review, ['preprocessed_review', 'score', 'review'])
    # pre_review = pre_review.reset_index(drop=True)
    # train_dataloader, test_dataloader = bert_prepro(pre_review)
    # bert_model(train_dataloader, test_dataloader) # bert 모델 저장


    # 6. BERT 분류모델 - DEC 라벨
    review_after_dec = review_after_dec[['preprocessed_review', 'DEC_y']]
    review_after_dec.rename(columns={'DEC_y': 'score'}, inplace=True) # 버트모델 안에 변수명이랑 일치시킬려고
    train_dataloader, test_dataloader = bert_prepro(review_after_dec)
    bert_model(train_dataloader, test_dataloader)  # bert 모델 저장 # 실행전에 모델 저장하는거 이름 바꿔주기


    # # 7. KoBERT 분류 모델
    # df = pd.read_csv('data/min_score_count_data.csv')
    # df.dropna(axis=0)
    # df['score'] = df['score'].astype(int)
    #
    # # 점수마다 갯수 일치 시켜 데이트 프레임 생성
    # score_5 = df[df['score'] == 5.0][:]
    # score_4 = df[df['score'] == 4.0][:]
    # score_3 = df[df['score'] == 3.0][:]
    # score_2 = df[df['score'] == 2.0][:]
    # score_1 = df[df['score'] == 1.0][:]
    # # print(len(score_5), len(score_4),  len(score_3),  len(score_2),  len(score_1))
    # df = pd.concat([score_1, score_2, score_3, score_4, score_5])
    #
    # scraping_data = df.reset_index(drop=True)
    #
    # print('KOBERT START') # kobert model 학습 (평점 model)
    # column = 'preprocessed_review' # 'preprocessed_review'
    # num_classes = 5; max_len = 256; batch_size = 64; epochs = 5
    # model_name = 'kobert_test'
    # kobert_train(scraping_data, column, num_classes, max_len, batch_size, epochs, model_name)
    #
    # print('KOBERT predict START') # 학습 시킨 모델로 예측
    # model_name = 'kobert_test'
    # predict_data = kobert_predict(scraping_data, model_name)
    #
    # print(predict_data)


if __name__ == '__main__':
    # review_after_dec = main('data/diningcode_total_pre_reviews_1110.csv') # 전처리된 리뷰 데이터 넣음
    # review_data = main('data/storeInfo_2.csv') # 최종 ver
    main()
    # print(review_after_dec)




