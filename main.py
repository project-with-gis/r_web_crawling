
from csv_handler import *
from preprocessing import *
from site_crawling import *
import kss
from hanspell import spell_checker
import pandas as pd
from word2vec import *
from DEC import *
from DEC_min import *
from bert_min import *

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

    # 중간 test용(워드임베딩_ver) - 전처리 된 데이터
    pre_review = pd.read_csv("data/total_40000_review.csv")
    score1 = pre_review[:300]
    score2 = pre_review[800:1100]
    score3 = pre_review[1600:1900]
    score4 = pre_review[2400:2700]
    score5 = pre_review[3200:3500]
    pre_review = concat_df(score1,score2,score3,score4,score5)

    # 3. DEC 모델의 input 만들기 - 리뷰 데이터 워드임베딩 - 토큰화 컬럼 추가/Word2vec모델 학습시켜서 저장
    param = {'size': 100, 'window':5, 'min_count':2}
    token_review = word2vec(df=pre_review, column='review', **param)
    print(token_review[['review']])

    # 중간 test용(DEC모델_ver) - 토큰화 된 데이터
    # token_review = pd.read_csv("data/text_to_token_review.csv") # 중복제거 전
    # token_review = pd.read_csv("data/complete_duplicated.csv") # 중복제거 후

    # # 4. DEC 모델 돌리기 (학습된 Word2vec 모델 사용해서 워드임베딩 + autoencoder로 weight학습 + 클러스터링 -> dec_y 컬럼 추가)
    # token_review = remove_nan(token_review, ['preprocessed_review', 'score', 'review'])
    # token_review = token_review.reset_index()
    # review_after_dec = dec_play(token_review)
    # save_csv(review_after_dec, "민정1.csv")


    # # 5. BERT 분류모델 - original 라벨
    # pre_review = remove_nan(pre_review, ['preprocessed_review', 'score', 'review'])
    # pre_review = pre_review.reset_index(drop=True)
    # train_dataloader, test_dataloader = bert_prepro(pre_review)
    # bert_model(train_dataloader, test_dataloader) # bert 모델 저장


    # # 6. BERT 분류모델 - DEC 라벨
    # review_after_dec = review_after_dec[['preprocessed_review', 'DEC_y']]
    # review_after_dec.rename(columns={'DEC_y': 'score'}, inplace=True)
    # train_dataloader, test_dataloader = bert_prepro(review_after_dec)
    # bert_model(train_dataloader, test_dataloader)  # bert 모델 저장 # 실행전에 모델 저장하는거 이름 바꿔주기


if __name__ == '__main__':
    # review_after_dec = main('data/diningcode_total_pre_reviews_1110.csv') # 전처리된 리뷰 데이터 넣음
    # review_data = main('data/storeInfo_2.csv') # 최종 ver
    main()
    # print(review_after_dec)

