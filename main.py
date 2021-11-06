from crawler_api.diningcode_api import *
from csv_handler import *
from crawler_api.siksin_api import *
from crawler_api.google_api import *
from crawler_api.naver_api import *
from preprocessing import *
import kss
from hanspell import spell_checker
import pandas as pd

# 띄어쓰기, 맞춤법 검사
# !pip install git+https://github.com/ssut/py-hanspell.git

# 다이닝코드 리뷰 크롤링 함수
def diningcode_crawling(path):
    # store_info 파일 읽어오는 함수 실행
    info_df = read_csv(path)
    # 다이닝코드 리뷰 크롤링 함수 실행
    link_df = diningcode_link(info_df)
    review_df_da = diningcode_review(link_df)
    # # csv 파일로 저장
    # save_csv(review_df_da, path, name)
    return review_df_da


# 구글 리뷰 크롤링 함수
def google_crawling(path):
    # store_info 파일 읽어오는 함수 실행
    info_df = read_csv(path)

    # 특정 가게만 지정할 때
    # info_df = info[1800:1900].reset_index(drop=True)

    # 리뷰데이터 크롤링
    storeInfo, review_df_go = google(info_df)

    # 영어리뷰 번역리뷰 제거
    review_df_go = google_eng_transfer_del(review_df_go)

    # csv 파일로 저장
    # save_csv(review_df_goo, path, name)

    return review_df_go

# 네이버 리뷰 크롤링 함수
def naver_crawling(path):
    return 1

def siksin_crawling(path):
    return 1

def main(path):

    # 다이닝리뷰 크롤링 함수 실행
    review_df_di = diningcode_crawling(path)
    # print(review_df_di)


    # 구글리뷰 크롤링 함수 실행
    review_df_go = google_crawling(path)
    # print(review_df_go)

    # 영어나, 번역된 리뷰 제거
    review_df_go = google_eng_transfer_del(review_df_go)
    # print(review_df_go)


    # 네이버 리뷰 크롤링 함수 실행
    review_df_na = naver_crawling(path)
    # print(review_df_na)


    #식신리뷰 크롤링 함수 실행
    review_df_si = siksin_review_crawling(path)
    #print(reveiw_df_si)


    # 4사이트 합침
    concat_review = concat_df(review_df_di,review_df_go,review_df_na,review_df_si)
    del concat_review['Unnamed: 0']
    # print(concat_review)

    # subset에 컬럼명 적기 (하나여도 리스트로 작성 필수)
    # 데이터의 'review', 'score' null일 경우 해당 행 삭제
    total_review = remove_nan(concat_review, ['review', 'score'])
    # print(total_review)


    # 데이터 전처리
    # sentence_tokenized_review에 문장단위로 분리된 corpus가 저장된
    # print(total_review['review'])
    after_review = []
    for i in range(len(total_review['review'])):
        after = prepro(total_review['review'][i])
        after_review.append(after)
        # print(after_review)

    # review 파일에 전처리 컬럼 추가

    total_review['after_review'] = after_review
    # print(total_review)

    # csv 파일로 저장
    # save_csv(total_review, path, name)
    total_review.to_csv('./data/total_reviews.csv', header=True, index=False)

    return total_review


if __name__ == '__main__':
    review_data = main('data/10.csv')
    print(review_data)

