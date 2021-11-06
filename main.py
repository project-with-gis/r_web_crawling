
from csv_handler import *
from preprocessing import *
from site_crawling import *
import kss
from hanspell import spell_checker
import pandas as pd

# 띄어쓰기, 맞춤법 검사
# !pip install git+https://github.com/ssut/py-hanspell.git


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

