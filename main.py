
from csv_handler import *
from preprocessing import *
from site_crawling import *
import kss
from hanspell import spell_checker
import pandas as pd

# 띄어쓰기, 맞춤법 검사
# !pip install git+https://github.com/ssut/py-hanspell.git


def main(path):

    # 사이트별 크롤링 함수 실행
    review_df_di, review_df_go, review_df_na, review_df_si = site_crawling(path)

    # 4사이트 합침
    concat_review = concat_df(review_df_di,review_df_go,review_df_na,review_df_si)
    del concat_review['Unnamed: 0']
    # print(concat_review)

    # subset에 컬럼명 적기 (하나여도 리스트로 작성 필수)
    # 데이터의 'review', 'score' null일 경우 해당 행 삭제
    total_review = remove_nan(concat_review, ['review', 'score'])
    # print(total_review)


    # 데이터 전처리
    # print(total_review['review'])
    after_review_total = prepro_2(total_review['review'])
    # print(after_review_total)


    # review 파일에 전처리 컬럼 추가
    total_review['after_review'] = after_review_total
    # print(total_review)

    #전처리 후 리뷰가 '' 비어있는 상태인 행 삭제
    total_review = remove_after_nan(total_review)

    # csv 파일로 저장
    # save_csv(total_review, path, name)
    total_review.to_csv('./data/total_reviews.csv', header=True, index=False)

    return total_review


if __name__ == '__main__':
    review_data = main('data/storeInfo_2.csv')
    print(review_data)

