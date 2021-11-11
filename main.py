
from csv_handler import *
from preprocessing import *
from site_crawling import *
import kss
from hanspell import spell_checker
import pandas as pd

# 띄어쓰기, 맞춤법 검사
# !pip install git+https://github.com/ssut/py-hanspell.git


def main(path):

    # 사이트별 크롤링 함수 실행 - 전체 사이트 크롤링부터 시작할 때 (최종 project_ver)
    # review_df_di, review_df_go, review_df_na, review_df_si = site_crawling(path)

    # 4사이트 합침 - 최종
    # concat_review = concat_df(review_df_di,review_df_go,review_df_na,review_df_si)
    # print(concat_review)

    # google만 리뷰데이터 있는 상태에서 전처리하는 코드
    # - main('data/google_total_reviews_1105.csv') 로 변경
    concat_review =pd.read_csv(path)
    #concat_review = concat_review[8240:8260].reset_index(drop=True)


    # subset에 컬럼명 적기 (하나여도 리스트로 작성 필수)
    # 데이터의 'review', 'score' null일 경우 해당 행 삭제
    total_review = remove_nan(concat_review, ['review', 'score'])
    # print(total_review)
    total_review = total_review[530000:600000]

    #특정리뷰 테스트할 때
    # total_review.loc[0]=[1,2,3,4,'각종 해산물(전복, 각종조개...)이  많이 들어 있어서인지, 국물이 시원하고 좋았다. 전날 먹은 술이 완전 해장되었고, 국물이 좋아, 다시 술 먹고 싶은 생각이 들었다. 또한, 식당에 들어가 전복해물뚝배기 가격이 19,000원인것을 보고, 조금 비싸지  않나 싶었는데, 먹고나니, 돈이 아깝지 않았다. 다만, 반찬에 김치가 없는게, 조금 아쉬었다.']
    # print(total_review)

    # 데이터 전처리
    # print(total_review['review'])
    after_review_total, total_review = prepro(total_review['review'],total_review)
    # print(after_review_total)


    # review 파일에 전처리 컬럼 추가
    total_review['preprocessed_review'] = after_review_total
    # print(total_review)

    #전처리 후 리뷰가 '' 비어있는 상태인 행 삭제
    total_review = remove_after_nan(total_review)

    # csv 파일로 저장
    save_csv(total_review, 'naver_total(530000~600000).csv')
    return total_review


if __name__ == '__main__':
    review_data = main('data/naver_total_reviews_1107.csv') # 사이트리뷰데이터 넣으면됨
    # review_data = main('data/storeInfo_2.csv') # 최종 ver
    print(review_data)
