from crawler_api.diningcode_api import *
from csv_handler import *
from crawler_api.siksin_api import *
from crawler_api.google_api import *
from crawler_api.naver_api import *
from preprocessing import *


# 다이닝코드 리뷰 크롤링해서 저장하는 함수
def diningcode_crawling(path):
    # store_info 파일 읽어오는 함수 실행
    info_df = read_csv(path)
    # 다이닝코드 리뷰 크롤링 함수 실행
    link_df = diningcode_link(info_df)
    review_df_da = diningcode_review(link_df)
    # # csv 파일로 저장
    # save_csv(review_df_da, path, name)
    return review_df_da

#
# def main(path):  # master의 메인함수 # 포털별 크롤링 함수 생성 후
#     # ## master 버전
#     # review_df_da = diningcode_crawling(path) # 다이닝코드 크롤링
#     # review_df_na = naver_crawling(path)
#     # review_df_go = google_crawling(path)
#     # review_df_si = siksin_crawling(path)
#     # 개별 전처리 (추가)
#     # total_review = concat_df(review_df_si, review_df_da, review_df_go, review_df_na) # 합치기
#
#     ## branch 버전 (전체 실행시 주석)
#     total_review = read_csv(path)  # 크롤링 결과 파일의 path 넣기
#     total_review = total_review[:20]
#     #########
#
#     total_review = delete_null(total_review)  # 리뷰없는거 제거
#     # 포털 4개 공통부분 전처리
#     pre_review = []
#     for i in range(len(total_review)):  # 리뷰하나씩 돌면서 전처리
#         review = total_review['review'][i]
#         basic_preprocessed_corpus = basic_check(review)
#         spell_preprocessed_corpus = spell_check_text(basic_preprocessed_corpus)
#         pre_review.append(spell_preprocessed_corpus)
#     total_review['preprocessed_review'] = pre_review  # 전처리 컬럼 생성 후 데이터 추가
#     # total_review = delete_null(total_review) # 리뷰 없어졌을 수도 있으니 한번 더 null값 제거
#     save_csv(total_review, "total_review_전처리.csv")  # csv 파일로 저장
#     return total_review




def main(path):  # branch의 메인함수
    review_df = read_csv(path)  # 크롤링 결과 파일의 path 넣기
    #review_df = review_df[333:500]
    review_df_da = delete_null(review_df)  # 리뷰없는거 제거
    # 전처리 시작
    pre_review = []
    for i in range(len(review_df_da)):  # 리뷰하나씩 돌면서 전처리
        review = review_df_da['review'][i]
        print(i,"=============================")
        basic_preprocessed_corpus = basic_check(review)
        spell_preprocessed_corpus = spell_check_text(basic_preprocessed_corpus)
        pre_review.append(spell_preprocessed_corpus)
    review_df_da['preprocessed_review'] = pre_review  # 전처리 컬럼에 추가
    save_csv(review_df_da, "diningcode_전처리_1109.csv")

    return review_df_da


if __name__ == '__main__':
    review_df_da = main("data/diningcode_total_review_1105.csv")

