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


def main(path): # branch의 메인함수
    review_df = read_csv(path) # 크롤링 결과 파일의 path 넣기
    review_df_da = delete_null(review_df) # 리뷰없는거 제거
    # 전처리 시작
    pre_review = []
    for i in range(len(review_df_da)): # 리뷰하나씩 돌면서 전처리
        review = review_df_da['review'][i]
        basic_preprocessed_corpus = basic_check(review)
        spell_preprocessed_corpus = spell_check_text(basic_preprocessed_corpus)
        pre_review.append(spell_preprocessed_corpus)
    review_df_da['preprocessed_review'] = pre_review # 전처리 컬럼에 추가
    #save_csv(review_df_da, path, "diningcode_전처리_테스트1106")

    return review_df_da


# def main(path): # 처음만든 메인함수. 포털별로 크롤링 분리하기 전
#     # info_df = diningcode_crawling(path) # master
#     # store_info 파일 읽어오는 함수 실행
#     info_df = read_csv(path) # branch
#
#     # 다이닝코드 리뷰 크롤링 함수 실행
#     link_df = diningcode_link(info_df)
#     review_df_da = diningcode_review(link_df)
#
#     # 식신 리뷰 크롤링 함수 실행
#     store_df = add_siksin_info(info_df)
#     review_df_si = siksin_review_scraping(store_df)
#
#     # 구글 리뷰 크롤링 함수 실행
#     # storeInfo, review_df_go = google(info_df, True, True)
#     review_df_go = google(info_df, True, True)
#
#     # 네이버 리뷰 크롤링 함수 실행
#     df = naver_store_id(info_df)
#     store_info = pd.concat([info_df, df['n_link']], axis=1)
#     review_df_na = naver_review_crawling(store_info)
#
#     # 사이트4개 리뷰 합치기
#     total_review = concat_df(review_df_si, review_df_da, review_df_go, review_df_na)
#     # review 파일에 전처리 컬럼 추가
#
#     # 전처리 시작
#
#     # csv 파일로 저장
#     save_csv(total_review, path, name)
#
#     return total_review




# def main(path): # master의 메인함수 # 포털별 크롤링 함수 생성 후
#     review_df_da = diningcode_crawling(path) # 다이닝코드 크롤링
#     review_df_na = naver_crawling(path)
#     review_df_go = google_crawling(path)
#     review_df_si = siksin_crawling(path)
#     # 개별 전처리
#     ######(추가)
#     total_review = concat_df(review_df_si, review_df_da, review_df_go, review_df_na) # 합치기
#     # 공통부분 전처리
#     #######(추가)
#     save_csv(total_review, path, name) # csv 파일로 저장

if __name__ == '__main__':
    main()
