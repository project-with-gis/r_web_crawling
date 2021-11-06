from crawler_api.diningcode_api import *
from csv_handler import *
from crawler_api.siksin_api import *
from crawler_api.google_api import *
from crawler_api.naver_api import *

# 구글 리뷰 크롤링해서 저장하는 함수
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






def main(path):
    # store_info 파일 읽어오는 함수 실행
    info_df = read_csv(path)

    # 다이닝코드 리뷰 크롤링 함수 실행
    link_df = diningcode_link(info_df)
    review_df_da = diningcode_review(link_df)

    # 식신 리뷰 크롤링 함수 실행
    store_df = add_siksin_info(info_df)
    review_df_si = siksin_review_scraping(store_df)

    # 구글 리뷰 크롤링 함수 실행
    # storeInfo, review_df_go = google(info_df, True, True)
    review_df_go = google(info_df, True, True)

    # 네이버 리뷰 크롤링 함수 실행
    df = naver_store_id(info_df)
    store_info = pd.concat([info_df, df['n_link']], axis=1)
    review_df_na = naver_review_crawling(store_info)

    # 사이트4개 리뷰 합치기
    total_review = concat_df(review_df_si, review_df_da, review_df_go, review_df_na)
    # review 파일에 전처리 컬럼 추가


    # csv 파일로 저장
    save_csv(total_review, path, name)

    return total_review




if __name__ == '__main__':
    main()
