from diningcode_api import *
from csv_handler import *
from siksin_api import *

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
    # 네이버 리뷰 크롤링 함수 실행
    # 사이트4개 리뷰 합치기
    total_review = concat_df(review_df_si, review_df_da)
    # review 파일에 전처리 컬럼 추가


    # csv 파일로 저장
    save_csv(total_review, path, name)

    return




if __name__ == '__main__':
    main()
