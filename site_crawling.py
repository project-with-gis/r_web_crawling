from csv_handler import *
from crawler_api.siksin_api import *
from crawler_api.google_api import *
from crawler_api.naver_api import *
from crawler_api.diningcode_api import *
from preprocessing import *

# 전체데이터 크롤링
def site_crawling(path):
    review_df_di = diningcode_crawling(path)
    review_df_go = google_crawling(path)
    review_df_na = naver_crawling(path)
    review_df_si = siksin_crawling(path)

    return review_df_di, review_df_go, review_df_na, review_df_si




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
    review_df_go = google_eng_transfer_del1(review_df_go)

    edit_reviewlist = google_eng_transfer_del2(review_df_go['review'])
    del review_df_go['review']
    review_df_go['review'] = edit_reviewlist
    # csv 파일로 저장
    # save_csv(review_df_goo, path, name)

    return review_df_go

# 네이버 리뷰 크롤링 함수
def naver_crawling(path):
  # store_info 파일 읽어오기
  info_df = pd.read_csv(path)

  # n_link 크롤링
  link_df = naver_store_id(info_df)

  # 네이버 리뷰 크롤링
  review_df = naver_review_crawling(link_df)

  change_df = swap_columns_with_num_df(review_df, 0,1,4,3,2) # 컬럼순서 변경
  del_df = remove_nan(change_df, ['reveiw','score']) # 결측값 제거
  round_df = rounding_off_scores_df(del_df,3) # 평점 반올림
  review_df_na = naver_transform_datetime_df(round_df) # 날짜 형태 변환

  # # csv 파일로 저장
  # save_csv(review_df_na, path, name)
  return review_df_na


def siksin_crawling(path):
    #store_info 파일 읽어오기
    info_df = read_csv(path)
    #웹사이트 s_link등 가져오기
    store_df = add_siksin_info(info_df)
    #식신 리뷰 크롤링
    date_df = siksin_review_scraping(store_df)
    #식신 별점 반올림하기
    score_df = rounding_off_scores_df(date_df, 3)
    #식신 데이트 형식 바꾸기(YY-mm-dd)
    review_df_si = siksin_transform_datetime_df(score_df, 2)
    return review_df_si
