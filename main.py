from crawler_api.diningcode_api import *
from csv_handler import *
from crawler_api.siksin_api import *
from crawler_api.google_api import *
from crawler_api.naver_api import *
from preprocessing import *
import kss
from hanspell import spell_checker
# 띄어쓰기, 맞춤법 검사
# !pip install git+https://github.com/ssut/py-hanspell.git

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


# 구글 리뷰 크롤링해서 저장하는 함수
def google_crawling(path):
    # store_info 파일 읽어오는 함수 실행
    info_df = read_csv(path)

    # 특정 가게만 지정할 때
    # info_df = info[1800:1900].reset_index(drop=True)

    # 리뷰데이터 크롤링
    storeInfo, review_df_go = google(info_df, True, True)

    # 영어리뷰 번역리뷰 제거
    review_df_go = google_eng_transfer_del(review_df_go)

    # csv 파일로 저장
    # save_csv(review_df_goo, path, name)

    return review_df_go

def main(path):
    # 다이닝리뷰 크롤링 함수 실행
    review_df_di = diningcode_crawling(path)


    # 구글리뷰 크롤링 함수 실행
    review_df_go = google_crawling(path)
    # 영어나, 번역된 리뷰 제거
    review_df_go_notnul = google_eng_transfer_del(review_df_go)


    # 네이버 리뷰 크롤링 함수 실행
    # df = naver_store_id(info_df)
    # store_info = pd.concat([info_df, df['n_link']], axis=1)
    review_df_na = naver_review_crawling(path)

    #식신리뷰 크롤링 함수 실행
    review_df_si = siksin_review_crawling(path)


    # 4사이트 리뷰데이터 리스트에 담기
    review4_list =[review_df_di, review_df_go,review_df_na, review_df_si]

    # null값 있는 행 삭제 후 다시 담을 리스트
    review4_del_list =[]
    for reviw4 in review4_list:
        # subset에 컬럼명 적기 (하나여도 리스트로 작성 필수)
        # 데이터의 'review', 'score' null일 경우 해당 행 삭제
        review_df_go_notnul = remove_nan(reviw4, ['review', 'score'])
        review4_del_list.append(review_df_go_notnul)

    # 4사이트 합침
    concat_review = concat_df(review4_del_list)



    # 데이터 전처리
    # sentence_tokenized_review에 문장단위로 분리된 corpus가 저장된
    lines = concat_review['review']
    sentence_tokenized_review = sentence_tokenized(lines)

    # print(sentence_tokenized_review)

    # 특수문자나 기호 사이 띄어짐
    cleaned_corpus = clean_punc_2(sentence_tokenized_review)
    # print(cleaned_corpus)

    # 정규표현식을 사용한 특수문자 처리
    basic_preprocessed_corpus = clean_text(cleaned_corpus)
    # for i in range(len(basic_preprocessed_corpus)):
    #     print(basic_preprocessed_corpus[i])

    # 띄어쓰기, 맞춤법 검사

    # review 파일에 전처리 컬럼 추가

    # csv 파일로 저장
    # save_csv(total_review, path, name)

    # return total_review
    return review_df_go

if __name__ == '__main__':
    review_df_go = main('data/google_pre_test.csv')
    # print(review_df_go)




