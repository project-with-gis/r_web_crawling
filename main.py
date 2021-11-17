
from csv_handler import *
from preprocessing import *
from site_crawling import *
import kss
from hanspell import spell_checker
import pandas as pd
from word2vec import *
from DEC import *
from DEC_min import *

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
    # concat_review =pd.read_csv(path)
    #concat_review = concat_review[8240:8260].reset_index(drop=True)


    # subset에 컬럼명 적기 (하나여도 리스트로 작성 필수)
    # 데이터의 'review', 'score' null일 경우 해당 행 삭제
    # total_review = remove_nan(concat_review, ['review', 'score'])
    # print(total_review)

    # 데이터 전처리
    # print(total_review['review'])
    # pre_review = prepro(total_review)
    # print(pre_review)


    pre_review = pd.read_csv(path)

    pre_review = remove_nan(pre_review, ['preprocessed_review', 'score', 'review'])
    # 토큰화 컬럼 추가 # Word2Vec모델 학습시켜서 저장                 # **param은 word2vec내에 정의되어있는 Word2Vec함수의 파라미터
    token_review = word2vec(df=pre_review, column='preprocessed_review',  size=100, window=5, min_count=5)


    # DEC 모델 돌리기 (이 안에서 워드투벡터함수도사용-load_review함수에서) # 클러스터링해서 알려줌
    # y_pred = dec_play(token_review) # weights 저장
    review_after_dec = dec_play(token_review) # 돌려보는거
    # 5개로 그룹핑된 결과를 군집마다 1~5점 할당하고 token_review 컬럼에 추가 (label이 0이라고 score가 0인것은 아님)


    # csv 파일로 저장
    save_csv(review_after_dec, 'review_after_dec.csv')

    return y_pred


if __name__ == '__main__':
    review_after_dec = main('data/diningcode_total_pre_reviews_1110.csv') # 전처리된 리뷰 데이터 넣음
    # review_data = main('data/storeInfo_2.csv') # 최종 ver
    print(review_after_dec)



# 주연언니 naver 추가 전처리
# def main(path):
#
#     concat_review =pd.read_csv(path)
#     concat_review = concat_review.astype({'preprocessed_review': 'str'})
#     index_re = concat_review[concat_review['preprocessed_review'] == 'nan'].index.tolist()
#     total_review = concat_review.loc[index_re]
#     total_review = total_review.reset_index(drop=True)
#     total_review = total_review.drop(columns = 'preprocessed_review')
#
#     after_review_total, total_review = prepro(total_review['review'],total_review)
#
#     # review 파일에 전처리 컬럼 추가
#     total_review['preprocessed_review'] = after_review_total
#
#     # csv 파일로 저장
#     save_csv(total_review, 'naver_plus.csv')
#     return total_review
#
#
# if __name__ == '__main__':
#     review_data = main('data/naver_preprocessed_reviews.csv') # 사이트리뷰데이터 넣으면됨
#     # review_data = main('data/storeInfo_2.csv') # 최종 ver
#     print(review_data)
#
#     pd.read_csv("data/naver_plus.csv")

a = pd.read_csv('data/dining_review_after_dec.csv')

correct = 0
for i in range(len(a)):
    if a['DEC_y'][i] == a['score'][i]:
        correct += 1
print(correct/len(a))

a[a['DEC_y']==0]['score'].value_counts()
a[a['DEC_y']==1]['score'].value_counts()
a[a['DEC_y']==2]['score'].value_counts() # 4점
a[a['DEC_y']==3]['review'].value_counts()
a[a['DEC_y']==4]['score'].value_counts()

a['score'].value_counts()