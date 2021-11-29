import glob
import pandas as pd
import os

# csv 파일 로드
def read_csv(path):  # 따옴표 잊지말기
    data = pd.read_csv(path, encoding='utf-8-sig') # 마지막에 .csv 써주기
    df = pd.DataFrame(data)
    return df

# csv 저장
def save_csv(df, name):
    df.to_csv(os.path.join('./data/', name), header=True, index=False)  # header와 index는 필요하면 True해주기
                                                                        # name은 파일명.csv 를 의미
# 데이터프레임 concat
def concat_df(*args):
    total_df = pd.concat([*args])
    # total_df.to_csv('./data/total_reviews.csv', header=True, index=False)
    return total_df

# 중복리뷰 제거
def remove_duple(df, score, review):
    good= df[(df['score'] ==score) & (df['preprocessed_review']==review)][1:].index
    df.drop(good, inplace=True)
    return df

# def concat_review_csv(input_path):
#     all_csv_list = glob.glob(os.path.join(input_path, '*review.csv'))  # review.csv로 끝나는 모든 파일 리스트로 가져오기
#     # print(all_csv_list)
#     allreviews = []
#     for csv in all_csv_list:
#         df = pd.read_csv(csv)
#         # print(type(df))
#         allreviews.append(df)
#     totalcsv = pd.concat(allreviews, axis=0, ignore_index=True)
#     totalcsv.to_csv('./data/total_reviews.csv', header=True, index=False)  # data 폴더에 total_reviews.csv라고 지정함
#     return totalcsv


# if __name__=='__main__':
# df = read_csv('data/storeInfo_2.csv')
# save_csv(df, './', 'csv_test.csv')
# concat_review_csv('./data', './data/totalcsv.csv')


