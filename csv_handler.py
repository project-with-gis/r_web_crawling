import glob
import pandas as pd
import os


def read_csv(path): #따옴표 잊지말기
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    return df



def save_csv(df,path,name):
    df.to_csv(os.path.join(path, name), header=False, index=False) #header와 index는 필요하면 True해주기

def concat_review_csv(input_path):
    all_csv_list = glob.glob(os.path.join(input_path, '*review.csv')) #review.csv로 끝나는 모든 파일 리스트로 가져오기
    # print(all_csv_list)
    allreviews = []
    for csv in all_csv_list:
        df = pd.read_csv(csv)
        # print(type(df))
        allreviews.append(df)
    totalcsv = pd.concat(allreviews, axis=0, ignore_index=True)
    totalcsv.to_csv('./data/total_reviews.csv', header=True,index=False) #data 폴더에 total_reviews.csv라고 지정함
    return totalcsv

def concat_review_df(df1, *args):
    totaldf = pd.concat([df1, *args])
    totaldf.to_csv('./data/total_reviews.csv', header=True, index=False)

if __name__=='__main__':
     # df = read_csv('data/storeInfo_2.csv')
     # save_csv(df, './', 'csv_test.csv')
     # concat_review_csv('./data', './data/totalcsv.csv')