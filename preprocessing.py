from datetime import datetime

import pandas as pd


def remove_english(df):
    print("전처리하는 함수")


def transform_datetime(path, int): #날짜 먼저 형식 바꾸고 컬럼위치 바꾸기 주의
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    date = df.iloc[:, int].astype(str)
    date = date.str.split(" ")
    df.iloc[:, int] = date.str.get(0)
    # df.iloc[:, int] = df.iloc[:, int].apply(lambda _: datetime.strptime(_, "%Y-%m-%d"))
    # df.iloc[:, int] = pd.to_datetime(df.iloc[:, int], format="%Y-%m-%d")
    # print(df.iloc[:, int])
    return df


def swap_columns_with_name(path, *args): # (*args)에는 원하는 columns 이름 순서대로(따옴표 잊지말기)
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    df = df[[*args]]
    # print(df.head())
    return df


def swap_columns_with_num(path, *args): # (*args)에는 원하는 columns index순서대로
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    col = df.columns.to_numpy()
    col = col[[*args]]
    df = df[col]
    # print(df.head())
    return df


def rounding_off_scores(path, num):
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    num = int(num)
    score = df.iloc[:, num]
    df.iloc[:, num] = score.round(0).astype(int)
    # print(df.iloc[:, num].head(21))
    return df

#-------------------------------------------------------------(위)csv 파일들 기준
#-----------------------------------------------------------(아래)dataframe 기준


def transform_datetime_df(df, int): #날짜 먼저 형식 바꾸고 컬럼위치 바꾸기 주의
    date = df.iloc[:, int].astype(str)
    date = date.str.split(" ")
    df.iloc[:, int] = date.str.get(0)
    # df.iloc[:, int] = df.iloc[:, int].apply(lambda _: datetime.strptime(_, "%Y-%m-%d"))
    # df.iloc[:, int] = pd.to_datetime(df.iloc[:, int], format="%Y-%m-%d")
    # print(df.iloc[:, int])
    return df


def swap_columns_with_name_df(df, *args): # (*args)에는 원하는 columns 이름 순서대로(따옴표 잊지말기)
    df = df[[*args]]
    # print(df.head())
    return df


def swap_columns_with_num_df(df, *args): # (*args)에는 원하는 columns index순서대로
    col = df.columns.to_numpy()
    col = col[[*args]]
    df = df[col]
    # print(df.head())
    return df


def rounding_off_scores_df(df, num):
    num = int(num)
    score = df.iloc[:, num]
    df.iloc[:, num] = score.round(0).astype(int)
    # print(df.iloc[:, num].head(21))
    return df


# if __name__ == '__main__':
    # swap_columns_with_num('data/siksin_1review_test.csv', 0,1,2,4,3)
    # transform_datetime('data/siksin_1review_test.csv', 4)
    # rounding_off_scores('data/siksin_1review_test.csv', 2)