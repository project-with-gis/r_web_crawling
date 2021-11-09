def siksin_transform_datetime_df(df, int): #날짜 먼저 형식 바꾸고 컬럼위치 바꾸기 주의
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