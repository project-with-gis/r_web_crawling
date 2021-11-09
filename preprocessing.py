from dateutil.parser import parse


# 식신 사이트 날짜 먼저 형식 바꾸고 컬럼위치 바꾸기 주의
def siksin_transform_datetime_df(df, int):
    date = df.iloc[:, int].astype(str)
    date = date.str.split(" ")
    df.iloc[:, int] = date.str.get(0)
    # df.iloc[:, int] = df.iloc[:, int].apply(lambda _: datetime.strptime(_, "%Y-%m-%d"))
    # df.iloc[:, int] = pd.to_datetime(df.iloc[:, int], format="%Y-%m-%d")
    # print(df.iloc[:, int])
    return df

#네이버사이트 date 형식변환
def naver_transform_datetime_df(df):
    # 요일 제거
    for i, line in enumerate(df['date']):
        line = line.rstrip('.월화수목금토일')
        df['date'][i] = line

    # 날짜 변환
    for i in range(len(df)):
      a = parse(df['date'][i], yearfirst=True)
      df['date'][i] = a.strftime("%Y-%m-%d")

# 구글 사이트/ 영어나, 번역된 리뷰 제거
def google_eng_transfer_del(google_review_data):
    for i, review in enumerate(google_review_data['review']):
        if type(review) != 'str':
            review = str(review)
        if "번역" in review:
            google_review_data = google_review_data.drop(google_review_data.index[i])
        elif "원문" in review:
            google_review_data = google_review_data.drop(google_review_data.index[i])

    return google_review_data


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


####################################################
def basic_check(review):  # 한 행마다 실행되도록. 이 함수가 받아오는건 하나의 리뷰
    cleaned_corpus = clean_punc(review)
    basic_preprocessed_corpus = clean_text(cleaned_corpus)
    return basic_preprocessed_corpus

def spell_check_text(texts): # 한 댓글에 대한 문장들
    lownword_map = make_dictionary() # 외래어 사전
    spelled_sent = spell_checker.check(texts) # 띄어쓰기, 맞춤법
    checked_sent = spelled_sent.checked
    normalized_sent = repeat_normalize(checked_sent) # 반복되는 이모티콘이나 자모를 normalization
    for lownword in lownword_map: # 왜래어 바꿔줌 (miss spell -> origin spell)
        normalized_sent = normalized_sent.replace(lownword, lownword_map[lownword])
    corpus = normalized_sent

    print(corpus)
    return corpus