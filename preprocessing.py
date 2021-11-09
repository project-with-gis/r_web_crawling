from dateutil.parser import parse
import re
import os
import datetime
import pandas as pd
from hanspell import spell_checker
from soynlp.normalizer import *


# 식신 사이트 date 형식변환
def siksin_transform_datetime_df(df, int):
    date = df.iloc[:, int].astype(str)
    date = date.str.split(" ")
    df.iloc[:, int] = date.str.get(0)
    return df

# 네이버사이트 date 형식변환
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


# def swap_columns_with_name_df(df, *args): # (*args)에는 원하는 columns 이름 순서대로(따옴표 잊지말기)
#     df = df[[*args]]
#     # print(df.head())
#     return df

# 컬럼위치조정
def swap_columns_with_num_df(df, *args): # (*args)에는 원하는 columns index순서대로
    col = df.columns.to_numpy()
    col = col[[*args]]
    df = df[col]
    # print(df.head())
    return df

# 평점 반올림해주는 함수
def rounding_off_scores_df(df, num):
    num = int(num)
    score = df.iloc[:, num]
    df.iloc[:, num] = score.round(0).astype(int)
    # print(df.iloc[:, num].head(21))
    return df


#######################공통으로쓰는 전처리 함수#############################
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

    return corpus

def make_dictionary():
    lownword_map = {}
    lownword_data = open('data/confused_loanwords.txt', 'r', encoding='utf-8')
    lines = lownword_data.readlines()
    for line in lines:
        line = line.strip()
        miss_spell = line.split('\t')[0]
        ori_word = line.split('\t')[1]
        lownword_map[miss_spell] = ori_word
    return lownword_map

def clean_punc(texts): #문장부호같은거 다 삭제
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }

    for p in punct_mapping:
        texts = texts.replace(p, punct_mapping[p])  # punct_mapping에 있는 변수가 있으면 대응되는걸로 변경되도록 한다.

    for p in punct:
        texts = texts.replace(p, '')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': '', '\r': ''} # 대체하고싶은거 알아서 추가하기
    for s in specials:
        texts = texts.replace(s, specials[s])
    texts = texts.strip() #빈칸 삭제

    return texts

def clean_text(texts):
    review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\☆\★\♡\♥\^\"]', '', texts)  # remove punctuation
    review = re.sub(r'([ㄱ-ㅎㅏ-ㅣ]+)', '', review)
    review = re.sub(r'\s+', ' ', review)  # remove extra space
    review = re.sub(r'<[^>]+>', '', review)  # remove Html tags
    review = re.sub(r'\s+', ' ', review)  # remove spaces
    review = re.sub(r"^\s+", '', review)  # remove space from start
    review = re.sub(r'\s+$', '', review)  # remove space from the end
    # review = re.sub(r'\d+','', review)  # remove number : 숫자를 삭제하면 의미가 이상해져서 사용x
    # review = review.lower() # lower case

    # 이모티콘 제거
    emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    corpus = re.sub(emoji_pattern, "", review)
    return corpus

