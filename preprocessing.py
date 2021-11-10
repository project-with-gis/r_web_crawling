from dateutil.parser import parse
import re
import os
import datetime
import pandas as pd
from hanspell import spell_checker
from soynlp.normalizer import *

# 특정행을 기준으로 null값이 있으면 해당 행을 삭제
def remove_nan(df,subset):
    df.dropna(subset=subset, inplace=True)
    df = df.reset_index(drop=True)
    return df

# 전처리 후 ''리뷰가 비어있는 행 삭제
def remove_after_nan(total_review):
    for i, after_review in enumerate(total_review['preprocessed_review']):
        if after_review == '':
            total_review = total_review.drop(total_review.index[i])
            total_review.reset_index(drop=True)
    return total_review

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
    print(google_review_data)
    for i, review in enumerate(google_review_data['review']):
        if type(review) != 'str':
            review = str(review)
        if "Google 번역 제공" in review:
            print(review)
            google_review_data.drop(index=i, inplace=True)
    google_review_data.reset_index(drop=True)
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


#######################공통으로쓰는 전처리 함수###########################
# 전처리과정 전부 진행하는 함수
# 리뷰하나가 전체 전처리과정을 돌고 -> 리스트에 전처리 후 리뷰들이 하나씩 리스트에 담긴다
def prepro(review_list):
    after_review_total = []
    for i, one_review in enumerate(review_list):
        print(i, "=======================================")
        print(one_review)
        after_basic_check = basic_check(one_review)
        print(after_basic_check)
        after_spell_check = spell_check_text(after_basic_check)
        print(after_spell_check)
        after_review_total.append(after_spell_check)

    return after_review_total


# 가장 기초적인 전처리
# html tag 제거
# 숫자 제거
# Lowercasing
# "@%*=()/+ 와 같은 punctuation 제거
def basic_check(review):  # 한 행마다 실행되도록. 이 함수가 받아오는건 하나의 리뷰
    cleaned_corpus = clean_punc(review)
    basic_preprocessed_corpus = clean_text(cleaned_corpus)
    return basic_preprocessed_corpus

# 사전 기반의 오탈자 교정
# 줄임말 원형 복원 (e.g. I'm not happy -> I am not happy)
def spell_check_text(texts): # 한 댓글에 대한 문장들
    lownword_map = make_dictionary() # 외래어 사전
    spelled_sent = spell_checker.check(texts) # 띄어쓰기, 맞춤법
    checked_sent = spelled_sent.checked
    normalized_sent = repeat_normalize(checked_sent) # 반복되는 이모티콘이나 자모를 normalization
    for lownword in lownword_map: # 외래어 바꿔줌 (miss spell -> origin spell)
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
        texts = texts.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': '', '\r': ''} # 대체하고싶은거 알아서 추가하기
    for s in specials:
        texts = texts.replace(s, specials[s])
    texts = texts.strip() #빈칸 삭제

    return texts

def clean_text(line):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\|\.\:\;\!\,\_\~\$\'\"\(\)\♥\♡\ㅋ\ㅠ\ㅜ\ㄱ\ㅎ\ㄲ\ㅡ\?\^\!\-]', '',str(line)) #remove punctuation
    # review = re.sub(r'\d+','', review)# remove number# remove number
    # review = review.lower() #lower case
    review = re.sub(r'~', '에서', review)  #50~60대 에서 ~
    review = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]', '', review)  # 한글 ㅎㅎ,ㅜ,ㅣ 등 오탈자 제거
    review = re.sub(r'[a-zA-Z]', '', review) #영어 제거
    review = re.sub(r'\s+', ' ', review) #remove extra space
    review = re.sub(r'<[^>]+>','',review) #remove Html tags
    review = re.sub(r'\s+', ' ', review) #remove spaces
    review = re.sub(r"^\s+", '', review) #remove space from start
    review = re.sub(r'\s+$', '', review) #remove space from the end
    review = emoticon_normalize(review, num_repeats=2) #하하, 이모티콘 등 제거
    review = emoji_pattern.sub(r'', review) #이모지 제거

    return review

