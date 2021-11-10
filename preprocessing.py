import re
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from csv_handler import read_csv, save_csv
from pre2 import basic_preprocessing


def remove_english(df):
    print("전처리하는 함수")


def sik_sin_transform_datetime(path, int): #날짜 먼저 형식 바꾸고 컬럼위치 바꾸기 주의
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

#----------------------------------------------------------------한국어전처리
import kss
from hanspell import spell_checker
from csv_handler import read_csv, save_csv
from soynlp.normalizer import *



def basic_preprocessing(data,i): #문장 조각조각
    # print(data['review'][i]) #줄바꿈 확인용
    line = list(str(data['review'][i]))
    # print(i)#엔터미리 제거, \r\n, \n 상관없이 가능
    # line = kss.split_sentences(new) #왜 굳이 문장문장 조각낼까?
    line = ''.join(line).strip()
    # print(type(line))
    return line #line은 list형태


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
    review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\☆\★\♡\♥\^\"]', '', texts)  # remove punctuation ~는 에서로 바꾸는게 낫지 않나
    review = re.sub(r'([ㄱ-ㅎㅏ-ㅣ]+)', '', review)
    review = re.sub('r([a-zA-Z]+)', '', review)
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

def spell_check(line):
    spelled_sent = spell_checker.check(line)
    checked_sent = spelled_sent.checked

    # print(type(checked_sent))
    return checked_sent


def normalizer(sent):
    review = repeat_normalize(sent, num_repeats=2)
    # print(type(review))
    return review


def loanword_dic_open():
    loanword_map = {}
    loanword_data = open('./data/confused_loanwords.txt', 'r', encoding='UTF-8')
    word = loanword_data.readlines()

    for s in word:
        s = s.strip()
        miss_spell = s.split('\t')[0]
        ori_word = s.split('\t')[1]
        loanword_map[miss_spell] = ori_word
    # print(type(loanword_map))
    return loanword_map


def loanword_corrector(sent, map):
    for loan in map:
        corr = sent.replace(loan, map[loan]).rstrip()
    # print(type(corr))
    return corr


def preprocessing_all_in_one(path, name):
    lines = []
    map = loanword_dic_open()
    data = read_csv('./data/siksin_전전처리_1107.csv', index=False)[:200]
    for i in tqdm(range(len(data.index))):
        basic = basic_preprocessing(data,i)
        punc = clean_punc(basic)
        review = spell_check(punc.encode(encoding='UTF-8'))
        # fin = normalizer(review)
        lw = loanword_corrector(review, map)
        lines.append(lw)
    # print(lines)
    data['preprocessed_review'] = lines
    # print(data)
    save_csv(data, path, name)


def delete_null(df):  # 리뷰없는 데이터 삭제 함수
    new_df = df.astype({'review': 'str'})
    new_df = new_df[new_df.review != 'nan']
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def delete_row(c_df):
    # score와 review에서 결측값인 행 삭제
    new_df = c_df.dropna(axis=0)

    # 리뷰 수를 줄이기 위해 .만 작성한 리뷰도 삭제
    df = new_df.astype({'review': 'str'})
    df = df[new_df.review != '.']
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == '__main__': #한국어 전처리 메인함수
    # preprocessing_all_in_one('./data', 'siksin_전처리_1108')
    lines = []
    map = loanword_dic_open()
    data = read_csv('./data/siksin_전전처리_1107.csv')
    # data = delete_row(data)
    for i in tqdm(range(len(data.index))):
        basic = basic_preprocessing(data,i)
        # print(i)
        punc = clean_punc(basic)
        chan = clean_text(punc)
        review = spell_check(chan)
        fin = normalizer(review)
        lw = loanword_corrector(fin, map)
        lines.append(lw)
    # print(lines)
    data['preprocessed_review'] = lines
    print(data)
    save_csv(data,'./data', 'siksin_전처리_1110_2.csv')
    # csv = read_csv('./data/siksin_1review_test.csv')
    # print(csv.head())
#
# if __name__ == '__main__': #날짜형식, 반올림, 컬럼 순서 바꿔주기 메인함수
#     raw = read_csv('./data/naver_review.csv')
#     date = transform_datetime_df(raw, 5)
#     print(date)
#     score = rounding_off_scores_df(date, 2)
#     col = swap_columns_with_num_df(score, 0,1,4,2,3)
#     save_csv(col, './data', 'siksin_전전처리_1107.csv')
