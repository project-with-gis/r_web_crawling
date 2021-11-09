from datetime import datetime

import pandas as pd
from tqdm import tqdm

from csv_handler import read_csv, save_csv


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



def basic_preprocessing(data): #문장 조각조각
    # print(data['review'][i]) #줄바꿈 확인용
    line = list(data['review'][i].strip().replace('\r', '').replace('\n', '')) #엔터미리 제거, \r\n, \n 상관없이 가능
    # line = kss.split_sentences(new) #왜 굳이 문장문장 조각낼까?
    line = ''.join(line).strip()
    # print(type(line))
    return line #line은 list형태


def clean_punc(line): #문장부호같은거 다 삭제
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&' + 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ' + 'ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅔㅢㅟㅚㅞㅙㅝㅘ' #웃음같은 자음만 있는거 제거 추가

    mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }
    # for sent in line:
    sent = line[:]
    for p in mapping:
        sent = sent.replace(p, '')
    # print(mapping[p])

    for p in punct:
        sent = sent.replace(p, '')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        sent = sent.replace(s, '')
    line = sent.strip() #빈칸 삭제
    print(line)
    return line


def spell_check(line):
    spelled_sent = spell_checker.check(line)
    checked_sent = spelled_sent.checked

    # print(checked_sent)
    return checked_sent


def normalizer(sent):
    review = repeat_normalize(sent, num_repeats=2)
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
    # print(loanword_map)
    return loanword_map


def loanword_corrector(sent, map):
    for loan in map:
        corr = sent.replace(loan, map[loan])
    return corr


def preprocessing_all_in_one(path, name):
    lines = []
    map = loanword_dic_open()
    data = read_csv('./data/siksin_전전처리_1107.csv', index=False)[:200]
    for i in tqdm(range(len(data.index))):
        basic = basic_preprocessing(data)
        punc = clean_punc(basic)
        review = spell_check(punc)
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
        basic = basic_preprocessing(data)
        punc = clean_punc(basic)
        review = spell_check(punc)
        fin = normalizer(review)
        lw = loanword_corrector(review, map)
        lines.append(lw)
    # print(lines)
    data['preprocessed_review'] = lines
    print(data)
    save_csv(data,'./data', 'siksin_전처리_1109.csv')
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
