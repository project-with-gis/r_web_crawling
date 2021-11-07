import pandas as pd
import requests
import re
import json
import datetime
from tqdm import tqdm
import time
import kss
from hanspell import spell_checker
from soynlp.normalizer import *


# # 전처리 전 데이터 정리 # #
# 컬럼 순서 변경
def change_columns(df):
    df = df.loc[:, ['store_id', 'portal_id', 'date', 'score', 'review']]
    return df

# 결측값 제거
def delete_row(c_df):
    # score와 review에서 결측값인 행 삭제
    new_df = c_df.dropna(axis=0)

    # 리뷰 수를 줄이기 위해 .만 작성한 리뷰도 삭제
    df = new_df.astype({'review': 'str'})
    df = df[new_df.review != '.']
    df.reset_index(drop=True, inplace=True)

    return df

def score_roundUp(d_df):
    # 평점 반올림
    # Series → 리스트로 변환 후 반올림
    score_list = d_df['score'].values.tolist()

    for i in range(len(score_list)):
        if score_list[i] - int(score_list[i]) >= 0.5:
            score_list[i] = int(score_list[i]) + 1
        else:
            score_list[i] = int(score_list[i])

    # 리스트 → Series
    col_name = ['score']
    d_df['score'] = pd.DataFrame(score_list, columns=col_name)

    return d_df

# # 수정 필요
# def change_date(df):
#     # 요일 제거
#     for i, line in enumerate(df['date']):
#         line = line.rstrip('.월화수목금토일')
#         df['date'][i] = line
#
#     # 년도 추가
#     for i in range(len(df)):
#         if len(df['date'][i]) < 8:
#             new_str = []
#             new_str.append("21.")
#             new_str.append(df['date'][i])
#             new_str = ''.join(new_str)
#             df['date'][i] = new_str
#
#     return df


def tokenized_text(review):
    review = review.strip()
    rez = []
    for x in review:
        rez.append(x.replace("\n", ""))
    reviewstr = ''
    for x in rez:
        reviewstr += x
    print(reviewstr)

    sentence_tokenized_text = kss.split_sentences(reviewstr)
    print(sentence_tokenized_text)

    return sentence_tokenized_text


def cleaned_corpus(sentence_tokenized_text):
    cleaned_corpus = []
    for sent in sentence_tokenized_text:
        cleaned_corpus.append(clean_punc(sent))

    basic_preprocessed_corpus = clean_text(cleaned_corpus)

    return basic_preprocessed_corpus


def clean_text(texts):
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\☆\★\♡\♥\^\"]', '',
                        str(texts[i]))  # remove punctuation
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
        review = re.sub(emoji_pattern, "", review)
        corpus.append(review)
    return corpus


def clean_punc(text):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }
    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text.strip()


def spell_check_text(texts):
    lownword_map = lownword_dic()
    corpus = []
    for sent in texts:
        spelled_sent = spell_checker.check(sent)  # 띄어쓰기, 맞춤법
        checked_sent = spelled_sent.checked
        normalized_sent = repeat_normalize(checked_sent, num_repeats=2)  # 반복되는 이모티콘이나 자모를 normalization
        for lownword in lownword_map:  # 왜래어 변환
            normalized_sent = normalized_sent.replace(lownword, lownword_map[lownword])
        corpus.append(normalized_sent)
    return corpus


def lownword_dic():
    lownword_map = {}
    lownword_data = open(r'C:\Users\jeong\Downloads\confused_loanwords.txt', 'r', encoding='utf-8')
    lines = lownword_data.readlines()
    for line in lines:
        line = line.strip()
        miss_spell = line.split('\t')[0]
        ori_word = line.split('\t')[1]
        lownword_map[miss_spell] = ori_word
    return lownword_map

# 전처리 후 데이터 저장
def save_csv(df):
    df.to_csv("naver_preprocessed_review")
