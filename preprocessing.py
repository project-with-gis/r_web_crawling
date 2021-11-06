import re
import kss
import pandas as pd
from hanspell import spell_checker
from soynlp.normalizer import *
import requests
from bs4 import BeautifulSoup
from urllib.request import urlretrieve


### 커다란 기능 첫번째 함수
def basic_check(review):  # 한 행마다 실행되도록. 이 함수가 받아오는건 하나의 리뷰
    sentence_tokenized_text = []  # 문장 단위로 분리된 corpus가 저장
    review = review.strip()  # 문자열의 '맨앞'과, '맨뒤'의 띄어쓰기(' '), 탭('\t'), 엔터('\n') 제거
    for sent in kss.split_sentences(review):  # review 문장 단위로 분리시켜주는 듯
        sentence_tokenized_text.append(sent.strip())

    cleaned_corpus = []  # 불용어 정리된 문장 저장 (바꿀거 바꾸고 ..)
    for sent in sentence_tokenized_text:
        cleaned_corpus.append(clean_punc(sent))

    basic_preprocessed_corpus = clean_text(cleaned_corpus)

    return basic_preprocessed_corpus


def clean_punc(text):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }
    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])  # punct_mapping에 있는 변수가 있으면 대응되는걸로 변경되도록 한다.

    for p in punct:  # punct에 포함되는 변수가 있으면 그 변수의 양옆을 띄운다. 2=4 -> 2 = 4
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # punct_mapping에 없지만 추가로 대체하고싶은 변수 정의
    for s in specials:
        text = text.replace(s, specials[s])

    return text.strip()


def clean_text(texts): # 우리가 쓸 땐 주로 공백제거하는 기능 뿐인듯. 있어야할까 ?
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '', str(texts[i]))
        review = re.sub(r'\d+', '', str(texts[i]))  # remove number ## 숫자제거 질문) 숫자를 제거해야될까요 ?
        review = review.lower()  # lower case ## 소문자로 바꾸기 ## 이것도 우린 필요없는 듯
        review = re.sub(r'\s+', ' ', review)  # remove extra space ## 공백문자제거
        review = re.sub(r'<[^>]+>', '', review)  # remove Html tags
        review = re.sub(r"^\s+", '', review)  # remove space from start ## ^ : 문자열의 제일 앞 부분과 일치함을 의미
        review = re.sub(r'\s+$', '', review)  # remove space from the end ## $ : 문자열의 제일 끝 부분과 일치함을 의미
        corpus.append(review)
    return corpus


### 커다란 기능 두번째 함수
def spell_check_text(texts):  # 한 댓글에 대한 문장들
    lownword_map = make_dictionary()  # 외래어 사전
    corpus = []
    for sent in texts:
        spelled_sent = spell_checker.check(sent)  # 띄어쓰기, 맞춤법
        checked_sent = spelled_sent.checked
        normalized_sent = repeat_normalize(checked_sent)  # 반복되는 이모티콘이나 자모를 normalization
        for lownword in lownword_map:  # 왜래어 바꿔줌 (miss spell -> origin spell)
            normalized_sent = normalized_sent.replace(lownword, lownword_map[lownword])
        corpus.append(normalized_sent)
    return corpus


def make_dictionary():  ## 질문) 파이참에서 데이터 바로 받아서 딕셔너리 만들고싶은데 데이터 다운이 안된다
    lownword_map = {}
    lownword_data = open('C:/Users/MIN JEONG JO/OneDrive/바탕 화면/confused_loanwords.txt', 'r', encoding='utf-8')
    lines = lownword_data.readlines()
    for line in lines:
        line = line.strip()
        miss_spell = line.split('\t')[0]
        ori_word = line.split('\t')[1]  # line 뽑아보면 \t기준 뒤에꺼가 original
        lownword_map[miss_spell] = ori_word
    return lownword_map


def delete_null(df):  # 리뷰없는 데이터 삭제
    new_df = df.astype({'review': 'str'})
    new_df = new_df[new_df.review != 'nan']
    new_df.reset_index(drop=True, inplace=True)
    return new_df
