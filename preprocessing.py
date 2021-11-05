import re
import kss
import pandas as pd
from hanspell import spell_checker


# for i in range(len(df)): # main함수에서 이렇게 돌면 될듯
#   review = df['review'][i]
#   basic_check(review)

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
        text = text.replace(p, punct_mapping[p])  ## punct_mapping에 있는 변수가 있으면 대응되는걸로 변경되도록 한다.

    for p in punct:  ## punct에 포함되는 변수가 있으면 그 변수의 양옆을 띄운다. 2=4 -> 2 = 4
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  ## punct_mapping에 없지만 추가로 대체하고싶은 변수 정의
    for s in specials:
        text = text.replace(s, specials[s])

    return text.strip()


def clean_text(texts):  # 불용어 제거
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',
                        str(texts[i]))  # remove punctuation # “[abc]”라는 정규표현식의 의미는 “a 또는 b 또는 c 중 일치”이다.
        review = re.sub(r'\d+', '', str(texts[i]))  # remove number ## 숫자제거
        review = review.lower()  # lower case ## 소문자로 바꾸기
        review = re.sub(r'\s+', ' ', review)  # remove extra space ## 공백문자제거
        review = re.sub(r'<[^>]+>', '', review)  # remove Html tags
        review = re.sub(r"^\s+", '', review)  # remove space from start ## ^ : 문자열의 제일 앞 부분과 일치함을 의미
        review = re.sub(r'\s+$', '', review)  # remove space from the end ## $ : 문자열의 제일 끝 부분과 일치함을 의미
        corpus.append(review)
    return corpus

## 다음함수로 넘어감
def spell_check(df): # 띄어쓰기, 맞춤법, 이모티콘
    print("전처리하는 함수")


