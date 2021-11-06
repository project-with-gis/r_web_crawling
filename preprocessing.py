import pandas as pd
import numpy as np
import re
import kss
from hanspell import spell_checker

# # 형식 변환 # #
# 평점 반올림 하는 함수
def Score_roundUp(df):

  # NaN값 0으로 대체
  df['score'] = df['score'].fillna(0)

  # Series → 리스트 변환 후 반올림
  score_list = df['score'].values.tolist()
  for i in range(len(score_list)):
    if score_list[i] - int(score_list[i]) >= 0.5:
      score_list[i] = int(score_list[i]) + 1
    else:
      score_list[i] = int(score_list[i])

  # 리스트 → Series
  col_name = ['score']
  df['score'] = pd.DataFrame(score_list, columns=col_name)

  return df


# 컬럼 순서 변경 함수
def columns_change(df):

  df = df[['store_id', 'portal_id', 'date', 'score', 'review']]

  return df


# 날짜 형식 변환 함수(수정필요)
def change_date(df):
    # 요일 제거
    for i, line in enumerate(df['date']):
        line = line.rstrip('.월화수목금토일')
        df['date'][i] = line

    # 년도 추가
    for i in range(len(df)):
        if len(df['date'][i]) < 8:
            new_str = []
            new_str.append("21.")
            new_str.append(df['date'][i])
            new_str = ''.join(new_str)
            df['date'][i] = new_str

    return df


# # 1.Basic Preprocessing # #

def tokenizer(review):
    sentence_tokenized_text = []
    for i, line in enumerate(review):
        line = line.strip()
        for sent in kss.split_sentences(line):
            sentence_tokenized_text.append(sent.strip())

    cleaned_corpus = []
    for sent in sentence_tokenized_text:
        cleaned_corpus.append(clean_punc(sent))

    basic_preprocessed_corpus = clean_text(cleaned_corpus)

    return basic_preprocessed_corpus


def clean_punc(text, punct, mapping):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }

    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text.strip()


def clean_text(texts):
    def clean_text(texts):
        corpus = []
        for i in range(0, len(texts)):
            review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\☆\★\♡\♥\"]', '', str(texts[i]))  # remove punctuation
            review = re.sub(r'([ㄱ-ㅎㅏ-ㅣ]+)', '', review)
            review = re.sub(r'\s+', ' ', review)  # remove extra space
            review = re.sub(r'<[^>]+>', '', review)  # remove Html tags
            review = re.sub(r'\s+', ' ', review)  # remove spaces
            review = re.sub(r"^\s+", '', review)  # remove space from start
            review = re.sub(r'\s+$', '', review)  # remove space from the end
            # review = re.sub(r'\d+','', review)  # remove number : 숫자로 인한 의미 변화로 사용X
            # review = review.lower()  #lower case : 영어리뷰 사용하지 않기 때문에 사용X

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


# # 2.Spell Check # #



