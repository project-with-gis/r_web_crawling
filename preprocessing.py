import kss
from hanspell import spell_checker
import re
from soynlp.normalizer import *  # 이모티콘 하하하 등 제거



# 특정행을 기준으로 null값이 있으면 해당 행을 삭제
def remove_nan(df,subset):
    df.dropna(subset=subset, inplace=True)
    df = df.reset_index(drop=True)
    return df

# 전처리과정 전부 진행하는 함수
def prepro(line):
    sentence_tokenized_review = sentence_tokenized(line) #리스트 형태로 나옴

    # print(sentence_tokenized_review)
    # print(len(sentence_tokenized_review))

    # 특수문자나 기호 사이 띄어짐
    cleaned_corpus = clean_punc(sentence_tokenized_review) #리스트 형태로 나옴
    # print(cleaned_corpus)
    # print(len(cleaned_corpus))

    # 정규표현식을 사용한 특수문자 처리
    basic_preprocessed_corpus = clean_text(cleaned_corpus) #리스트 형태로 나옴
    # for i in range(len(basic_preprocessed_corpus)):
    #     print(basic_preprocessed_corpus[i])
    # print(basic_preprocessed_corpus)
    # print(len(basic_preprocessed_corpus))

    # 띄어쓰기, 맞춤법 검사
    checked_sent=(sent_check(basic_preprocessed_corpus))
    # print(checked_sent)
    # print(len(checked_sent))

    return checked_sent


def sentence_tokenized(lines):
    sentence_tokenized_text = []
    lines = lines.strip()
    print(lines)
    for sent in kss.split_sentences(lines):
        sentence_tokenized_text.append(sent.strip())

    return sentence_tokenized_text


def clean_punc(lines):
    for line in lines:
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                         "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                         '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                         '∅': '', '³': '3', 'π': 'pi', }
        for p in punct_mapping:
            text = line.replace(p, punct_mapping[p])

        for p in punct:
            text = text.replace(p, f' {p} ')

        specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
        for s in specials:
            text = text.replace(s, specials[s])

    print(text)
    return text



def clean_text(texts):

    review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\|\.\:\;\!\,\_\~\$\'\"\(\)\♥\♡\ㅋ\ㅠ\ㅜ\ㄱ\ㅎ\ㄲ\ㅡ\?\^\!\-]', '',str(texts)) #remove punctuation
    review = re.sub(r'\d+','', review)# remove number# remove number
    review = review.lower() #lower case
    review = re.sub(r'~', '', review)  #50~60대 에서 ~ 제거
    review = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]', '', review)  # 한글 ㅎㅎ,ㅜ,ㅣ 등 오탈자 제거
    review = re.sub(r'[a-zA-Z]', '', review) #영어 제거
    review = re.sub(r'원문', '', review)
    review = re.sub(r'Google 번역 제공', '', review)
    review = re.sub(r'\s+', ' ', review) #remove extra space
    review = re.sub(r'<[^>]+>','',review) #remove Html tags
    review = re.sub(r'\s+', ' ', review) #remove spaces
    review = re.sub(r"^\s+", '', review) #remove space from start
    review = re.sub(r'\s+$', '', review) #remove space from the end
    review = emoticon_normalize(review, num_repeats=2) #하하, 이모티콘 등 제거

    print(review)
    return review

# 맞춤법 검사 및 띄어쓰기
def sent_check(sent):
    spelled_sent = spell_checker.check(sent)
    checked_sent = spelled_sent.checked
    print(checked_sent)
    return checked_sent


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