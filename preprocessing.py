import kss
from hanspell import spell_checker
import re
from soynlp.normalizer import *  # 이모티콘 하하하 등 제거



# 특정행을 기준으로 null값이 있으면 해당 행을 삭제
def remove_nan(df,subset):
    df.dropna(subset=subset, inplace=True)
    df = df.reset_index(drop=True)
    return df

# 전처리 후 리뷰가 '' 비어있는 상태인 행 삭제
def remove_after_nan(total_review):
    for i, after_review in enumerate(total_review['after_review']):
        if after_review == '':
            total_review = total_review.drop(total_review.index[i])
            total_review.reset_index(drop=True)
    return total_review


# 전처리과정 전부 진행하는 함수
def prepro_1(line):
    sentence_tokenized_review = sentence_tokenized(line)
    # print(sentence_tokenized_review)
    # print(len(sentence_tokenized_review))

    # 특수문자나 기호 사이 띄어짐
    cleaned_corpus = clean_punc(sentence_tokenized_review)
    # print("클린",cleaned_corpus)
    # print(len(cleaned_corpus))

    # 정규표현식을 사용한 특수문자 처리
    basic_preprocessed_corpus = clean_text(cleaned_corpus)
    # for i in range(len(basic_preprocessed_corpus)):
    #     print(basic_preprocessed_corpus[i])
    # print("베이직",basic_preprocessed_corpus)
    # print(len(basic_preprocessed_corpus))

    # 띄어쓰기, 맞춤법 검사
    checked_sent=(sent_check(basic_preprocessed_corpus))
    # print(checked_sent)
    # print(len(checked_sent))

    prepro_sent = lownword_check(checked_sent)
    sents=''
    for sent in prepro_sent:
        sents += sent
    # print("들어가기전",sents)

    return sents

# 전처리 진행 후 컬럼에 넣기 전에 리스트만들기
def prepro_2(review_list):
    after_review_total = []
    for i, one_review in enumerate(review_list):
        after_review = prepro_1(one_review)
        after_review_total.append(after_review)

    return after_review_total



def sentence_tokenized(lines):
    sentence_tokenized_text = []
    lines = lines.strip()
    for sent in kss.split_sentences(lines):
        # print(sent)
        sentence_tokenized_text.append(sent.strip())

    # print(sentence_tokenized_text)
    return sentence_tokenized_text


def clean_punc(lines):
    texts =[]
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
        texts.append(text)

    # print(texts)
    return texts



def clean_text(lines):
    texts=[]
    for line in lines:
        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\|\.\:\;\!\,\_\~\$\'\"\(\)\♥\♡\ㅋ\ㅠ\ㅜ\ㄱ\ㅎ\ㄲ\ㅡ\?\^\!\-]', '',str(line)) #remove punctuation
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
        review = emoji_pattern.sub(r'', review) #이모지 제거
        texts.append(review)
    # print(review)
    return texts



# 맞춤법 검사 및 띄어쓰기
def sent_check(sents):
    checked_texts =[]
    for sent in sents:
        spelled_sent = spell_checker.check(sent)
        checked_sent = spelled_sent.checked
        # print(checked_sent)
        checked_texts.append(sent)
    return checked_texts

# 외래어
def lownword():
    lownword_map = {}
    lownword_data = open('data/confused_loanwords.txt', 'r', encoding='utf-8') #외래어 사전 데이터
    lines = lownword_data.readlines()

    for line in lines:
        line = line.strip()
        miss_spell = line.split('\t')[0]
        ori_word = line.split('\t')[1]
        lownword_map[miss_spell] = ori_word

    return lownword_map

def lownword_check(sents):
    lownword_map = lownword()
    normalized_texts =[]
    for sent in sents:
        for l_word in lownword_map:
            normalized_sent = sent.replace(l_word, lownword_map[l_word])
        normalized_texts.append(normalized_sent)
    return normalized_texts






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
