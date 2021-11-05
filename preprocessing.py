import kss
from hanspell import spell_checker
import re


# 특정행을 기준으로 null값이 있으면 해당 행을 삭제
def remove_nan(df,subset):
    df.dropna(subset=subset, inplace=True)
    df = df.reset_index(drop=True)
    return df


def sentence_tokenized(lines):
    sentence_tokenized_text = []
    for i, line in enumerate(lines):
        line = line.strip()
        for sent in kss.split_sentences(line):
            sentence_tokenized_text.append(sent.strip())

    return sentence_tokenized_text


def clean_punc_1(lines):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }
    for p in punct_mapping:
        text = lines.replace(p, punct_mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])


    return text.strip()

def clean_punc_2(sentence_tokenized_text):
    cleaned_corpus = []
    for text in sentence_tokenized_text:
        cleaned_corpus.append(clean_punc_1(text))
    # for i in range(0, 10):
    #     print(cleaned_corpus[i])
    return cleaned_corpus

def clean_text(texts):
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\|\.\:\;\!\,\_\~\$\'\"\(\)\♥\♡\ㅋ\ㅠ\ㅜ\ㄱ\ㅎ\ㄲ\ㅡ\?\^\!\-]', '',str(texts[i])) #remove punctuation
        review = re.sub(r'\d+','', str(texts[i]))# remove number# remove number
        review = review.lower() #lower case
        review = re.sub(r'~', '', review)  #50~60대 에서 ~ 제거
        review = re.sub(r'[a-zA-Z]', '', review) #영어 제거
        review = re.sub(r'원문', '', review)
        review = re.sub(r'Google 번역 제공', '', review)
        review = re.sub(r'\s+', ' ', review) #remove extra space
        review = re.sub(r'<[^>]+>','',review) #remove Html tags
        review = re.sub(r'\s+', ' ', review) #remove spaces
        review = re.sub(r"^\s+", '', review) #remove space from start
        review = re.sub(r'\s+$', '', review) #remove space from the end
        corpus.append(review)
    return corpus

# 맞춤법 검사 및 띄어쓰기
def sent_check(sent):
    spelled_sent = spell_checker.check(sent)
    checked_sent = spelled_sent.checked


# 구글 사이트/ 영어나, 번역된 리뷰 제거
def google_eng_transfer_del(google_review_data):
    for i, review in enumerate(google_review_data['review']):
        if "번역" in review:
            google_review_data = google_review_data.drop(google_review_data.index[i])
        elif "원문" in review:
            google_review_data = google_review_data.drop(google_review_data.index[i])

    return google_review_data