import pandas as pd
import requests
import re
import json
import datetime
from tqdm import tqdm
import time
import random


def google(storeInfo, link, review):
    if link == True and review == True:
        all_reviews = pd.DataFrame(columns=['store_id', 'date', 'score', 'review'])  # 빈 데이터프레임 생성해서 넣음

        if 'website' not in storeInfo.columns:
            storeInfo['website'] = pd.Series()
        storeInfo['g_link'] = pd.Series()

        for i in tqdm(range(len(storeInfo))):
            time.sleep(random.uniform(1, 5))
            google_link, website, review_param, review_cnt = search_store(storeInfo['store_name'][i],
                                                                          storeInfo['store_addr'][i],
                                                                          storeInfo['store_addr_new'][i],
                                                                          storeInfo['store_tel'][i])
            if google_link:
                storeInfo.loc[i, 'g_link'] = google_link
            if website:
                storeInfo.loc[i, 'website'] = website

            if review_cnt != 0:
                reviews = collecting_reviews(review_param, review_cnt)
                reviews.insert(0, 'store_id', storeInfo['store_id'][i])
                all_reviews = pd.concat([all_reviews, reviews])

        storeInfo = storeInfo[['store_id', 'website', 'g_link']]
        all_reviews.insert(1, 'portal_id', 1002)

        return storeInfo, all_reviews

    elif link == True and review == False:
        if 'website' not in storeInfo.columns:
            storeInfo['website'] = pd.Series()
        storeInfo['g_link'] = pd.Series()

        for i in tqdm(range(len(storeInfo))):
            google_link, website, review_param, review_cnt = search_store(storeInfo['store_name'][i],
                                                                          storeInfo['store_addr'][i],
                                                                          storeInfo['store_addr_new'][i],
                                                                          storeInfo['store_tel'][i])
            if google_link:
                storeInfo.loc[i, 'g_link'] = google_link
            if website:
                storeInfo.loc[i, 'website'] = website

        storeInfo = storeInfo[['store_id', 'website', 'g_link']]
        return storeInfo

    elif link == False and review == True:
        all_reviews = pd.DataFrame(columns=['store_id', 'date', 'score', 'review'])  # 빈 데이터프레임 생성

        for i in tqdm(range(len(storeInfo))):
            if not pd.isna(storeInfo['g_link'][i]):
                review_param, review_cnt = find_review_param(storeInfo['g_link'][i])
                if review_cnt != 0:
                    reviews = collecting_reviews(review_param, review_cnt)
                    reviews.insert(0, 'store_id', storeInfo['store_id'][i])
                    all_reviews = pd.concat([all_reviews, reviews])

        all_reviews.insert(1, 'portal_id', 1002)

        return all_reviews


def search_store(store_name, store_addr, store_addr_new, store_tel):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36', }
    response = requests.get('https://www.google.com/maps/search/' + store_addr + ' ' + store_name, headers=headers)
    response_text = response.text

    p = re.compile('window.APP_INITIALIZATION_STATE=.*?[;]window')
    tmp = p.search(response_text).group()
    tmp = tmp.lstrip('window.APP_INITIALIZATION_STATE=').rstrip(';window')

    response_json = json.loads(tmp)
    response_json = response_json[3][2].lstrip(")]}'\n")
    response_json = json.loads(response_json)
    response_json = response_json[0][1]

    search_addr = search_tel = ''
    search_link = search_website = None
    review_param = None
    review_cnt = 0

    if pd.isna(store_addr_new): store_addr_new = ''
    if pd.isna(store_tel): store_tel = ''

    for searchInfo in response_json:
        searchInfo = searchInfo[-1]

        if len(searchInfo) < 3:
            continue
        try:
            if searchInfo[39] != None:
                search_addr = searchInfo[39]
            if searchInfo[178] != None:
                search_tel = searchInfo[178][0][0]

            if store_addr in search_addr or store_addr_new in search_addr or store_tel.replace('-',
                                                                                               '') == search_tel.replace(
                    '-', ''):
                search_link = searchInfo[42]

            if searchInfo[7] != None:
                search_website = searchInfo[7][0]

            if searchInfo[4] != None:
                review_cnt = int(
                    searchInfo[4][3][1].replace('리뷰 ', '').replace('개', '').replace(',', '').replace(' reviews',
                                                                                                     '').replace(
                        ' review', ''))

            if review_cnt != 0:
                if searchInfo[37][0] != None:
                    review_param = searchInfo[37][0][0][29]
        #                 elif len(searchInfo[52][0]) > 0:
        #                     review_param = searchInfo[52][0]

        except:
            if searchInfo[3][0] != None:
                search_addr = searchInfo[3][0]
            search_link = ''

        break

    return search_link, search_website, review_param, review_cnt


def find_review_param(g_link):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
    }

    response = requests.get(g_link, headers=headers)
    response_text = response.text
    p = re.compile('window.APP_INITIALIZATION_STATE=.*?[;]window')
    tmp = p.search(response_text).group()
    tmp = tmp.lstrip('window.APP_INITIALIZATION_STATE=').rstrip(';window')

    response_json = json.loads(tmp)
    response_json = json.loads(response_json[3][6].lstrip(")]}'\n"))[6]

    review_cnt = int(response_json[4][3][1].replace('리뷰 ', '').replace('개', '').replace(',', ''))

    review_param = None
    if review_cnt != 0:
        if response_json[37][0] != None:
            review_param = response_json[37][0][0][29]
    #         elif:
    #             len(response_json[52][0]) > 0:
    #             review_param = response_json[52][0]

    return review_param, review_cnt


def collecting_reviews(review_param, review_cnt):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
        'cookie': 'HSID=AMRdGxra6aILq24qJ; SSID=ATZs_aqQ7EvcpbM5K; APISID=KQX0wgCzrBW7f3ze/Arv0vL667kHiUNM6_; SAPISID=le_FFFc0jfjlpbgC/ARt8zG6Aa7xKMiwCx; __Secure-1PAPISID=le_FFFc0jfjlpbgC/ARt8zG6Aa7xKMiwCx; __Secure-3PAPISID=le_FFFc0jfjlpbgC/ARt8zG6Aa7xKMiwCx; SID=CggpmPyP1L6ziQOPOcoJhIb7Oln4dpSfmRp1Gl13CBPmuukoFdbm9ZpBTXesAZHRqX9l6w.; __Secure-1PSID=CggpmPyP1L6ziQOPOcoJhIb7Oln4dpSfmRp1Gl13CBPmuukohy1G0wHVZMJMPhffNNAQAA.; __Secure-3PSID=CggpmPyP1L6ziQOPOcoJhIb7Oln4dpSfmRp1Gl13CBPmuukodG53sHNYHVvlvgQWnCj2Ww.; OGPC=19022519-1:; SEARCH_SAMESITE=CgQI6ZMB; OTZ=6213684_20_20__20_; 1P_JAR=2021-10-25-03; NID=511=eOEKRKwN8vuDj06JJtR1PDL9j6B9B6utMr1Xcn643iT7zMigQVLBABhudknhUb_KpUu-Ac59IZVYrAsWJJ_Qlnqe3fvAcMuQ031owo1w56Q-csc9mTUdlipEiep-ZrAIDrAE-HwOwv0NUsQEvfR0iwSyRqqxYFGJBNC7bZZUvkwWU544AR_sdjLylyn7IMfyKRsntlp7XG_HS0ksFBvj8P5seb41nxK9P_o8NJS-mIzLby9rvD6CmS23aFzu; SIDCC=AJi4QfFuLZVsNlHYVeLYipI1GtaR-52AV50bcR5eD-tMKkr5xT-Z0V4XIU1TFPmuuzs_YT4TM0I-; __Secure-3PSIDCC=AJi4QfFTOD4Jrs-zWjCB09v39NFsJFAXL2YFibWie-MkCac2UPoEzc1y9jqjEuMbvt42gtlbqiUW',
    }

    iter_cnt = int(review_cnt / 10) + 1
    if review_cnt % 10 == 0:
        iter_cnt = int(review_cnt / 10)

    collected_reviews = []

    try:
        if len(review_param[0]) == 19:
            for i in range(iter_cnt):
                time.sleep(random.uniform(1, 10))
                pb = '!1m2!1y' + review_param[0] + '!2y' + review_param[1] + '!2m2!1i' + str(
                    i) + '0!2i10!3e2!4m5!3b1!4b1!5b1!6b1!7b1!5m2!1s7yd2YdunBZT_0ASUi4mICA!7e81'  # 3e1(관련성순) 20개씩 넘어옴 / 3e2 (최신순) 10개씩 넘어옴 -> 최신순으로 설정해야 모든 리뷰를 다 가져올 수 있음

                response = requests.get('https://www.google.com/maps/preview/review/listentitiesreviews',
                                        headers=headers, params=(('pb', pb),))
                response_text = response.text.lstrip(")]}'")
                response_json = json.loads(response_text)
                review_list = response_json[2]

                for reviewInfo in review_list:
                    review = reviewInfo[3]
                    if review != None:
                        review = review.replace('\n', ' ')
                    score = reviewInfo[4]
                    date = datetime.datetime.fromtimestamp(int(reviewInfo[27]) / 1000).strftime('%Y-%m-%d')
                    collected_reviews.append((date, score, review))
        else:
            for reviewInfo in review_param:
                review = reviewInfo[3]
                if review != None:
                    review = review.replace('\n', ' ')
                score = reviewInfo[4]
                date = datetime.datetime.fromtimestamp(int(reviewInfo[27]) / 1000).strftime('%Y-%m-%d')
                collected_reviews.append((date, score, review))
    except:
        None

    collected_reviews = pd.DataFrame(collected_reviews, columns=['date', 'score', 'review'])

    return collected_reviews






# if __name__=='__main__':
#     # 데이터 불러오기
#     info = pd.read_csv("C:/Users/aj878/OneDrive/바탕 화면/호용의후예/기업/웹크롤링/storeInfo_2.csv")
#     storeInfo = info[1800:1900].reset_index(drop=True)
#
#
#
#     # 필요한 정보 가져오기
#     storeInfo, all_reviews = google(storeInfo, True , True)
#
#     reviewconcat = pd.DataFrame(columns=['store_id', 'portal_id', 'date', 'score', 'review'])
#     reviewconcat = pd.concat([reviewconcat, all_reviews])
#     reviewconcat.to_csv("./data/google_1700_1800.csv", encoding='utf-8-sig')
# #     # 엑셀로 저장
# #     all_reviews.to_csv("google_9000.csv", encoding='utf-8-sig')