import pandas as pd
import requests
import json
# import psycopg2
import datetime
from tqdm import tqdm
import re
import os


# headers 주기적으로 갱신해줘야함
def siksin_review_scraping(store_info):
    auth = store_info['auth'][0]
    headers = {
        'authority': 'api.siksinhot.com',
        'sec-ch-ua': '"Chromium";v="94", "Google Chrome";v="94", ";Not A Brand";v="99"',
        'accept': 'application/json, text/plain, */*',
        'siksinoauth': auth,    #store_info에서 가져옴-인증키
        'sec-ch-ua-mobile': '?0',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
        'sec-ch-ua-platform': '"Windows"',
        'origin': 'https://www.siksinhot.com',
        'sec-fetch-site': 'same-site',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': 'https://www.siksinhot.com/',
        'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    }

    check = []
    df = pd.DataFrame(columns=['store_id', 'portal_id', 'review_score', 'review', 'write_date'])
    for i in tqdm(range(len(store_info))):
        store_id = store_info.loc[i]['store_id']
        store_num = store_info.loc[i]['s_link']
        print(store_id, store_num)

        idx = 0
        num = 0
        while 1:
            params = (
                ('idx', idx),
                ('limit', '100'),
                ('sort', 'T'),
            )

            response = requests.get(f'https://api.siksinhot.com/v1/hp/{store_num}/review', headers=headers, params=params)
            response = json.loads(response.text)

            if response['data']['list'] == None:
                # print(response['data']['list'])
                check.append(store_num)
                break

            l = (len(response['data']['list']))
            cnt = response['data']['cnt']

            for i in range(l) :
                review_list = response['data']['list'][i]
                review = review_list['storyContents']
                score = review_list['score']
                timestamp = review_list['writeDt']
                write_date = datetime.datetime.fromtimestamp(timestamp/1000)
                portal_id = 1001
                num += 1

                data = [store_id, portal_id, score, review, write_date]
                df = df.append(pd.Series(data, index=df.columns), ignore_index=True)

            if num == cnt:
                break
            else:
                idx += 100

    return df



def add_siksin_info(store_info):
    headers = {
        'authority': 'www.siksinhot.com',
        'sec-ch-ua': '"Chromium";v="94", "Google Chrome";v="94", ";Not A Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-user': '?1',
        'sec-fetch-dest': 'document',
        'referer': 'https://www.siksinhot.com/',
        'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    df = pd.DataFrame(columns=['store_id', 'portal_id', 's_link', 'auth', 'website', 'img_url'])


    for i in tqdm(range(len(store_info))):
        store_id = store_info.loc[i]['store_id']
        s_link = store_info.loc[i]['s_link']

        response = requests.get(f'https://www.siksinhot.com/P/{s_link}', headers=headers)
        response = response.text


        s = re.search("window.__INITIAL_STATE__ =", response)
        s = s.end()
        e = re.search('</head>', response)
        e = e.start()
        data = response[s:e]
        data = data.replace('</script>','')
        data = json.loads(data)

        portal_id = 1001
        auth = data['headers']['siksinOauth']
        img_url = data['meta']['metaImg']
        # store_tel = data['meta']['metaOgPlace'][1]['content']
        website = data['hotPlaceDetail']['homepage']

        data = [store_id, portal_id, s_link, auth, img_url, website]
        df = df.append(pd.Series(data, index=df.columns), ignore_index=True)
    return df