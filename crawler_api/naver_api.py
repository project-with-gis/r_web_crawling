import pandas as pd
import requests
import json
import datetime
import time
from tqdm import tqdm
import re
import os
import random

def naver_store_id(store_info):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'content-type': 'application/json',
    }

    df = pd.DataFrame(columns=['store_id', 'portal_id', 'n_link'])
    for i in tqdm(range(len(store_info))):
        store_id = store_info.loc[i]['store_id']
        store_name = store_info.loc[i]['store_name']
        store_data = store_info.loc[i][['store_addr', 'store_addr_new', 'store_tel']]
        time.sleep(random.uniform(2, 10))

        for store in store_data:
            if store == None:
                continue
            place_str = store_name + ' ' + str(store)

            params = (
                ('sm', 'tab_hty.top'),
                ('where', 'nexearch'),
                ('query', place_str),
                ('tqi', 'hdBvIdprvmsssFWFcN8ssssssid-018697'),
            )

            response = requests.get('https://search.naver.com/search.naver', headers=headers, params=params)

            try:
                n_link = re.search('PlaceSummary:(.*)"', response.text).group(1)

                n_link = n_link.split('","typename')[0]
                # print(n_link, place_str)

                if re.search('id', n_link) != None:
                    check = n_link.split('":{"id":')
                    n_link = check[0]
                    # print(check, n_link)

                portal_id = 1004
                data = [store_id, portal_id, n_link]
                df = df.append(pd.Series(data, index=df.columns), ignore_index=True)
                # print(store_id, n_link)
                break
            except:
                continue

        # n_link = None
        if store_id not in df['store_id'].tolist():
            data = [store_id, portal_id, None]
            df = df.append(pd.Series(data, index=df.columns), ignore_index=True)

    return df

# 에러코드 429(Too many request) 주의...
def naver_review_crawling(store_info):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'content-type': 'application/json',
    }
    df = pd.DataFrame(columns=['store_id', 'portal_id', 'review_score', 'review', 'write_date'])

    for i in tqdm(range(len(store_info))):
        store_id = store_info.loc[i]['store_id']
        # print(store_id)
        n_link = store_info.loc[i]['n_link']
        theme_list = ['allTypes']

        data = '[{"operationName":"getVisitorReviews","variables":{"input":{"businessId":"12024487","businessType":"restaurant","item":"0","bookingBusinessId":null,"page":1,"display":1000000,"isPhotoUsed":false,"theme":"taste","includeContent":true,"getAuthorInfo":true}},"query":"query getVisitorReviews($input: VisitorReviewsInput) {////n visitorReviews(input: $input) {////n items {////n id////n rating////n author {////n id////n nickname////n from////n imageUrl////n objectId////n url////n review {////n totalCount////n imageCount////n avgRating////n __typename////n }////n __typename////n }////n body////n thumbnail////n media {////n type////n thumbnail////n __typename////n }////n tags////n status////n visitCount////n viewCount////n visited////n created////n reply {////n editUrl////n body////n editedBy////n created////n replyTitle////n __typename////n }////n originType////n item {////n name////n code////n options////n __typename////n }////n language////n highlightOffsets////n translatedText////n businessName////n showBookingItemName////n showBookingItemOptions////n bookingItemName////n bookingItemOptions////n __typename////n }////n starDistribution {////n score////n count////n __typename////n }////n hideProductSelectBox////n total////n __typename////n }////n}////n"},{"operationName":"getVisitorReviews","variables":{"id":"12024487"},"query":"query getVisitorReviews($id: String) {////n visitorReviewStats(input: {businessId: $id}) {////n id////n name////n review {////n avgRating////n totalCount////n scores {////n count////n score////n __typename////n }////n starDistribution {////n count////n score////n __typename////n }////n imageReviewCount////n authorCount////n maxSingleReviewScoreCount////n maxScoreWithMaxCount////n __typename////n }////n visitorReviewsTotal////n ratingReviewsTotal////n __typename////n }////n visitorReviewThemes(input: {businessId: $id}) {////n themeLists {////n name////n key////n __typename////n }////n __typename////n }////n}////n"},{"operationName":"getVisitorReviewPhotosInVisitorReviewTab","variables":{"businessId":"12024487","businessType":"restaurant","item":"0","theme":"taste","page":1,"display":10},"query":"query getVisitorReviewPhotosInVisitorReviewTab($businessId: String//u0021, $businessType: String, $page: Int, $display: Int, $theme: String, $item: String) {////n visitorReviews(input: {businessId: $businessId, businessType: $businessType, page: $page, display: $display, theme: $theme, item: $item, isPhotoUsed: true}) {////n items {////n id////n rating////n author {////n id////n nickname////n from////n imageUrl////n objectId////n url////n __typename////n }////n body////n thumbnail////n media {////n type////n thumbnail////n __typename////n }////n tags////n status////n visited////n originType////n item {////n name////n code////n options////n __typename////n }////n businessName////n __typename////n }////n starDistribution {////n score////n count////n __typename////n }////n hideProductSelectBox////n total////n __typename////n }////n}////n"},{"operationName":"getVisitorRatingReviews","variables":{"input":{"businessId":"12024487","businessType":"restaurant","item":"0","bookingBusinessId":null,"page":1,"display":10,"includeContent":false,"getAuthorInfo":true},"id":"12024487"},"query":"query getVisitorRatingReviews($input: VisitorReviewsInput) {////n visitorReviews(input: $input) {////n total////n items {////n id////n rating////n author {////n id////n nickname////n from////n imageUrl////n objectId////n url////n review {////n totalCount////n imageCount////n avgRating////n __typename////n }////n __typename////n }////n visitCount////n visited////n originType////n reply {////n editUrl////n body////n editedBy////n created////n replyTitle////n __typename////n }////n businessName////n status////n __typename////n }////n __typename////n }////n}////n"}]'
        t = re.search('12024487', data).group(0)
        data = data.replace(str(t), str(n_link))
        data = data.replace('////n', '')

        time.sleep(random.uniform(5,30))
        t = re.search('"theme":"(.*)","includeContent":true,', data).group(1)
        data = data.replace(t, theme_list[0])
        i = 0
        while 1:
            try:
                i += 1
                t1 = re.search('"page":', data).end()
                t2 = re.search(',"display"', data).start()
                data = data[:t1] + str(i) + data[t2:]
                # print(data)
                time.sleep(random.uniform(10,40))
                response = requests.post('https://pcmap-api.place.naver.com/graphql', headers=headers, data=data)
                if response.status_code != 200:
                    print(f"store_id  : {store_id}, error_code : {response.status_code}")

                cnt = 0
                try:
                    response = json.loads(response.text)

                    for ii in range(5):
                        if len(response[ii]['data']['visitorReviews']['items']) == 0:
                            continue

                        review_list = response[0]['data']['visitorReviews']['items']
                        total = response[0]['data']['visitorReviews']['total']

                        if total == 0:
                            continue

                        for j in range(len(review_list)):
                            review = review_list[j]['body']
                            review_hi = review_list[j]['highlightOffsets']
                            score = review_list[j]['rating']
                            write_date = review_list[j]['created']
                            cnt += 1
                            portal_id = 1004

                            data = [store_id, portal_id, score, review, write_date]
                            print(data)
                            df = df.append(pd.Series(data, index=df.columns), ignore_index=True)
                        print(len(review_list))


                except:
                    if cnt < 100:
                        break
                    continue

            except:
                print("리뷰 크롤링 종료; 귀찮지만 갯수 확인..♡")
                break

    return df

# if __name__=='__main__':
#
#     info_df = pd.read_csv('store_info.csv') # 경로변경
#     info_df = info_df[0:].reset_index(drop=True) # 인덱스 수정
#
#     # 네이버에서 음식점 경로값 크롤링
#     link_df = naver_store_id(info_df)
#     link_df = pd.concat([info_df, link_df['n_link']], axis=1)
#
#     # 네이버 음식점 경로값을 통해 리뷰 크롤링
#     path = './'
#     review_df_na = naver_review_crawling(link_df)
#     review_df_na.to_csv(os.path.join(path, 'naver_review.csv'), header=False, index=False)

