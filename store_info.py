import requests
import json
import pandas as pd


def region_id(headers):
    response = requests.get('https://api.siksinhot.com/v1/hp/area', headers=headers)
    response = json.loads(response.text)

    region_dict = {}
    for i in range(17):
        id = response['data']['list'][i]['upHpAreaId']
        region = response['data']['list'][i]['upHpAreaTitle']
        region_dict[id] = region

    return region_dict

def areaId_crawling(area_id):
    params = (
        ('upHpAreaId', area_id),
    )

    response = requests.get('https://api.siksinhot.com/v1/hp/area', headers=headers, params=params)
    response = json.loads(response.text)

    l = (len(response['data']['list'][0]['list']))
    AreaId = []

    for i in range(l):
        review_list = response['data']['list'][0]['list'][i]
        area_id = review_list['hpAreaId']
        # area_title = review_list['hpAreaTitle']
        AreaId.append(area_id)

    # print(AreaId, len(AreaId))
    return AreaId

def store_crawling(headers, area_id):
    hpAreaId = areaId_crawling(area_id)

    df = pd.DataFrame(columns=['store_name', 'store_addr', 'store_addr_new', 'store_tel', 'store_x', 'store_y', 'store_category','store_food', 's_link'])

    for area_num in range(len(hpAreaId)):
        print(hpAreaId[area_num])
        idx = 0
        num = 0
        while 1:
            params = (
                ('idx', idx),
                ('limit', '100'),
                ('upHpAreaId', area_id),
                ('hpAreaId', hpAreaId[area_num]),
                ('lat', ''),
                ('lng', ''),
            )

            response = requests.get('https://api.siksinhot.com/v1/hp', headers=headers, params=params)
            response = json.loads(response.text)

            cnt = response['data']['cnt']
            l = len(response['data']['list'])

            for i in range(l):
                review_list = response['data']['list'][i]
                pid = review_list['pid']
                pname = review_list['pname']
                addr = review_list['addr']
                addr_new = review_list['addr2']
                lat = review_list['lat']
                lng = review_list['lng']
                cate = review_list['hpSchCateNm']
                cate_d = review_list['mcateNm']
                # score_avg = review_list['score']
                phone = add_phone(pid)

                num += 1

                info = [pname, addr, addr_new, phone, lat, lng, cate, cate_d, pid]
                df = df.append(pd.Series(info, index=df.columns), ignore_index=True)

            if cnt == num:
                break
            else:
                idx += 100
    return df

def add_phone(pid):
    phone_response = requests.get(f'https://api.siksinhot.com/v1/hp/{pid}', headers=headers)
    phone_response = json.loads(phone_response.text)

    phone = phone_response['data']['phone']
    return phone

if __name__== "__main__" :
    headers = {
        'authority': 'api.siksinhot.com',
        'sec-ch-ua': '"Google Chrome";v="93", " Not;A Brand";v="99", "Chromium";v="93"',
        'accept': 'application/json, text/plain, */*',
        'siksinoauth': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjAsImlhdCI6MTYzMjcyMzY3NSwiZXhwIjoxNjMyODEwMDc1LCJpc3MiOiJzaWtzaW4ifQ.tJULWjX89MRDGwUomvING7rLAi6IQhUWzrJlv33Bnt4',
        'sec-ch-ua-mobile': '?0',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36',
        'sec-ch-ua-platform': '"Windows"',
        'origin': 'https://www.siksinhot.com',
        'sec-fetch-site': 'same-site',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': 'https://www.siksinhot.com/',
        'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    }

    region_dict = region_id(headers)
    store_info = pd.DataFrame(columns=['store_name', 'store_addr', 'store_addr_new', 'store_tel', 'store_x', 'store_y', 'store_category','store_food', 's_link','region'])

    for idx, region in region_dict.items():
        # print(idx, region)
        info = store_crawling(headers, idx)
        info['region'] = region
        store_info = store_info.append(info)
        store_info.reset_index(drop=True)

    # print(store_info)
    # store_info.to_csv('./store_info_test.csv', index=False)
