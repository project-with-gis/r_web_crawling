from tqdm import tqdm
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

def diningcode_link(info_df):
    info_df.reset_index(drop=True, inplace=True)

    id_list = []
    url_list = []
    openHours_list = []
    # 데이터프레임에서 필요한 정보 가져옴
    for i in tqdm(range(len(info_df))):
        store_id = info_df['store_id'][i]
        store_name = info_df['store_name'][i]
        store_addr = info_df['store_addr'][i]
        store_tel = info_df['store_tel'][i]

        name = store_name

        # print(store_id, '검색어 :', name)  # 검색어
        # print(store_id, 'db 정보 :', store_name, '/', store_addr, '/', store_tel)

        # search
        response = requests.get('https://www.diningcode.com/list.php?query=' + name + '&rn=1') # 검색창에 음식점 쳐서 들어가기
        response_text = response.text
        soup = BeautifulSoup(response_text, 'html.parser')
        soup = soup.select('div.localeft-cont > div > ul > li')

        # place
        for store_info in soup:
            try:
                if store_info.select('a') != []:
                    profile_url = store_info.select_one('a').get('href')
                    response = requests.get('https://www.diningcode.com' + profile_url)
                    profile_soup = BeautifulSoup(response.text, 'html.parser')
                    soup_info = profile_soup.select_one('div.basic-info')

                    # dc_name = profile_soup.select_one('div.tit-point > p.tit').get_text()
                    addr = soup_info.select_one('li.locat').get_text()
                    tel = soup_info.select_one('li.tel').get_text()

                    test_addr = ' '.join(store_addr.split()[2:4])

                    try:
                        if test_addr[2].isdigit(): # 문자열이 숫자로만 이루어져 있는지 확인하는 함수
                            test_addr = test_addr[:2] + test_addr[3:]
                    except:
                        pass

                    openHours_tmp = []
                    if test_addr in addr and tel == store_tel:
                        open_info = profile_soup.select_one('div.busi-hours').select_one('ul.list')
                        for i, j in zip(open_info.select('p.l-txt'), open_info.select('p.r-txt')):
                            openHours_tmp.append((i.get_text().strip(), j.get_text().strip()))
                        id_list.append(store_id)
                        url_list.append(profile_url.replace('/profile.php?rid=', ''))
                        openHours_list.append(openHours_tmp)
                    elif tel == store_tel:
                        open_info = profile_soup.select_one('div.busi-hours').select_one('ul.list')
                        for i, j in zip(open_info.select('p.l-txt'), open_info.select('p.r-txt')):
                            openHours_tmp.append((i.get_text().strip(), j.get_text().strip()))
                        id_list.append(store_id)
                        url_list.append(profile_url.replace('/profile.php?rid=', ''))
                        openHours_list.append(openHours_tmp)
            except:
                pass

    link_df = pd.DataFrame({'store_id': id_list, 'open_hours': openHours_list, 'd_link': url_list})
    link_df.insert(1, 'portal_id', 1003)

    return link_df

def diningcode_review(link_df):
    link_df = link_df[link_df['d_link'].isnull() == False]
    link_df = link_df.reset_index(drop=True)

    review_df = pd.DataFrame(columns=['store_id', 'date', 'score', 'review']) # 빈파일 생성
    for i in tqdm(range(len(link_df))):
        store_id = link_df['store_id'][i]
        profile_url = '/profile.php?rid=' + link_df['d_link'][i]

        response = requests.get('https://www.diningcode.com' + profile_url) #d_link 이용해서 각 음식점 접속
        profile_soup = BeautifulSoup(response.text, 'html.parser')

        data = profile(store_id, profile_soup, profile_url)
        review_df = pd.concat([review_df, data], ignore_index=True)

    review_df.insert(1, 'portal_id', 1003)

    return review_df

def profile(store_id, soup, profile_url):
# place & review
    review_score_list = []
    review_contents_list = []
    review_date_list = []

    soup_review = soup.select_one('div.appraisal')

    try:
        # review_count = soup_review.select_one('p.tit').get_text()
        # score_avg = float(soup_review.select_one('div.grade-info > p#lbl_star_point > span.point').get_text().replace('점', ''))

        review = soup_review.select('div#div_review > div.latter-graph')

        for index, i in enumerate(review):
            review_score = int(0.05 * int(''.join(re.findall(r'\d+', i.select_one('p.person-grade > span.star-date > i.star > i')['style']))))
            review_contents = i.select_one('p.btxt').get_text()

            review_date = i.select_one('p.person-grade > span.star-date > i.date').get_text()
            if len(review_date.split()) == 2:
                review_date = datetime.strptime(review_date, '%m월 %d일')
                review_date = review_date.strftime('2021-%m-%d')
            else:
                review_date = datetime.strptime(review_date, '%Y년 %m월 %d일')
                review_date = review_date.strftime('%Y-%m-%d')

            review_score_list.append(review_score)
            if review_contents.strip() == '':
                review_contents_list.append(None)
            else:
                review_contents_list.append(review_contents)
            review_date_list.append(review_date)

# review
        count = 2
        while(1):
            profile_url = profile_url.replace('/profile.php?rid=', '')
            data = {
                'mode': 'LIST',
                'type': 'profile',
                'v_rid': profile_url,
                'page': count,
                'rows': '5'
            }

            response = requests.post('https://www.diningcode.com/2018/ajax/review.php', headers = {'referer': 'https://www.diningcode.com/profile.php?rid=' + profile_url,}, data=data)
            if response.text == '':
                break
            else:
                response_text = response.text
                soup = BeautifulSoup(response_text, 'html.parser')
                review = soup.select('div.latter-graph')
                for index, i in enumerate(review):
                    review_score = int(0.05 * int(''.join(re.findall(r'\d+', i.select_one('p.person-grade > span.star-date > i.star > i')['style']))))
                    review_contents = i.select_one('p.btxt').get_text()

                    review_date = i.select_one('p.person-grade > span.star-date > i.date').get_text()
                    if len(review_date.split()) == 2:
                        review_date = datetime.strptime(review_date, '%m월 %d일')
                        review_date = review_date.strftime('2021-%m-%d')
                    else:
                        review_date = datetime.strptime(review_date, '%Y년 %m월 %d일')
                        review_date = review_date.strftime('%Y-%m-%d')

                    review_score_list.append(review_score)
                    if review_contents.strip() == '':
                        review_contents_list.append(None)
                    else:
                        review_contents_list.append(review_contents)
                    review_date_list.append(review_date)

                count += 1
    except:
        pass

    df = pd.DataFrame({'date': review_date_list, 'score': review_score_list, 'review': review_contents_list})
    df.insert(0, 'store_id', store_id)

    return df