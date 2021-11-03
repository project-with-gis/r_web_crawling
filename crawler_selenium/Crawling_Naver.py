from lib2to3.pgen2 import driver

import requests
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup as bs

# 크롬드라이버 창 닫히는 오류 수정
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options,
                          executable_path=r'C:/Users/jeong/Downloads/chromedriver_win32/chromedriver.exe')

driver.get("https://map.naver.com/v5/search")

# 검색어 = store_name + store_addr
data = pd.read_csv(r'C:\Users\jeong\Downloads\store_info.csv', encoding='UTF-8')
df = pd.DataFrame(data)
keyword = df['store_name'] + df['store_addr']
store_name_val = keyword.values
store_name_list = store_name_val.tolist()

# 검색어 입력(우선 고서막국수만 사용)
# for i in store_name_list:
# print("식당: ", i)
a = '공항칼국수 서울 강서구 공항동 49-2'
naver_map_search_url = f"https://m.map.naver.com/search2/search.naver?query={a}&sm=hty&style=v5"
driver.get(naver_map_search_url)
time.sleep(2)

# 검색결과 클릭
driver.find_element_by_css_selector(
    '#ct > div.search_listview._content._ctList > ul > li > div.item_info > a.a_item.a_item_distance._linkSiteview > div').click()
time.sleep(2)

# 방문자 리뷰 클릭
driver.find_element_by_css_selector(
    '#app-root > div > div > div.place_detail_wrapper > div.place_section.no_margin.GCwOh > div > div > div._3XpyR > div > span:nth-child(2)').click()
time.sleep(3)

# 리뷰 더보기 클릭
while True:
    try:
        driver.find_element_by_tag_name('body').send_keys(Keys.END)
        time.sleep(1)
        driver.find_element_by_css_selector(
            '#app-root > div > div > div.place_detail_wrapper > div:nth-child(5) > div:nth-child(4) > div:nth-child(4) > div._2kAri > a > svg').click()
        time.sleep(1)
        driver.find_element_by_tag_name('body').send_keys(Keys.END)
        time.sleep(1)

    except NoSuchElementException:
        print('<< 더보기 버튼 모두 클릭 완료 >>')
        break
        time.sleep(10)

# 크롤링
full_html = driver.page_source
soup = bs(full_html, 'html.parser')

# 날짜 추출
content_date = soup.find_all('div', class_='ZvQ8X')
content_date = [dates.span.text for dates in content_date]

# 리뷰 추출
content_review = soup.find_all('span', class_='WoYOw')
content_review = [reviews.text for reviews in content_review]

# 평점 추출
content_score = soup.find_all('span', class_='_2tObC')
content_score = [scores.text for scores in content_score]

df1 = pd.DataFrame({"date": content_date, "score": content_score, "review": content_review})
total_df = pd.concat([total_df, df1])
















