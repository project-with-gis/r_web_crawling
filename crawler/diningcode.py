from selenium import webdriver
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
import time

url = "https://www.diningcode.com/"  # 다이닝코드 url

# 웹 드라이버
driver = webdriver.Chrome('C:/Users/MIN JEONG JO/Downloads/chromedriver_win32/chromedriver.exe')
driver.get(url)  # 실행

driver.find_element_by_xpath('//*[@id="div_popup"]/div/div[3]').click()  # 광고창 닫기

# 검색창에 store_name & store_addr 치고 엔터
searchbox = driver.find_element_by_xpath('//*[@id="txt_keyword"]')
searchbox.send_keys("고성막국수서울")  # 일단 임의지정
searchbox.send_keys(Keys.ENTER)
time.sleep(5)

# 뜨는 식당중에서 첫번째 식당 클릭
driver.find_element_by_xpath('//*[@id="div_rn"]/ul/li').click()
time.sleep(5)

# 최근 열린 탭으로 전환
driver.switch_to.window(driver.window_handles[-1])
# 로딩 기다리기
time.sleep(5)

# 더보기 계속 클릭하기(스크롤 안내리고 더보기만 클릭해줌)
# while True:
#     try:
#         driver.find_element_by_css_selector('#div_more_review').click()
#         time.sleep(1)
#     except:
#         break

# 스크롤 내리면서 더보기 클릭하기
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    # (1) 4번의 스크롤링
    for i in range(4):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        # (2) 더보기 클릭
        driver.find_element_by_xpath('//*[@id="div_more_review"]').click()
        # (3) 종료 조건
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# 세팅
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

# 댓글 추출
contents = soup.find_all("p", {"class": "review_contents btxt"})
contents = [content.text for content in contents]

# 댓글 날짜 추출
dates = soup.find_all("i", {"class": "date"})
dates = [date.text for date in dates]

# 리뷰 점수 추출 (속성값가져오기)
attr = soup.select(' i.star > i ')
for a in attr:
    aa = int(a['style'][6:8])
    print(aa)