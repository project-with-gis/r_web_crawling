import pandas as pd
import requests as req
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument('--headless')  # 내부 창을 안 띄우게 설정
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--disable-dev-shm-usage')

read = pd.read_csv('storeInfo_2.csv')
df = pd.DataFrame(read)

s_link_list=[]
s_link= list(df['s_link'])
for s in s_link:
    s_link_list.append(str(s))
store_id_list = []
store_id = list(df['store_id'])
for id in store_id:
    store_id_list.append(id)
# store_id = store_id.append(list(df['store_id']))


store_name = df['store_name'].to_list()
# print(store_name)
base_url = 'https://www.siksinhot.com/P/'

index_role = 1
# total_nullist=[] # 결측치 있는 부분 알려줌
# store_review_onelist=[] #리뷰 한 행이 될 리스트



# 최종 리뷰 csv가 될 df 생성
total_df = pd.DataFrame(index=range(0,1),columns=["store_id","portal_id", "date","score", "review"])

contents = []
stars = []
i = 0
for a in range(len(store_name)):
    url = base_url + s_link_list[i]
    # i += 1
    print(url)
    driver = webdriver.Chrome('chromedriver.exe', options=chrome_options)
    driver.get(url)
    body = driver.find_element_by_tag_name("body")

    while True:
        try:
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(1.5)
            driver.find_element_by_xpath("//*[@id='siksin_review']/div[3]/a/span").click()
        except:
            break

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")

    contents = body.select("#siksin_review > div.rList > ul > li > div > div.cnt > div.score_story > p")
    contents = [content.text for content in contents]

    stars = body.select("div.score_story > div > span > strong")
    stars = [star.text for star in stars]

    df1 = pd.DataFrame({"store_id": store_id_list[i] , "portal_id": 1004, "score": stars, "review": contents})
    total_df = pd.concat([total_df, df1])

    i += 1

    total_df.to_csv("test1.csv", encoding='utf-8')


    # html = driver.page_source
    # soup = BeautifulSoup(html, "html.parser")
    # body = soup.find("body")
    #
    # contents = body.select("#siksin_review > div.rList > ul > li > div > div.cnt > div.score_story > p")
    # contents = [content.text for content in contents]
    # print(len(contents))
    #
    # stars = body.select("div.score_story > div > span > strong")
    # stars = [star.text for star in stars]

#     df1 = pd.DataFrame({"store_id": a, "portal_id": 1004, "score": stars, "review": contents})
#     total_df = pd.concat([total_df, df1])
#
#     total_df.to_csv("test1.csv", encoding='utf-8')
#

# num = []
# reviews = []
# score = []
#
# for i in range(len(contents)):
#   num.append( i + 1 )
#   reviews.append( contents[i])
#   score.append( stars[i])
#
# info = {'num':num, 'store_name':store_name, 'review':reviews}
# df = pd.DataFrame(info)
# # df.set_index('num', inplace=True)
# print(df.head())
#
# df.to_csv('reviews.csv', encoding='utf-8')















# html = driver.page_source
# result = BeautifulSoup(html, 'html.parser')
# time.sleep(10)
# body = result.find("h3")
# store_name = body.select('div.store_name_score > h3 > react-text')

# store_name = driver.find_element_by_xpath('')
# time.sleep(10)
# print(store_name)
# for ptag in store_name:
#     print(ptag.stirng)
#
# response = req.get(url)
# #
# if response.status_code == 200:
#     html = response.text
#     soup = BeautifulSoup(html, 'html.parser')
#     print(soup)
# #
# else:
#     print(response.status_code)

# url = 'https://www.siksinhot.com/P/254650'
