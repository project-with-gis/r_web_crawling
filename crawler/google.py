from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup as bs
from urllib.request import urlretrieve
from urllib.parse import quote_plus
import pandas as pd

path = "C:/Users/aj878/PycharmProjects/limajin/"
store_info = pd.read_csv(path + "store_info.csv")



# 검색할 가게의 주소와 주소명
name_addr=store_info[['store_name','store_addr']]

store_name_addr = []
for i in range(len(name_addr)):
    store_name_addr.append(list(name_addr.iloc[i, :]))

# store_addr = []
# for addr in store_info['store_addr']:
#     store_addr.append(addr)


# info_count=len(store_info)
# index_role = 0
# total_nullist=[] # 결측치 있는 부분 알려줌
# store_review_onelist=[] #리뷰 한 행이 될 리스트

# 최종 리뷰 csv가 될 df 생성
total_df = pd.DataFrame(index=range(0,1),columns=["store_id","portal_id", "date","score", "review"])

##########################################################################################
# 자동 웹크롤러 만드는 것

# test 1.2

for s_name, s_addr in store_name_addr:
    # 주소와 가게이름 입력
    index_role += 1  # 스토어 아이디
    driver = webdriver.Chrome(r"C:/Users/aj878/Downloads/chromedriver_win32 (1)/chromedriver.exe")
    driver.get("https://www.google.com/maps/")
    searchbox = driver.find_element_by_css_selector("input#searchboxinput")
    plusUrl = s_name
    searchbox.send_keys(plusUrl)
    plusUrl = s_addr
    searchbox.send_keys(plusUrl)
    time.sleep(3)

    #   검색누르기
    searchbutton = driver.find_element_by_css_selector("button#searchbox-searchbutton")
    searchbutton.click()
    time.sleep(3)

    #   전체리뷰보기 누르기
    try:
        driver.find_element_by_css_selector(
            '#pane > div > div.widget-pane-content.cYB2Ge-oHo7ed > div > div > div.x3AX1-LfntMc-header-title > div.x3AX1-LfntMc-header-title-ma6Yeb-haAclf > div.x3AX1-LfntMc-header-title-ij8cu > div.x3AX1-LfntMc-header-title-ij8cu-haAclf > div > div.gm2-body-2.h0ySl-wcwwM-RWgCYc > span:nth-child(3) > span > span:nth-child(1) > span.OAO0-ZEhYpd-vJ7A6b.OAO0-ZEhYpd-vJ7A6b-qnnXGd > span:nth-child(1) > button:nth-child(1)').click()
        time.sleep(3)
    except:
        print("대표장소 아님", s_name, s_addr)
        continue

    #     # 총리뷰갯수 ver2
    s = driver.find_element_by_css_selector(
        '#pane > div > div.widget-pane-content.cYB2Ge-oHo7ed > div > div > div.siAUzd-neVct.section-scrollbox.cYB2Ge-oHo7ed.cYB2Ge-ti6hGc > div.PPCwl > div > div.jANrlb > div.gm2-caption').text.split(
        " ")[1]
    s1 = s.replace(",", "")
    s1 = s1.replace("개", "")
    total_review_num = int(s1[:])
    total_review_num

    #     # 걍 쭉내리기(자세히보기 안되고 빨리 내려감 )
    #     scrollable_div = driver.find_element_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]')
    #     for i in range(0,(round(total_review_num/10)+1)):
    #         try:
    #             driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight',
    #                     scrollable_div)
    #             time.sleep(2)
    #         except:
    #             None

    # 스크롤 하면서 긴 리뷰 더보기 누르기

    num_of_page_downs = 20
    scrollable_div = driver.find_element_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]')
    #     while True:
    #     while num_of_page_downs:
    for i in range(0, (round(total_review_num / 10) + 1)):
        scrollable_div.send_keys(Keys.PAGE_DOWN)
        time.sleep(1.5)
        try:
            driver.find_element_by_css_selector('jsl > button').send_keys(Keys.ENTER)
        except:
            print('<< 더보기 버튼 없음>>')
        num_of_page_downs -= 1

    #   댓글 추출
    html = driver.page_source
    soup = bs(html, "html.parser")

    #   리뷰내용 추출
    contents = soup.find_all("span", {"class": "ODSEW-ShBeI-text"})
    contents = [content.text for content in contents]

    # 댓글 날짜 추출
    dates = soup.find_all("span", {"class": "ODSEW-ShBeI-RgZmSc-date"})
    dates = [date.text for date in dates]
    dates

    # 리뷰 점수 추출
    stars = soup.find_all("span", {"class": "ODSEW-ShBeI-H1e3jb"})
    stars = [int(star['aria-label'][4]) for star in stars]
    stars

    df1 = pd.DataFrame({"store_id": index_role, "portal_id": 1002, "date": dates, "score": stars, "review": contents})
    total_df = pd.concat([total_df, df1])

    total_df.to_csv("test1.csv", encoding='utf-8-sig')

