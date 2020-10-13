# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from bs4 import BeautifulSoup
import requests
import re
import time
from tqdm import tqdm
import json

class Web_Scraping:

    def __init__(self, navigator_url, city, url_pattern):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        self.driver = webdriver.Chrome(executable_path='./dependencies/chromedriver',
                                        chrome_options=chrome_options)
        self.url = navigator_url    # 导航页网址
        self.url_pattern = url_pattern
        #self.server_url = server_url
        self.city = city
        #if url: self.head = 'https' if 'https' == url[:5] else 'http' 

    def getLinks(self):
        """
        这里的链接选取规则应该视应用而定
        """
        try:
            elements = self.driver.find_elements_by_xpath(self.url_pattern)
        except NoSuchElementException as e:
            print('未找到符号要求的链接！')
            print(e)
        return [element.text for element in elements]

    def getReportLinks(self):
        all_links = []
        try:
            self.driver.get(self.url)
        except Exception as error:
            print("导航网页打开失败！")
            print(error)
            return None 
        # 获取符合 url_pattern 的报告链接
        links = self.getLinks()
        all_links += links
        #if bsObj == None: return None
        
        # 通过点击 页码 遍历所有的导航页，进而访问各个导航页中的所有符合要求的报告
        page_count = 1
        print("导航页：", str(page_count))
        while True:
            page_count += 1
            print("导航页：", str(page_count))
            try:
                # 访问下一导航页
                page = self.driver.find_element_by_link_text(str(page_count))
                self.driver.execute_script('arguments[0].click();', page)
                time.sleep(2)
            except Exception as error:
                print('无法遍历至下一导航页！')
                print(error)
                break
            # 获取该导航页中的报告链接
            links = self.getLinks()
            all_links += links
        # Close the driver
        self.driver.close() 
        return all_links
     
    
    def extractTextFromLinks(self):
        """
        从网页中抽取需要的信息，需要根据应用进行更改
        """
        links = self.getReportLinks()
        all_extracted = []
        patient_id = 0
        nums_patient = []
        for page_id, link in enumerate(tqdm(links)):
            extracted = {}  # {'city': str,'report_date': str, 'patient_list': []}
            patient_list = []   # [{'id': int, 'address': str, 'tracks':[]},...]
            try:
                html = requests.get(link)
                bsObj = BeautifulSoup(html.text, 'html.parser')
            except Exception as error:
                print("报告网页打开失败！")
                continue
            if not bsObj: continue 
            if page_id == 10:
                pass
                print("这个页面的轨迹未能抽取到！！")
            # 获取日期
            report_date = bsObj.find('li', string=re.compile("^发布日期")).text
            report_date = "-".join(re.findall(r"\d+", report_date)) # 2020-xx-xx
            # 获取患者信息
            # r"^.{0,6}\d{,2}月\d{1,2}日"
            ## 以地址信息为分割点
            tracks = bsObj.find_all('p', string=re.compile(r"[常居]住[地址]："))
            
            for i, track in enumerate(tracks):
                patient_track = {}  # {'tracks': [(track_date, track_record),...], 'address': str}
                track_records = []  # [(track_date, track_record),...]
                #colon_index = track.text.find("：") # 注意这里是中文格式下的冒号
                try:
                    address = re.search(r"：(.*?)[。，、]", track.text).group(1)  # 截取地址信息
                except AttributeError as e:
                    colon_index = track.text.find("：")
                    address = track.text[colon_index + 1:]
                    #address = re.search(r"：(.*?)", track.text).group(1)
                address = address.replace("哈市", "哈尔滨市")

                patient_id += 1
                #print(address)
                # 逐获取轨迹信息
                while True:
                    track = track.find_next_sibling('p')
                    if track == None: break  # 为空时跳转至下一个记录
                    text = track.text
                    if len(text) == 0: continue 
                    # 考虑到多个病例共同使用轨迹的情况，只把轨迹提取一次
                    if re.search(r"[常居]住[地址]：", text): break
                    # 若开头不包含日期信息，说明从这里开始不是轨迹信息
                    if not re.search(r"^.{0,6}\d{1,2}月\d{1,2}日", text): continue
                    if '3月28日,孙某、刘某、李某等3名无症状感染者乘俄航' in text: break
                    # 抽取日期
                    if text[0] == '系': continue
                    track_date = re.search(r".*?(\d{1,2})月(\d{1,2})日.*", text)
                    track_date = [track_date.group(i) for i in range(1,3)]
                    year = ["2020"] if int(track_date[0]) < 10 else ["2019"]
                    track_date = year + track_date
                    track_date = "-".join(track_date)
                    print(track_date, address)
                    # 轨迹
                    one_track = text

                    track_records.append((track_date, one_track))

                patient_track['id'] = patient_id - 1
                patient_track['address'] = address
                patient_track['tracks'] = track_records
                
                patient_list.append(patient_track)
                
            nums_patient.append(len(patient_list))
                    
            extracted['city'] = self.city
            extracted['report_date'] = report_date
            extracted['patient_list'] = patient_list

            all_extracted.append(extracted)

        return all_extracted, nums_patient          
    
    
if __name__ == '__main__':
    navigate_url = 'http://app3.harbin.gov.cn/hrbjrobot/search.do?webid=1&pg=12&p=1&tpl=&category=&q=%E8%BD%A8%E8%BF%B9&pq=%E5%93%88&oq=&eq=%E5%9C%B0%E5%9B%BE&doctype=&pos=title&od=2&date=&date='
    #server_url = 'http://app3.harbin.gov.cn'
    url_pattern = "//div[@class='jsearch-result-url']/a"
    scrap = Web_Scraping(navigate_url, "哈尔滨市", url_pattern)
  
    text, nums = scrap.extractTextFromLinks()
    
    with open('patient_tracks.json', 'w+', encoding='utf-8') as f:
        json.dump(text, f, ensure_ascii=False)
    print(text[0])
    
    print('finished! the numbers of patients is', nums)
    print("total number of patients is", sum(nums))


