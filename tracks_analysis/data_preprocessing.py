import os
import sys
import requests
import json
from tqdm import tqdm
import logging
import logging.config
import yaml

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from Utils.utils import is_file_empty
from requests import exceptions as req_exceptions


# 设置 logger
current_file_name = os.path.basename(__file__)[:-3]
logging_config_file = './configs/log_configs.yaml'

with open(logging_config_file, 'r') as f:
    config = yaml.safe_load(f.read())
    config["handlers"]["file"]["filename"] = \
        config["handlers"]["file"]["filename"].format(name=current_file_name)
    logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

# 定义数据所在目录以及数据所在网址
data_dir = './tracks_data'
data_source_url = 'http://182.92.174.94:3000/' # 'http://182.92.174.94:3000/?value=heilongjiang' 

# 从手动下载的文件中获取各省市的名称的拼音
# file_list = os.listdir(data_dir)
# province_list = [province[:-5] for province in file_list]
# print(province_list)

province_list = ['anhui', 'beijing', 'chongqing', 'fujian', \
    'gansu', 'guangdong', 'guangxi', 'guizhou', 'hainan', 'hebei', \
    'heilongjiang', 'hunan', 'jiangsu', 'jiangxi', 'jilin', 'liaoning', \
    'neimenggu', 'ningxia', 'shangdong', 'shanghai', 'shanxi_jin', 'shanxi_shan', \
    'sichuan', 'taiwan', 'tianjin', 'xianggang', 'yunnan', 'zhejiang']

def update_data():
    '''从网址上爬取或更新患者轨迹数据
    '''
    for province in tqdm(province_list):
        data_url = data_source_url + '?value={}'.format(province)
        try:
            rs = requests.get(url=data_url)
            rs_json = rs.json()
        except req_exceptions.Timeout as e:
            logger.error('请求超时：' + str(e.message))
            continue
        except ValueError as ve:
            logger.error('解码 json 失败：' + str(ve.message))
            continue
        file_path = data_dir + '/{}.json'.format(province)
        with open(file_path, mode='w+', encoding='utf-8') as f:
            json.dump(rs_json, f, ensure_ascii=False)
    logger.info("数据更新成功！")
    


if __name__ == '__main__':
    # update_data()
    # 判断哪些 json 文件不包含数据
    for f_name in os.listdir(data_dir):
        file_path = data_dir + '/{}'.format(f_name)
        if is_file_empty(file_path):
            print(f_name)

    
