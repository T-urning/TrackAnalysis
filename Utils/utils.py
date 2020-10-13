import requests
import os
from .coordTransform_utils import gcj02_to_wgs84

baiduApiKey = '9LHmCBG3xGGOu41Bo0ddLPivcgkAnh3M'


def is_file_empty(file_path):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0

def address2coords(address):
    """
    根据给定的地址文本 address ，返回该地址在 WGS84 坐标系的坐标值，并给出坐标解析时的可信度。
    """
    lat = None
    lon = None
    try:
        url = 'http://api.map.baidu.com/geocoding/v3'
        response = requests.get(url,
                                params={'address':address, 
                                        'output':'json',
                                        'ak':baiduApiKey,
                                       'ret_coordtype':'gcj02ll'}) 
        # If the response was successful, no Exceptiono will be raised
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        response = response.json() 
        lat = response['result']['location']['lat']
        lon = response['result']['location']['lng']
        confidence = response['result']['confidence']
        # 上面获取的坐标是 GCJ02 坐标系（火星坐标系）的，下面将坐标转至 OSM 使用的 WGS84
        lon, lat = gcj02_to_wgs84(lon, lat)
    return lat, lon, confidence       


