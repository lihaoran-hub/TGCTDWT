# encoding utf-8
'''
@Author: william
@Description:
@time:2020/8/18 19:14
'''
import pandas as pd
from datetime import datetime, timedelta
import calendar
from math import radians, cos, sin, asin, sqrt
import numpy as np


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000


if __name__ == '__main__':
    now_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    data = pd.read_csv(r'./OriginData/city_location_english.csv', header=None)

    W_matrix = []
    for i in range(len(data)):
        dist = []
        for j in range(len(data)):
            dist_temp = haversine(data[1][i], data[2][i], data[1][j], data[2][i])
            dist.append(int(dist_temp))
        W_matrix.append(dist)
    W_matrix = np.array(W_matrix)
    np.savetxt("./output/W_matrix_" + now_time + ".csv", W_matrix, delimiter=',')
    print('End')