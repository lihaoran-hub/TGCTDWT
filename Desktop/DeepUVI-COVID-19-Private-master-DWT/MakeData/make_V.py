# encoding utf-8
'''
@Author: william
@Description:  制作节点权重
@time:2020/6/15 19:21
'''

import pandas as pd
from datetime import datetime, timedelta
import calendar
from math import radians, cos, sin, asin, sqrt
import numpy as np


if __name__ == '__main__':
    now_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    data = pd.read_csv(r'./OriginData/city_ind_final_2020_8_19_eng.csv', header=None)

    data_temp = data[0:95]

    result_total = []

    for i in range(96):
        result_temp = []
        count = i
        for j in range(29):
            result_temp.append(data.values[count][6])
            count += 96
        result_total.append(result_temp)
    result_total = np.array(result_total)
    np.savetxt("./output/V_matrix_dead_" + now_time + ".csv", result_total, delimiter=',', fmt='%f')

    print('End')