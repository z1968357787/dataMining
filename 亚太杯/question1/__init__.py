from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('C_Data_1.1.csv')
data=np.array(df['AverageTemperature_1'])
list=[]
for i in range(len(data)):
    x=i+1744
    list.append(x)
plt.plot(list, data, label="temperature", color="blue")  # 蓝线表示真实值
#plt.plot(range(len(y_pred_test)), result, label="pred_y", color="red")  # 红线表示预测值
plt.legend(loc='best')
plt.show()