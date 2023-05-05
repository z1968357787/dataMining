from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('C_Data_season_result.csv')
data=df['temperature模型_1']
result=[]
sum=0
for i in range(len(data)):
    sum+=data[i]
    if((i+1)%4==0):
        result.append(sum/4)
        sum=0

list=[]
for i in range(len(result)):
    x=i+2014
    list.append(x)
plt.plot(list, result, label="temperature", color="blue")  # 蓝线表示真实值
#plt.plot(range(len(y_pred_test)), result, label="pred_y", color="red")  # 红线表示预测值
plt.legend(loc='best')
plt.show()