import pandas as pd

df = pd.read_csv('C_Data_month.csv')
data=df['AverageTemperature_1']
result=[]
sum=0
for i in range(len(data)):
    sum+=data[i]
    if((i+1)%3==0):
        result.append(sum/3)
        sum=0

test=pd.DataFrame(columns=['temperature'],data=result)
test.to_csv('C_Data_season.csv',encoding='gbk')