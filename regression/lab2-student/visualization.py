# 导入Python的数据处理库pandas, 相当于python里的excel
import pandas as pd

#导入python高级数据可视化库seaborn
import seaborn as sns

# 导入python绘图matplotlib
import matplotlib.pyplot as plt

"""
各个字段的含义：
    CRIM     犯罪率
    ZN       住宅用地所占比例
    INDUS    城镇中非商业用地所占比例
    CHAS     是否处于查尔斯河边
    NOX      一氧化碳浓度
    RM       住宅房间数
    AGE      1940年以前建成的业主自住单位的占比
    DIS      距离波士顿5个商业中心的加权平均距离
    RAD      距离高速公路的便利指数
    TAX      不动产权税
    PTRATIO  学生/教师比例
    B        黑人比例
    LSTAT    低收入阶层占比
    MEDV     房价中位数
"""

# 读取数据集
df = pd.read_csv('./boston_house_price.csv', encoding='utf-8')
# 展示前五行
df.head()

#统计信息
df.describe()

#绘制房价的直方图
df['medv'].hist()

#绘制房价的箱线图
sns.boxplot(df['medv'])

#绘制房间数的散点图
plt.scatter(df['rm'],df['medv'])

def box_plot_outliers(df,s):
    q1,q3=df[s].quantile(0.25),df[s].quantile(0.75)
    iqr=q3-q1
    low,up=q1-1.5*iqr,q3+1.5*iqr
    df=df[(df[s]>up)|(df[s]<low)]
    return df

df_filter=box_plot_outliers(df,'rm')
df_filter.mean()

#输出距离和房间的散点图
plt.scatter(df['dis'],df['medv'])

#输出rad和房间的散点图
plt.scatter(df['rad'],df['medv'])

#输出B和房间的散点图
plt.scatter(df['b'],df['medv'])

#两个变量之间的相关系数
df.corr()

#设置绘图大小
plt.style.use({'figure.figsize':(15,10)})

#绘制直方图
df.hist(bins=15)

#绘制箱线图
sns.boxplot(data=df)

#绘制散点图
plt.figure(figsize=(12,12))
for i in range(13):
    plt.subplot(4,4,(i+1))
    plt.scatter(df.iloc[:,i],df['medv'])
    plt.title('{}-price scatter'.format(df.columns[i]))
    plt.xlabel(df.columns[i])
    plt.ylabel('boston house price')

plt.tight_layout()


