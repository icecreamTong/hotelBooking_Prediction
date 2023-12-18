#!usr/bin/env python
#-*- coding:utf8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#一、读取文件/数据概览
hotel_bookings=pd.read_csv('hotel_bookings.csv')
hotel_booking=hotel_bookings.copy()
print(hotel_bookings.info(),'\n')
print(hotel_bookings.describe(),'\n')
print(hotel_bookings.head(5),'\n')

#二、数据预处理
#1、为了方便对特征进行one-hot编码，先选出预测值并进行数据编码
y=hotel_booking.pop('assigned_room_type')

#数据可视化。预测值Assigned Room Type的比例示意图
import brewer2mpl
bmap=brewer2mpl.get_map('Paired','qualitative',12)
colors=bmap.mpl_colors
size1=y.value_counts()
labels=['A','D','E','F','G','C','B','H','I','K','P','L']
plt.figure(figsize=(8,8))
plt.title('Assigned Room Type')
plt.pie(size1,labels=labels,colors=colors,shadow=True,autopct='%.2f%%')
plt.axis=('equal')
plt.legend()
plt.show()

mapping={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'K':11,'L':12,'P':16}
y=y.map(mapping)

#2、缺失值处理，用上下条数据填充
hotel_booking['children'].fillna(method='backfill',inplace=True)
hotel_booking['country'].fillna(method='backfill',inplace=True)
hotel_booking['agent'].fillna(method='backfill',inplace=True)
hotel_booking = hotel_booking.drop(['company'],axis=1)#对于'company'列，由于缺失了94.3%，缺失量过多，故删除该特征

#特征工程
#1、数据编码（映射的方式）
hotel_booking_f=hotel_bookings.copy()
hotel_booking_f['children'].fillna(method='backfill',inplace=True)
hotel_booking_f['country'].fillna(method='backfill',inplace=True)
hotel_booking_f['agent'].fillna(method='backfill',inplace=True)
hotel_booking_f= hotel_bookings.drop(['company','reservation_status_date'],axis=1)

mapping1 = {'City Hotel': 0, 'Resort Hotel': 1}
hotel_booking_f['hotel'] = hotel_booking_f['hotel'].map(mapping1)
mapping2={'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
          'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
hotel_booking_f['arrival_date_month'] = hotel_booking_f['arrival_date_month'].map(mapping2)
mapping3={'BB':1,'HB':2,'SC':3,'FB':4,'Undefined':0}
hotel_booking_f['meal'] = hotel_booking_f['meal'].map(mapping3)
mapping4={'PRT':1,'GBR':2,'FRA':3,'ESP':4,'DEU':5}
hotel_booking_f['country'] = hotel_booking_f['country'].map(mapping4)
mapping5={'Online TA':1,'Offline TA/TO':2,'Groups':3,'Direct':4,'Corporate':5,'Complementary':6,'Aviation':7,'Undefined':8}
hotel_booking_f['market_segment'] = hotel_booking_f['market_segment'].map(mapping5)
mapping6={'TA/TO':1,'Direct':2,'Corporate':3,'GDS':4,'Undefined':5}
hotel_booking_f['distribution_channel'] = hotel_booking_f['distribution_channel'].map(mapping6)
mapping7={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'L':12,'P':16}
hotel_booking_f['reserved_room_type'] = hotel_booking_f['reserved_room_type'].map(mapping7)
mapping8={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'K':11,'L':12,'P':16}
hotel_booking_f['assigned_room_type'] = hotel_booking_f['assigned_room_type'].map(mapping8)
mapping9={'No Deposit':1,'Non Refund':2,'Refundable':3}
hotel_booking_f['deposit_type'] = hotel_booking_f['deposit_type'].map(mapping9)
mapping10={'Transient':1,'Transient-Party':2,'Contract':3,'Group':4}
hotel_booking_f['customer_type'] = hotel_booking_f['customer_type'].map(mapping10)
mapping11={'Check-Out':1,'Canceled':2,'No-Show':3}
hotel_booking_f['reservation_status'] = hotel_booking_f['reservation_status'].map(mapping11)

#热力图
import seaborn as sns
plt.figure(figsize=(25,25))
sns.set()
ax=sns.heatmap(hotel_booking_f.corr(), vmin=0.0001, vmax=None,cmap=None, center=None, robust=False, annot=True,
               fmt='.1g', annot_kws=None,linewidths=0, linecolor='white', cbar=True, cbar_kws=None,
               cbar_ax=None,square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None)
plt.show()

#3、数据编码（不同于特征工程，在此对分类标签进行one-hot编码）
mapping1 = {'City Hotel': 0, 'Resort Hotel': 1}
hotel_booking['hotel'] = hotel_booking['hotel'].map(mapping1)
mapping2={'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
          'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
hotel_booking['arrival_date_month'] = hotel_booking['arrival_date_month'].map(mapping2)
dummy1=pd.get_dummies(hotel_booking.pop('meal'),prefix='meal')
dummy2=pd.get_dummies(hotel_booking.pop('country'),prefix='country')
dummy3=pd.get_dummies(hotel_booking.pop('market_segment'),prefix='market_segment')
dummy4=pd.get_dummies(hotel_booking.pop('distribution_channel'),prefix='distribution_channel')
dummy5=pd.get_dummies(hotel_booking.pop('reserved_room_type'),prefix='reserved_room_type')
dummy7=pd.get_dummies(hotel_booking.pop('deposit_type'),prefix='deposit_type')
dummy8=pd.get_dummies(hotel_booking.pop('customer_type'),prefix='customer_type')
dummy9=pd.get_dummies(hotel_booking.pop('reservation_status'),prefix='reservation_status')

#把'reservation_status_date'中的字符串型的日期用pd.to_datetime(df.date)转化为日期格式，再用函数转化成年、月、日三列,数据类型为int
reservation_status_date=hotel_booking.pop('reservation_status_date')
reservation_status_date=pd.to_datetime(reservation_status_date)
def get_date(date):
    '''这里的输入date是一列年月日数据'''
    Y, M, D = [], [], []
    for i in range(len(date)):
        oneday=date[i]
        year=oneday.year
        month=oneday.month
        day=oneday.day

        Y.append(year)
        M.append(month)
        D.append(day)
    date=pd.DataFrame()
    date['year']=Y
    date['month']=M
    date['day']=D
    return date
rsd_in_3colums=get_date(reservation_status_date)

#重新合并处理好的特征到一张表中
hotel_booking=pd.concat([hotel_booking,dummy1,dummy2,dummy3,dummy4,dummy5,dummy7,dummy8,dummy9,rsd_in_3colums],axis=1,ignore_index=True)

#三、算法建模
#划分训练集和测试集，7：3
X=hotel_booking
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
#结果评估：用准确率和f1作为结果评价指标。
from sklearn.metrics import accuracy_score,f1_score

#1、逻辑回归
from sklearn.linear_model import LogisticRegression
#分别探究模型迭代次数（max_iter）和损失函数优化器（solver）对模型性能的影响。以下是探究的是max_iter：
def LR(solver):
    '''构建逻辑回归模型,返回准确率,f1,运行时间'''
    import time
    start = time.time()
    clf_lr = LogisticRegression(penalty='l2', solver=solver, random_state=0, max_iter=100)
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)

    accuracy=accuracy_score(y_test, y_pred_lr)
    f1=f1_score(y_test, y_pred_lr, labels=None, pos_label=1, average='weighted', sample_weight=None)
    end = time.time()
    return accuracy,f1,end-start

solver_set=['saga','liblinear','newton-cg','lbfgs']#损失函数优化器集合
accuracy_set=[]
f1_set=[]
time_set=[]
for i in solver_set:
    ac,f1,time=LR(i)
    accuracy_set.append(ac)
    f1_set.append(f1)
    time_set.append(time)

#可视化
#accuracy，f1随不同损失函数优化器变化的折线图图
plt.figure(figsize=(12, 12))
plt.grid(visible=True, ls=':')
plt.title('LR: solver Varying Accuracy&F1 Plot',fontsize=18)
plt.plot(solver_set,accuracy_set,label='Accuracy',c='#20B2AA',lw=2,ls='dashed',mfc='#008080',marker='o',ms=8)
plt.plot(solver_set,f1_set,label='F1',c='#483D8B',lw=2,ls='dashed',mfc='#4B0082',marker='s',ms=8)
plt.xlabel('Solver Set', fontsize=16)
plt.ylabel('Accuracy&F1', fontsize=16)
plt.legend()
plt.show()

#time的图
plt.figure(figsize=(10, 10))
plt.grid(visible=True, ls=':')
plt.title('LR: solver Varying Running Time Plot',fontsize=18)
plt.plot(solver_set,time_set,label='Time',c='slateblue',lw=2,ls='dashed',mfc='slateblue',marker='o',ms=8)
plt.xlabel('Min Samples Leaf', fontsize=16)
plt.ylabel('Running Time', fontsize=16)
plt.legend()
plt.show()

#2、决策树（可视化的代码同上，略）
#探究最大深度（max_depth）对决策树模型性能的影响
from sklearn.tree import DecisionTreeClassifier
def decision_tree(max_depth):
    '''构建决策树模型,返回准确率,f1,运行时间'''
    import time
    start = time.time()
    clf_DTC=DecisionTreeClassifier(criterion='gini',max_depth=max_depth)
    clf_DTC=clf_DTC.fit(X_train,y_train)
    y_pred_test=clf_DTC.predict(X_test)

    accuracy= accuracy_score(y_test, y_pred_test)
    f1=f1_score(y_test, y_pred_test, average='weighted')
    end = time.time()

    return accuracy,f1,end-start

max_depths=[]
accuracy_set=[]
f1_set=[]
train_results = []
test_results = []
time_set=[]
for i in range(1,101):
    max_depths.append(i)
    ac,f1=decision_tree(i)
    accuracy_set.append(ac)
    f1_set.append(f1)
    time_set.append(time)

#3、随机森林（可视化的代码同上，略）
#探究最大深度（max_depth）、分类器个数（n_estimators），以及“min_samples_leaf”& “min_samples_split” 对决策树模型性能的影响
from sklearn.ensemble import RandomForestClassifier
def Random_Forest_Classifier(max_depth):
    '''构建随机森林模型,返回准确率,f1,训练集的ROC-AUC值,测试集的ROC-AUC值'''
    import time
    start = time.time()
    clf_rf = RandomForestClassifier(n_estimators=80, max_depth=max_depth)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    accuracy= accuracy_score(y_test, y_pred_rf)
    f1=f1_score(y_test, y_pred_rf, average='weighted')
    end = time.time()
    return accuracy,f1,end-start

max_depth=[]
accuracy_set=[]
f1_set=[]
time_set=[]
train_results = []
test_results = []
for i in range(1,101,5):
    max_depth.append(i)
    ac,f1,time=Random_Forest_Classifier(i)
    accuracy_set.append(ac)
    f1_set.append(f1)
    time_set.append(time)
