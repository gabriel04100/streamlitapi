import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


pip install matplotlib


st.image('images.jpg')
st.title('Prediction sales explorary data analysis')


st.sidebar.header('Info')
st.sidebar.write('we\'re looking at data of sales the goal is tho predict monthly number of sold items')

st.sidebar.markdown("""Made by *Gabriel Pizzo* \

                    [Kaggle](https://www.kaggle.com/gabrieldu69)\
                    
                    [Linkedin](https://www.linkedin.com/in/gabriel-pizzo-486163128/)\
                    
                    *gabrielpdata@gmail.com*
                    """)


st.markdown(""" Explorary data analysis of [Kaggle](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales) dataset 
""")

sales_train=pd.read_csv('sales_train.csv',index_col='date',parse_dates=True)
test=pd.read_csv('test.csv')
sample_submission=pd.read_csv('sample_submission.csv')
items=pd.read_csv('items.csv')
items_category=pd.read_csv('item_categories.csv')
shops=pd.read_csv('shops.csv')
items_merged=pd.merge(items,items_category,on='item_category_id')
sales_train_merged=pd.merge(sales_train,shops,on='shop_id')
sales_train_merged=pd.merge(sales_train_merged,items_merged,on='item_id')


st.header('Display data')

if st.button('Show sales'):
    st.dataframe(sales_train)

if st.button('Show items'):
    st.dataframe(items)
    
   
from sklearn.ensemble import IsolationForest


def detect_outliers(dataframe,contamination):
    

    a=list(dataframe.select_dtypes(['int64']).columns)+list(dataframe.select_dtypes(['float64']).columns)
    model=IsolationForest(contamination=contamination)
    model.fit(dataframe[a])
    outliers = model.predict(dataframe[a]) ==-1
    
    return outliers
    


st.header('Data functions')
contamination=st.slider('contamination level', min_value=0.04, max_value=0.06)

if st.button('outliers cleaning'):
    index=detect_outliers(sales_train[['item_price','item_cnt_day']],contamination)
    lign,col=sales_train[index].shape
    sales_train=sales_train[index==False]

   
st.header('Display Feature repartition')

figure2=plt.figure(2,figsize=(15,10))

plt.subplot(3,1,1)
plt.title("item price",size=15)
sns.boxplot(data=sales_train,x="item_price")
plt.subplot(3,1,2)
plt.title("item count day",size=15)
sns.boxplot(data=sales_train,x="item_cnt_day")

figure2.tight_layout(pad=3.0)



if st.button('item price repartition'):
    st.pyplot(figure2)
    
  
#top 10 categories
top10cat=plt.figure(figsize=(8,5))
plt.title('top category')
plt.ylabel('Sales')
sales_train_merged.groupby('item_category_name')['item_cnt_day'].sum().sort_values(ascending=False)[0:10].plot(kind='bar',color='Orange',ls='dashed',edgecolor='Black')  


#top 10 shops
top10s=plt.figure(figsize=(8,5))
plt.title('top shops')
plt.ylabel('Sales')
sales_train_merged.groupby('shop_name')['item_cnt_day'].sum().sort_values(ascending=False)[0:10].plot(kind='bar',color='Red',ls='dashed',edgecolor='Black')



st.header('Display biggest shops and category in sold items')


if st.button('Top category in item sold'):
    st.pyplot(top10cat)
    
if st.button('Top shops in item sold'):
    st.pyplot(top10s)
    

#trend and seasonality decomposition
import statsmodels.api as sm

decomposition=sm.tsa.seasonal_decompose(sales_train['item_cnt_day'].resample('M').agg(['sum']),model='additive')

figtrend=plt.figure(figsize=(6,4))
plt.title("item counts trend")
plt.plot(sales_train['item_cnt_day'].resample('M').sum(),c='blue', lw=1,ls='--')
plt.plot(decomposition.trend.index, decomposition.trend, c='red',lw=1)
plt.legend(["sum of item counts","trend of item count"])
plt.xlabel('time')
plt.ylabel('item sold')
plt.xticks(size=3)

 
st.header('Display seasonal decomposition')

if st.button('trend'):
    st.pyplot(figtrend)
    

