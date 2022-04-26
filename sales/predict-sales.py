import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm






st.image('./sales/images.jpg')
st.title('Prediction sales explorary data analysis')


st.sidebar.header('Info')
st.sidebar.write('we\'re looking at data of sales the goal is to predict monthly number of sold items')

st.sidebar.markdown("""Made by *Gabriel Pizzo* \

                    [Kaggle](https://www.kaggle.com/gabrieldu69)\
                    
                    [Linkedin](https://www.linkedin.com/in/gabriel-pizzo-486163128/)\
                    
                    *gabrielpdata@gmail.com*
                    """)


st.markdown(""" Explorary data analysis of [Kaggle](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales) dataset 
""")

@st.cache
def load_data(path):
    return pd.read_csv(path)

@st.cache
def load_data_time(path):
    return pd.read_csv(path,index_col='date',parse_dates=True)
    
 
sales_train=load_data_time('./sales/sales_train.csv')
items=load_data('./sales/items.csv')
items_category=load_data('./sales/item_categories.csv')
shops=load_data('./sales/shops.csv')

@st.cache
def merging(df1,df2,key):
    return pd.merge(df1,df2,on=key)
 
items_merged=merging(items,items_category,'item_category_id')
sales_train_merged=merging(sales_train,shops,'shop_id')
sales_train_merged=merging(sales_train_merged,items_merged,'item_id')


st.header('Display data')

if st.button('Show sales'):
    st.dataframe(sales_train.head(10))

if st.button('Show items'):
    st.dataframe(items.head(10))
    
   

@st.cache
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
    


st.header('Display biggest shops and category in sold items')


if st.button('Top category in item sold'):
    top10cat=plt.figure(figsize=(8,5))
    plt.title('top category')
    plt.ylabel('Sales')
    sales_train_merged.groupby('item_category_name')['item_cnt_day'].sum().sort_values(ascending=False)[0:10].plot(kind='bar',color='Orange',ls='dashed',edgecolor='Black')  
    st.pyplot(top10cat)
    
if st.button('Top shops in item sold'):
    top10s=plt.figure(figsize=(8,5))
    plt.title('top shops')
    plt.ylabel('Sales')
    sales_train_merged.groupby('shop_name')['item_cnt_day'].sum().sort_values(ascending=False)[0:10].plot(kind='bar',color='Red',ls='dashed',edgecolor='Black')
    st.pyplot(top10s)
    



#trend and seasonality decomposition
decomposition=sm.tsa.seasonal_decompose(sales_train['item_cnt_day'].resample('M').agg(['sum']),model='additive')
 
st.header('Display seasonal decomposition')

st.write("seasonal decomposition using statsmodels")

if st.button('trend'):
    figtrend=plt.figure(figsize=(6,4))
    plt.title("item counts trend")
    plt.plot(sales_train['item_cnt_day'].resample('M').sum(),c='blue', lw=1,ls='--')
    plt.plot(decomposition.trend.index, decomposition.trend, c='red',lw=1)
    plt.legend(["sum of item counts","trend of item count"])
    plt.xlabel('time')
    plt.ylabel('item sold')
    plt.xticks(size=4)
    plt.grid()
    st.pyplot(figtrend)
    st.write("there is a decreasing trend over time")
    
if st.button('seasonal component'):
    figseason=plt.figure(figsize=(15,7))
    plt.title("sales total values and trend")
    plt.plot(sales_train['item_cnt_day'].resample('M').sum(),c='blue')
    plt.plot(decomposition.seasonal.index, decomposition.seasonal, c='red')
    plt.legend(["sum of sales","seasonal sales"])
    plt.grid()
    st.pyplot(figseason)
    st.write("we can see there is a heavy seasonal component in our case")
    

