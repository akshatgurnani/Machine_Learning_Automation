import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def main():
    st.title("Semi Automatic ML Model")
    dataset=st.sidebar.file_uploader("Upload file here",type=['csv','txt','xls'])
    data=pd.read_csv(dataset,delimiter=',')
    choice = st.sidebar.radio('what you want to do',("Data info","Data Cleaning","Data Visualization"))

    if choice=="Data info":
        st.write("Here is the complete information about the data")
        st.subheader(" **Data column names**")
        column_names=data.columns
        for i in column_names:
            st.write("*","*** ",i," ***")
        st.write("---")
        st.subheader(" **Data shape**")

        values=data.shape
        st.write("#### No of rows = ",values[0])
        st.write("#### No of columns = ", values[1])
        st.write("---")

        st.subheader(" **Data head**")
        st.table(data.head())
        st.write("---")
        st.subheader(" **Data info**")
        st.table(data.info())
        st.write("---")
        st.subheader(" **Data describe**")
        st.table(data.describe())
    elif choice=="Data Cleaning":
        st.write("***Data Before Clening***")
        st.table(data.head())
        st.subheader(" **Null values in the data**")
        null_val=data.isnull().sum()
        null_val=null_val.reset_index()
        null_val=null_val.rename(columns={'index':"Col_names"})
        st.table(null_val)
        col_names=data.columns
        lst=[]
        le=LabelEncoder()
        for i in col_names:
            if len(data[i].unique())<6:
                lst.append(i)
        st.write("**There are** ",len(lst)," **Categorical column in the dataset **")
        for i in lst:
            st.write("*",i)
        st.radio("Select Categorical encoding technique",('Label Encoding','One hot encoding'))
        st.write("Converting Categorical columns into numerical by using LabelEncoder")
        for i in col_names:
            if len(data[i].unique())<6:
                lst.append(i)
                data[i]=le.fit_transform(data[i])
        st.table(data.head())












if __name__=="__main__":
    main()