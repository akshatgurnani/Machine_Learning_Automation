from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet,PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score, ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    image = Image.open(
        "business-manufacturing-process-automation-smart-industry-innovation-modern-technology-concept-143567093.jpg")
    st.set_page_config(
        page_title="Automated-ML-Model",
        page_icon=image,
        layout="wide",
    )

    st.markdown("<h1 style='text-align: center;font-family:georgia; color:#C3447A;'>Semi Automation Model</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center;font-family:georgia; color:#000000;'>You're either the one that creats the automation or you're getting automated</h3>",
        unsafe_allow_html=True)

    choice = st.sidebar.radio('what you want to do',("Data info","Data Cleaning, Visualization and Machine Learning","About"))
    st.sidebar.write("** Please upload valid data set **")
    dataset = st.sidebar.file_uploader("Upload file here", type=['csv', 'txt', 'xls'])




    if choice=="Data info":

        if dataset is not None:
            data = pd.read_csv(dataset, delimiter=',')
            st.image(image)
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
            st.write(data.info())
            st.write("---")
            st.subheader(" **Data describe**")
            st.table(data.describe())
    elif choice=="Data Cleaning, Visualization and Machine Learning":
        if dataset is not None:
            data = pd.read_csv(dataset, delimiter=',')
            st.markdown("<h3 style='text-align: center;font-family:georgia;font-size:32px;color: #1e90ff;'>Data Cleaning</h3>", unsafe_allow_html=True)
            st.write("**Data Before Clening**")
            st.table(data.head())
            st.write("**Null values in the data**")
            null_val=data.isnull().sum()
            null_val=null_val.reset_index()
            null_val=null_val.rename(columns={'index':"Col_names",0:"Sum_of_Null_values"})
            st.table(null_val)
            plt.figure(figsize=(5,5))
            st.write(sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='plasma'))
            st.pyplot()
            count = 0
            var=0
            total_col=0
            min_val=int((data.shape[0])/2)
            cleaned_data=data.copy()
            for i in data.columns:
                if null_val[null_val['Col_names'] == i]['Sum_of_Null_values'][count] > min_val:
                    cleaned_data=data.drop(labels=[i], axis=1)
                    var=var+1
                else:
                    if total_col==len(data.columns) and var==0:
                        cleaned_data = data.copy()
                total_col=total_col+1
                count = count+1
            # st.table(cleaned_data.head())
            if var==data.shape[1]:
                st.write("Can't proceed furture as it's all columns are containing more than 50% null values")
            else:
                if var > 0:
                    st.write("After Droping the columns which has more than 50% **NULL** values")
                    st.table(cleaned_data.head())
                else:
                    st.write("Since NULL Values are Less than 50% We are not dropping any column")
                st.write("** Want to see Unique values of a column: **")
                yes=st.selectbox("",cleaned_data.columns)
                if yes is not None:
                    st.write(cleaned_data[yes].unique())
                st.write("**Select the column names which are of no use**")
                Col_names_which_has_to_be_dropped = st.multiselect(" ", data.columns)
                waste_columns = []
                for i in range(0, len(Col_names_which_has_to_be_dropped)):
                    drop_name = Col_names_which_has_to_be_dropped[i]
                    waste_columns.append(drop_name)
                cleaned_data = cleaned_data.drop(labels=waste_columns, axis=1)
                st.table(cleaned_data.head())
                cleaned_data_for_visualization=pd.DataFrame()
                cleaned_data_for_visualization=cleaned_data
                columns_with_null_values = []
                count=0
                for i in null_val['Col_names']:
                    if null_val[null_val['Col_names'] == i]['Sum_of_Null_values'][count] > 0 and null_val[null_val['Col_names'] == i]['Sum_of_Null_values'][count] < min_val :
                        columns_with_null_values.append(i)
                    count = count + 1

                if len(columns_with_null_values)!=0:
                    st.write("**Do you want to drop all null values?**")
                    choice = st.radio('', ("Yes", "No"))
                    j = 0
                    if choice=="No":
                        a=0
                        b=600
                        for i in range(0,len(columns_with_null_values)):
                            st.write(" **Select the name of null values containing column name :** ")
                            null_colum_name=st.selectbox("",columns_with_null_values,key=a)
                            a=a+1
                            if len(cleaned_data[null_colum_name].unique()) > 10 :
                                st.subheader(" ** How you want to handle NUll values for that particular Column: **")
                                method_of_treating_null_value = st.selectbox("",['Mean of Values','Median of Values', 'Most Occuring element','Maximum val in column', 'Minimum val in column'],key=b)
                                b=b+1
                                if method_of_treating_null_value=='Mean of Values':
                                    cleaned_data[null_colum_name]=cleaned_data[null_colum_name].replace(np.NAN,cleaned_data[null_colum_name].mean())
                                elif method_of_treating_null_value=='Median of Values':
                                    cleaned_data[null_colum_name].fillna(cleaned_data[null_colum_name].median(), inplace=True)
                                elif method_of_treating_null_value=='Most Occuring element':
                                    cleaned_data[null_colum_name].fillna(cleaned_data[null_colum_name].mode()[0], inplace=True)
                                elif method_of_treating_null_value=='Maximum val in column':
                                    cleaned_data[null_colum_name].fillna(cleaned_data[null_colum_name].max(), inplace=True)
                                elif method_of_treating_null_value == 'Minimum val in column':
                                    cleaned_data[null_colum_name].fillna(cleaned_data[null_colum_name].min(), inplace=True)
                            else:
                                st.subheader(" ** As this column is Categorical We can only replace with most occuring element: **")
                                cleaned_data[null_colum_name].fillna(cleaned_data[null_colum_name].mode()[0], inplace=True)
                    else:
                        cleaned_data.dropna(inplace=True)
                    null_val = cleaned_data.isnull().sum()
                    null_val = null_val.reset_index()
                    null_val = null_val.rename(columns={'index': "Col_names", 0: "Sum_of_Null_values"})
                    st.table(null_val)
                    values =cleaned_data.shape
                    st.write("#### No of rows = ", values[0])
                    st.write("#### No of columns = ", values[1])
                lst = []
                names = cleaned_data.columns
                st.write("**Select target column name for Applying ML model**")
                target = st.selectbox("", names,key="target")
                st.write("Target:", "**", target, "**")
                col_names = []
                for i in cleaned_data.columns:
                    if i != target:
                        col_names.append(i)
                for i in col_names:
                    if len(data[i].unique()) < ((data.shape[0])/100):
                        lst.append(i)
                st.write("**There are** ", len(lst), " **Categorical column in the dataset **")
                for i in lst:
                    st.write("*", i)
                st.write("**Select encoding technique:**")
                encoding_choice = st.radio("",('Label Encoding', 'One hot encoding', "Not Required"))
                st.write("Converting Categorical columns into numerical by using",encoding_choice)
                cat_clm_names = lst
                if len(lst)>0:
                    if encoding_choice == 'Label Encoding':
                        d = defaultdict(LabelEncoder)
                        cleaned_data[cat_clm_names] = cleaned_data[cat_clm_names].apply(lambda x: d[x.name].fit_transform(x))
                    elif encoding_choice == 'One hot encoding':
                        cleaned_data = pd.get_dummies(cleaned_data, columns=cat_clm_names)
                    elif encoding_choice == 'Not Required':
                        pass
                else:
                    st.write("** As there are no categorical columns in the feature columns Please select Not required as a option in encoding technique**")
                st.table(cleaned_data.head())
                st.write("**Data for Visualization:**")
                st.table(cleaned_data_for_visualization.head())
                if st.checkbox("Correlation"):
                    st.write(sns.heatmap(cleaned_data_for_visualization.corr(),annot=True))
                    st.pyplot()
                    st.write(sns.pairplot(cleaned_data_for_visualization))
                    st.pyplot()
                if st.checkbox("Bar grapgh"):
                    x_axis = st.selectbox("Select x axis:", cleaned_data_for_visualization.columns)
                    x_axis = cleaned_data_for_visualization[x_axis]
                    y_axis = st.selectbox("Select y axis:", cleaned_data_for_visualization.columns)
                    y_axis = cleaned_data_for_visualization[y_axis]
                    st.write(sns.barplot(x_axis, y_axis))
                    st.pyplot()
                    plt.xticks(rotation=90)
                    plt.legend()
                    plt.grid()
                if st.checkbox("COUNT PLOT"):
                    c = st.selectbox("Select  axis:", cleaned_data_for_visualization.columns)
                    c_main = cleaned_data_for_visualization[c]
                    st.write(sns.countplot(c_main))
                    st.pyplot()
                    plt.grid()
                    plt.xticks(rotation=90)
                    plt.legend()

                if st.checkbox("PIE CHART"):
                    col = st.selectbox("Select 1 column", cleaned_data_for_visualization.columns)
                    pie = cleaned_data_for_visualization[col].value_counts().plot.pie(autopct="%1.1f%%")
                    st.write(pie)
                    st.pyplot()
                st.write("-----")
                st.markdown(
                    "<h3 style='text-align: center;font-family:georgia;font-size:32px;color:#8000ff;'>Machine Learning-Model</h3>",
                    unsafe_allow_html=True)
                st.write("** Do you want to Apply Machine Learning**")
                tell=st.radio("",("Yes","No"),key=25)

                ml_type=""
                if len(cleaned_data[target].unique())<10:
                    ml_type="Classification"
                else:
                    ml_type="Regression"
                if tell=="Yes":
                    st.write("** After doing Train-Test-Split we have **")
                    X = cleaned_data.drop(labels=target, axis=1)
                    y = cleaned_data[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
                    values1 = X_train.shape
                    values2 = X_test.shape
                    values3 = y_train.shape
                    values4 = y_test.shape

                    st.write(" Shape of X_train = ", values1)
                    st.write(" Shape of X_test = ", values2)
                    st.write(" Shape of y_train = ", values3)
                    st.write(" Shape of y_test = ", values4)
                    st.write("")

                    st.write("** Do you want to apply Scaling to the data: **")
                    ss = st.radio("", ("Yes", "No"),key="SS")
                    if ss == "Yes":
                        st.write("** Select which standard scaling you want to use: **")
                        ss_tech = st.radio("", ("Standard Scalar", "Min Max Scalar"),key="SS_tech")
                        if ss_tech == "Standard Scalar":
                            meth = StandardScaler()
                            X_train=meth.fit_transform(X_train)
                            X_test=meth.transform(X_test)
                            st.write("** Data After Standard Scaling**")
                            st.write(X_train[0:5])

                            st.write("* According to the target Column which you have selected earlier we found that it's a ","**",ml_type,"**"," problem ")
                            if ml_type=="Classification":
                                if len(cleaned_data[target].unique())==2:
                                    st.write("**Select the classification Algorithm:**")
                                    ml_class_algorithm=st.selectbox("",("Logistic Regression","k-Nearest Neighbors","Decision Trees","Support Vector Machine","Naive Bayes","Random Forest Classifier"))
                                else:
                                    st.write("**Select the classification Algorithm:**")
                                    ml_class_algorithm=st.selectbox("", ("Naive Bayes", "k-Nearest Neighbors", "Decision Trees", "Support Vector Machine","Random Forest Classifier"))
                                model=''
                                algorithm=''
                                if ml_class_algorithm=="Logistic Regression":
                                    model=LogisticRegression()
                                    algorithm="Logistic Regression"
                                    model.fit(X_train,y_train)
                                    predictions=model.predict(X_test)
                                if ml_class_algorithm=="k-Nearest Neighbors":
                                    algorithm = "KNN"
                                    model=KNeighborsClassifier()
                                    model.fit(X_train,y_train)
                                    predictions=model.predict(X_test)
                                if ml_class_algorithm=="Decision Trees":
                                    algorithm = "DTC"
                                    model=DecisionTreeClassifier()
                                    model.fit(X_train,y_train)
                                    predictions=model.predict(X_test)
                                if ml_class_algorithm=="Support Vector Machine":
                                    algorithm = "SVM"
                                    model=svm.SVC(kernel='linear')
                                    model.fit(X_train,y_train)
                                    predictions=model.predict(X_test)
                                if ml_class_algorithm=="Naive Bayes":
                                    algorithm = "Naive Bayes"
                                    model=GaussianNB()
                                    model.fit(X_train,y_train)
                                    predictions=model.predict(X_test)
                                if ml_class_algorithm=="Random Forest Classifier":
                                    algorithm = "RFC"
                                    model=RandomForestClassifier()
                                    model.fit(X_train,y_train)
                                    predictions=model.predict(X_test)
                                st.write("**",algorithm,"**"," algoritm score is","**",model.score(X_test,y_test),"**")
                                st.write("")
                                st.markdown(
                                    "<h3 style='text-align: center;font-family:georgia;font-size:32px;color:#000000;'>Confusion Matrix</h3>",
                                    unsafe_allow_html=True)
                                cm = confusion_matrix(y_test, predictions, labels=model.classes_)
                                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                                disp.plot()
                                st.pyplot()
                            elif ml_type=="Regression":
                                st.write("** After doing Train-Test-Split we have **")
                                X = cleaned_data.drop(labels=target, axis=1)
                                y = cleaned_data[target]
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                                                    random_state=42)
                                X_train_for, X_test_for, y_train_for, y_test_for = train_test_split(X.to_numpy(),
                                                                                                    y.to_numpy(),
                                                                                                    test_size=0.33,
                                                                                                    random_state=42)
                                values1 = X_train.shape
                                values2 = X_test.shape
                                values3 = y_train.shape
                                values4 = y_test.shape

                                st.write(" Shape of X_train = ", values1)
                                st.write(" Shape of X_test = ", values2)
                                st.write(" Shape of y_train = ", values3)
                                st.write(" Shape of y_test = ", values4)
                                st.write("")
                                st.write(
                                    "* According to the target Column which you have selected earlier we found that it's a ",
                                    "**", ml_type, "**", " problem")
                                st.write("**Select the classification Algorithm:**")
                                ml_class_algorithm = st.selectbox("", (
                                    "Linear Regression", "Support Vector Regression", "Random Forest Regressor",
                                    "Decision Tree Regressor"))
                                model = ''
                                algorithm = ''
                                predictions = 0
                                if ml_class_algorithm == "Linear Regression":
                                    model = LinearRegression()
                                    algorithm = "Linear Regression"
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)

                                if ml_class_algorithm == "Support Vector Regression":
                                    algorithm = "SVR"
                                    model = svm.SVR()
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)

                                if ml_class_algorithm == "Decision Tree Regressor":
                                    algorithm = "DTR"
                                    model = DecisionTreeRegressor()
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)

                                if ml_class_algorithm == "Random Forest Regressor":
                                    algorithm = "RFR"
                                    model = RandomForestRegressor()
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)

                                st.write("**", algorithm, "**", " algoritm Root mean squared error is", "**",
                                         np.sqrt(mean_squared_error(y_test, predictions)), "**")
                                st.write("**", algorithm, "**", " algoritm R2 score", "**",
                                         r2_score(y_test, predictions), "**")
                                plt.scatter(predictions, y_test, color="violet")
                                plt.title("Predictions  vs True Values ")
                                plt.show()
                                st.pyplot()
                        elif ss_tech=="Min Max Scalar":
                            meth = MinMaxScaler()
                            X_train = meth.fit_transform(X_train)
                            X_test = meth.transform(X_test)
                            st.write("** Data After Min Max Scaling **")
                            st.write(X_train[0:5])

                            st.write(
                                "* According to the target Column which you have selected earlier we found that it's a ",
                                "**", ml_type, "**", " problem ")
                            if ml_type == "Classification":
                                if len(cleaned_data[target].unique()) == 2:
                                    st.write("**Select the classification Algorithm:**")
                                    ml_class_algorithm = st.selectbox("", (
                                    "Logistic Regression", "k-Nearest Neighbors", "Decision Trees",
                                    "Support Vector Machine", "Naive Bayes", "Random Forest Classifier"))
                                else:
                                    st.write("**Select the classification Algorithm:**")
                                    ml_class_algorithm = st.selectbox("", (
                                    "Naive Bayes", "k-Nearest Neighbors", "Decision Trees", "Support Vector Machine",
                                    "Random Forest Classifier"))
                                model = ''
                                algorithm = ''
                                if ml_class_algorithm == "Logistic Regression":
                                    model = LogisticRegression()
                                    algorithm = "Logistic Regression"
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)
                                if ml_class_algorithm == "k-Nearest Neighbors":
                                    algorithm = "KNN"
                                    model = KNeighborsClassifier()
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)
                                if ml_class_algorithm == "Decision Trees":
                                    algorithm = "DTC"
                                    model = DecisionTreeClassifier()
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)
                                if ml_class_algorithm == "Support Vector Machine":
                                    algorithm = "SVM"
                                    model = svm.SVC(kernel='linear')
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)
                                if ml_class_algorithm == "Naive Bayes":
                                    algorithm = "Naive Bayes"
                                    model = GaussianNB()
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)
                                if ml_class_algorithm == "Random Forest Classifier":
                                    algorithm = "RFC"
                                    model = RandomForestClassifier()
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)
                                st.write("**", algorithm, "**", " algoritm score is", "**", model.score(X_test, y_test),
                                         "**")
                                st.write("")
                                st.markdown(
                                    "<h3 style='text-align: center;font-family:georgia;font-size:32px;color:#000000;'>Confusion Matrix</h3>",
                                    unsafe_allow_html=True)
                                cm = confusion_matrix(y_test, predictions, labels=model.classes_)
                                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                                disp.plot()
                                st.pyplot()
                            elif ml_type=="Regression":
                                st.write("** After doing Train-Test-Split we have **")
                                X = cleaned_data.drop(labels=target, axis=1)
                                y = cleaned_data[target]
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                                                    random_state=42)
                                X_train_for, X_test_for, y_train_for, y_test_for = train_test_split(X.to_numpy(),
                                                                                                    y.to_numpy(),
                                                                                                    test_size=0.33,
                                                                                                    random_state=42)
                                values1 = X_train.shape
                                values2 = X_test.shape
                                values3 = y_train.shape
                                values4 = y_test.shape

                                st.write(" Shape of X_train = ", values1)
                                st.write(" Shape of X_test = ", values2)
                                st.write(" Shape of y_train = ", values3)
                                st.write(" Shape of y_test = ", values4)
                                st.write("")
                                st.write(
                                    "* According to the target Column which you have selected earlier we found that it's a ",
                                    "**", ml_type, "**", " problem")
                                st.write("**Select the classification Algorithm:**")
                                ml_class_algorithm = st.selectbox("", (
                                    "Linear Regression", "Support Vector Regression", "Random Forest Regressor",
                                    "Decision Tree Regressor"))
                                model = ''
                                algorithm = ''
                                predictions = 0
                                if ml_class_algorithm == "Linear Regression":
                                    model = LinearRegression()
                                    algorithm = "Linear Regression"
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)

                                if ml_class_algorithm == "Support Vector Regression":
                                    algorithm = "SVR"
                                    model = svm.SVR()
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)

                                if ml_class_algorithm == "Decision Tree Regressor":
                                    algorithm = "DTR"
                                    model = DecisionTreeRegressor()
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)

                                if ml_class_algorithm == "Random Forest Regressor":
                                    algorithm = "RFR"
                                    model = RandomForestRegressor()
                                    model.fit(X_train, y_train)
                                    predictions = model.predict(X_test)

                                st.write("**", algorithm, "**", " algoritm Root mean squared error is", "**",
                                         np.sqrt(mean_squared_error(y_test, predictions)), "**")
                                st.write("**", algorithm, "**", " algoritm R2 score", "**",
                                         r2_score(y_test, predictions), "**")
                                plt.scatter(predictions, y_test, color="blue")
                                plt.title("Predictions  vs True Values ")
                                plt.show()
                                st.pyplot()
                    elif ss=="No":
                        st.write(
                            "* According to the target Column which you have selected earlier we found that it's a ",
                            "**", ml_type, "**", " problem ")
                        if ml_type == "Classification":
                            if len(cleaned_data[target].unique()) == 2:
                                st.write("**Select the classification Algorithm:**")
                                ml_class_algorithm = st.selectbox("", (
                                    "Logistic Regression", "k-Nearest Neighbors", "Decision Trees",
                                    "Support Vector Machine", "Naive Bayes", "Random Forest Classifier"))
                            else:
                                st.write("**Select the classification Algorithm:**")
                                ml_class_algorithm = st.selectbox("", (
                                    "Naive Bayes", "k-Nearest Neighbors", "Decision Trees", "Support Vector Machine",
                                    "Random Forest Classifier"))
                            model = ''
                            algorithm = ''
                            if ml_class_algorithm == "Logistic Regression":
                                model = LogisticRegression()
                                algorithm = "Logistic Regression"
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                            if ml_class_algorithm == "k-Nearest Neighbors":
                                algorithm = "KNN"
                                model = KNeighborsClassifier()
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                            if ml_class_algorithm == "Decision Trees":
                                algorithm = "DTC"
                                model = DecisionTreeClassifier()
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                            if ml_class_algorithm == "Support Vector Machine":
                                algorithm = "SVM"
                                model = svm.SVC(kernel='linear')
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                            if ml_class_algorithm == "Naive Bayes":
                                algorithm = "Naive Bayes"
                                model = GaussianNB()
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                            if ml_class_algorithm == "Random Forest Classifier":
                                algorithm = "RFC"
                                model = RandomForestClassifier()
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                            st.write("**", algorithm, "**", " algoritm score is", "**", model.score(X_test, y_test),
                                     "**")
                            st.write("")
                            st.markdown(
                                "<h3 style='text-align: center;font-family:georgia;font-size:32px;color:#000000;'>Confusion Matrix</h3>",
                                unsafe_allow_html=True)
                            cm = confusion_matrix(y_test, predictions, labels=model.classes_)
                            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                            disp.plot()
                            st.pyplot()
                        elif ml_type=="Regression":
                            st.write("** After doing Train-Test-Split we have **")
                            X = cleaned_data.drop(labels=target, axis=1)
                            y = cleaned_data[target]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                                                random_state=42)
                            X_train_for, X_test_for, y_train_for, y_test_for = train_test_split(X.to_numpy(),
                                                                                                y.to_numpy(),
                                                                                                test_size=0.33,
                                                                                                random_state=42)
                            values1 = X_train.shape
                            values2 = X_test.shape
                            values3 = y_train.shape
                            values4 = y_test.shape

                            st.write(" Shape of X_train = ", values1)
                            st.write(" Shape of X_test = ", values2)
                            st.write(" Shape of y_train = ", values3)
                            st.write(" Shape of y_test = ", values4)
                            st.write("")
                            st.write(
                                "* According to the target Column which you have selected earlier we found that it's a ",
                                "**", ml_type, "**", " problem")
                            st.write("**Select the classification Algorithm:**")
                            ml_class_algorithm = st.selectbox("", (
                                "Linear Regression", "Support Vector Regression", "Random Forest Regressor",
                                "Decision Tree Regressor"))
                            model = ''
                            algorithm = ''
                            predictions = 0
                            if ml_class_algorithm == "Linear Regression":
                                model = LinearRegression()
                                algorithm = "Linear Regression"
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)

                            if ml_class_algorithm == "Support Vector Regression":
                                algorithm = "SVR"
                                model = svm.SVR()
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)

                            if ml_class_algorithm == "Decision Tree Regressor":
                                algorithm = "DTR"
                                model = DecisionTreeRegressor()
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)

                            if ml_class_algorithm == "Random Forest Regressor":
                                algorithm = "RFR"
                                model = RandomForestRegressor()
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)

                            st.write("**", algorithm, "**", " algoritm Root mean squared error is", "**",
                                     np.sqrt(mean_squared_error(y_test, predictions)), "**")
                            st.write("**", algorithm, "**", " algoritm R2 score", "**",
                                     r2_score(y_test, predictions), "**")
                            plt.scatter(predictions, y_test, color="yellow")
                            plt.title("Predictions  vs True Values ")
                            plt.show()
                            st.pyplot()
                            # st.write("Plot Predicted values ** v/s ** Actual values")
                            # st.write(sns.lmplot(x=predictions,y=y_test,data=cleaned_data,palette='red'))
                            # st.pyplot()
    elif choice=="About":
            st.subheader("--About Me--")
            st.write(''' ''')
            st.write(''' ***Built by Shubham Chitaguppe*** ''')
            st.write(''' ***Insta*** : https://www.instagram.com/shubham_s_c/''')
            st.write(''' ***Linkedin*** : https://www.linkedin.com/in/shubham-chitaguppe-2449821a9/''')
            st.write(''' ***Github*** : https://github.com/SHUBHAM-max449''')
            st.markdown(
                "<h3 style='text-align: center;font-family:sans-serif;font-size:60px;color:#000000;'>Thank you</h3>",
                unsafe_allow_html=True)


if __name__=="__main__":
    main()