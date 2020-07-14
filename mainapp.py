# Core Pkgs
import streamlit as st 
# EDA Pkgs
import pandas as pd 
import numpy as np 
# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 
#sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    """Automated ML App"""

    activities = ["EDA","Plots","ML_Algorithms"]
    choice = st.sidebar.selectbox("Select Activities",activities)

    data = st.file_uploader("Upload a Dataset", type=["csv","txt","xlsx"])

    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show Shape"):
                st.write(df.shape)

            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.checkbox("Summary"):
                st.write(df.describe())

            #Show Selected Columns to be done

            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts())

            if st.checkbox("Correlation Plot -- matplotlib"):
                plt.matshow(df.corr())
                st.pyplot()
            
            if st.checkbox("Correlation Plot -- seaborn"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()

            if st.checkbox("Pie Plot"):
                all_columns = df.columns.to_list()
                column_to_plot = st.selectbox("Select 1 Column", all_columns)
                pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()
    
    elif choice == 'Plots':
        st.subheader("Data Visualization")

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

        if st.checkbox("Show Value Counts"):
            st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
            st.pyplot()

        #Customized Plot

    elif choice == 'ML_Algorithms':
        st.subheader("ML Algorithms")

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
        
        if st.checkbox("Summary"):
                st.write(df.describe())
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        col_name = st.selectbox("Select Column Name",["X","y"])

        if col_name == 'X':
            st.write(X)
        elif col_name == 'y':
            st.write(y)
        
        params = dict()
        classifer_name = st.sidebar.selectbox("Select Classifer",("SVM","Random Forest"))

        #add parameters
        def add_parameters(clf_name):
            """Selection of parameters"""
            if clf_name == "SVM":
                C = st.sidebar.slider("C",0.01,15.0)
                params["C"] = C
            elif clf_name == "Random Forest":
                max_depth = st.sidebar.slider("max_depth",2,15)
                n_estimators = st.sidebar.slider("n_estimators",1,100)
                params["max_depth"] = max_depth
                params["n_estimators"] = n_estimators
            return params
        
        add_parameters(classifer_name)

        #get classifers
        def get_classifiers(clf_name,params):
            clf = None
            if clf_name == "SVM":
                clf = SVC(C=params["C"])
            elif clf_name == "Random Forest":
                clf = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=7)
            return clf

        clf = get_classifiers(classifer_name,params)

        #Classification
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=7)

        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test,y_pred)
        st.write(f"classifier = {classifer_name}")
        st.write(f"accuracy = {acc}")

        

if __name__ == '__main__':
	main()