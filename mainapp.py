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
from sklearn.neural_network import MLPClassifier,BernoulliRBM
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,f1_score,roc_auc_score


def main():
    """Automated ML App"""
    st.title('Machine Learning Application')
    activities = ["EDA","Plots","ML_Algorithms","Neural Network"]
    choice = st.sidebar.selectbox("Select Activities",activities)

    data = st.file_uploader("Upload a Dataset", type=["csv","txt","xlsx"])
    
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            lable = preprocessing.LabelEncoder()
            for col in df.columns:
                df[col] = lable.fit_transform(df[col])

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
                plt.subplots(figsize=(10,10));
                plt.matshow(df.corr())
                corr = df.corr()
                corr.style.background_gradient(cmap='coolwarm').set_precision(2)
                st.pyplot()
            
            if st.checkbox("Correlation Plot -- seaborn"):
                plt.subplots(figsize=(12,10));
                st.write(sns.heatmap(df.corr(),annot=True,fmt='.2g'))
                st.pyplot()

            if st.checkbox("Pie Plot"):
                all_columns = df.columns.to_list()
                column_to_plot = st.selectbox("Select 1 Column", all_columns)
                pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                plt.subplots(figsize=(12,10));
                st.write(pie_plot)
                st.pyplot()
    
    elif choice == 'Plots':
        st.subheader("Data Visualization")

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            lable = preprocessing.LabelEncoder()
            for col in df.columns:
                df[col] = lable.fit_transform(df[col])

        if st.checkbox("Show Value Counts"):
            st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
            st.pyplot()

        #Customized Plot

    elif choice == 'ML_Algorithms':
        st.subheader("Machine Learning Algorithms")

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            lable = preprocessing.LabelEncoder()
            for col in df.columns:
                df[col] = lable.fit_transform(df[col])
               
        if st.checkbox("Summary"):
                st.write(df.describe())
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        col_name = st.selectbox("Select Column Name",["X","y"])

        if col_name == 'X':
            st.dataframe(X)
        elif col_name == 'y':
            st.dataframe(y)
        
        st.write("Number of classes",len(np.unique(y)))
        params = dict()
        classifer_name = st.sidebar.selectbox("Select Classifer",("SVM Linear","SVM Radial","Decision Tree","Random Forest"))

        #add parameters
        def add_parameters(clf_name):
            """Selection of parameters"""
            if clf_name == "SVM Linear":
                C = st.sidebar.slider("C",0.01,15.0)
                params["C"] = C
            elif clf_name == "SVM Radial":
                C = st.sidebar.slider("C",0.01,15.0)
                params["C"] = C
            elif clf_name == "Decision Tree":
                max_depth = st.sidebar.slider("max_depth",2,15)
                max_leaf_nodes = st.sidebar.slider("max_leaf_nodes",2,20)
                params["max_depth"] = max_depth
                params["max_leaf_nodes"] = max_leaf_nodes
            elif clf_name == "Random Forest":
                max_depth = st.sidebar.slider("max_depth",2,15)
                n_estimators = st.sidebar.slider("n_estimators",1,200)
                params["max_depth"] = max_depth
                params["n_estimators"] = n_estimators
            return params
        
        add_parameters(classifer_name)

        #get classifers
        def get_classifiers(clf_name,params):
            clf = None
            if clf_name == "SVM Linear":
                clf = SVC(C=params["C"],kernel='linear')
            elif clf_name == "SVM Radial":
                clf = SVC(C=params["C"],kernel='rbf')
            elif clf_name == "Decision Tree":
                clf = DecisionTreeClassifier(max_depth=params["max_depth"],max_leaf_nodes=params["max_leaf_nodes"],random_state=100)
            elif clf_name == "Random Forest":
                clf = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=100)
            
            return clf

        clf = get_classifiers(classifer_name,params)

        #Classification
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=100)

        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test,y_pred)
        st.write(f'<div style="color: #1C2331; font-size: medium; font-style: italic; padding: 15px; background-color:#b2dfdb;border-radius:5px;">Classifier = {classifer_name}</div></br>',unsafe_allow_html=True)
        roc_auc = roc_auc_score(y_test,y_pred)
        st.success(f"ROC AUC = {roc_auc}")
        f_score = f1_score(y_test,y_pred)
        st.success(f"F1 score = {f_score}")
        st.warning(f"accuracy = {acc}")

    elif choice == 'Neural Network':
        st.subheader("Neural Networks (MLPClassifier)")

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            lable = preprocessing.LabelEncoder()
            for col in df.columns:
                df[col] = lable.fit_transform(df[col])
            
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        params = dict()
        classifer_name = "MLPClassifier"
        
        def add_parameters(clf_name):
            """Selection of parameters"""
            if clf_name == "MLPClassifier":
                max_iter = st.sidebar.slider("max_iter",2,30)
                params["max_iter"] = max_iter
            
            return params
        
        add_parameters(classifer_name)

        #get classifers
        def get_classifiers(clf_name,params):
            clf = None
            if clf_name == "MLPClassifier":
                clf = MLPClassifier(max_iter=params["max_iter"])
            
            return clf
        
        clf = get_classifiers(classifer_name,params)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=100)

        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        
        st.write(f'<div style="color: #1C2331; font-size: medium; font-style: italic; padding: 15px; background-color:#b2dfdb;border-radius:5px;">Classifier = {classifer_name}</div></br>',unsafe_allow_html=True)
        roc_auc = roc_auc_score(y_test,y_pred)
        st.success(f"ROC AUC = {roc_auc}")
        f_score = f1_score(y_test,y_pred)
        st.success(f"F1 score = {f_score}")
        acc = accuracy_score(y_test,y_pred)
        st.warning(f"accuracy = {acc}")

if __name__ == '__main__':
	main()