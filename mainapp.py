# Core Pkgs
import streamlit as st 
import streamlit.components.v1 as components
# EDA Pkgs
import pandas as pd 
import numpy as np
import codecs
from pandas_profiling import ProfileReport 
from streamlit_pandas_profiling import st_profile_report
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
    
    #st.title('Machine Learning Application')
    activities = ["Home","EDA","Plots","ML_Algorithms","Neural Network"]
    choice = st.sidebar.selectbox("Menu",activities)

    html_temp = """
        <div 
        style="background-color:royalblue;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;font-style: italic;">Classifying the survival of patients with heart failure using Various Machine Learning Algorithms</h1>
        </div>
        """
    components.html(html_temp)
    #data = st.file_uploader("Upload a Dataset", type=["csv","txt","xlsx"])
    data = pd.read_csv('heart_failure.csv')
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis using Pandas Profiling")
        if data is not None:
            
            df = pd.read_csv('heart_failure.csv')
            st.dataframe(df.head())
            lable = preprocessing.LabelEncoder()
            for col in df.columns:
                df[col] = lable.fit_transform(df[col])
            #pandas profiling
            profile = ProfileReport(df)
            st_profile_report(profile)
    
    elif choice == 'Plots':
        st.subheader("Data Visualization")

        if data is not None:
            df = pd.read_csv('heart_failure.csv')
            st.dataframe(df.head())
            lable = preprocessing.LabelEncoder()
            for col in df.columns:
                df[col] = lable.fit_transform(df[col])

        if st.checkbox("Show Value Counts"):
            st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
            st.pyplot()

        #Customized Plot
        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
            # Plot By Streamlit
            if type_of_plot == 'area':
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)
            elif type_of_plot == 'bar':
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)
            elif type_of_plot == 'line':
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)
    		# Custom Plot 
            elif type_of_plot:
                cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()

    elif choice == 'ML_Algorithms':
        st.subheader("Machine Learning Algorithms")

        if data is not None:
            df = pd.read_csv('heart_failure.csv')
            st.dataframe(df.head())
            lable = preprocessing.LabelEncoder()
            for col in df.columns:
                df[col] = lable.fit_transform(df[col])
               
        if st.checkbox("Summary"):
                st.write(df.describe())
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        #col_name = st.selectbox("Select Column Name",["X","y"])

        #if col_name == 'X':
        #    st.dataframe(X)
        #elif col_name == 'y':
        #    st.dataframe(y)
        
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
        clf_report = classification_report(y_test,y_pred)
        st.success(f"Classification Report:\n\n {clf_report}")
        st.warning(f"accuracy = {acc}")
        for i in range(1,10):
            st.write("Actual=%s, Predicted=%s" % (y_test[i], y_pred[i]))

    elif choice == 'Neural Network':
        st.subheader("Neural Networks (MLPClassifier)")

        if data is not None:
            df = pd.read_csv('heart_failure.csv')
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
        clf_report = classification_report(y_test,y_pred)
        st.success(f"Classification Report:\n\n {clf_report}")
        acc = accuracy_score(y_test,y_pred)
        st.warning(f"accuracy = {acc}")
        for i in range(1,10):
            st.write("Actual=%s, Predicted=%s" % (y_test[i], y_pred[i]))

if __name__ == '__main__':
	main()