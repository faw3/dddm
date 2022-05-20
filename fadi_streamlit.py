#!/usr/bin/env python
# coding: utf-8

# In[1]:

import csv
from turtle import title
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns 
import matplotlib.pyplot as plt
import hydralit_components as hc
from streamlit_option_menu import option_menu 
import plotly.graph_objects as go
from random import sample

# In[2]:


st.set_page_config(layout="wide",page_title=None)

#Data upload option
uploaded_file = st.sidebar.file_uploader(label="Upload your data",type=["csv"])

global df
if uploaded_file is not None:
 df=pd.read_csv(uploaded_file)
 

    
df=pd.read_csv("marketing_streamlit.csv")
#Creating new columns from existing ones
df["CTR"] = df["Clicks"]/df["Impressions"]
df["CPC"] = df["Spent"]/ df["Clicks"]

#navigation bar
menu_data = [
    {'label':"Marketing Analytics",'icon':"bi bi-activity"},
    {'label':"Guidance", 'icon':"bi bi-question-circle"},
    {'label':"Get your Conversion!", 'icon':"bi bi-person-check-fill"},
    {'label':"More EDA", 'icon':"bi bi-bar-chart-fill"}
    
]

menu_id = hc.nav_bar(menu_definition=menu_data)
#First page
if menu_id == "Marketing Analytics":
  column1, column2 = st.columns([3,4])
  
  with column1:
    st.header("What is Marketing Analytics?")
    
  
    st.markdown("""Marketing Analytics refers to the use of data in order to reveal insights, patterns, and optimize future projects. Several factors,including budget, reach, and target segments' demographics play a major role in the success of a campaign in retaining customers.This tool exploits the power of marketing insights, visualizations, and machine learning models to estimate the number of people to be retained by future campaigns.
    
Take a look and the table below, representing different campaigns and their success in retaining audience. The ones highlighted with red are the ones who failed to retain any customer.Feel free to sort by any field to compare the difference with those that succeeded""")
    
  with column2:
    st.markdown("""The line graph gives you an overall understanding of the main concepts. Notice how when more people are reached, they do inspect about the campaign, but only the instances with a bright color are the ones retained!""")
    figure1 = px.scatter(df , x="Impressions", y="Total_Conversion", color="Approved_Conversion",labels={
                     "Total_Conversion": "Convertion"})
    st.plotly_chart(figure1)
  #Highlighting the variables with no retention
  st.subheader("Sort by Values")
  def highlight_conversion(s):
      return ['background-color: white']*len(s) if s.Approved_Conversion else ['background-color: red']*len(s)

  st.dataframe(df.style.apply(highlight_conversion, axis=1))
    

#Page 2
if menu_id == "Guidance":
  st.header("Give it a try!")
  st.markdown("""Please refer to the sidebar to inspect more the inter-relationships existing between the factors""")
  
  #Side Bar Filters
  with st.sidebar:
    conversion_rate = df["Approved_Conversion"].unique().tolist()
    Rate = st.selectbox("Choose Convertion Rate",conversion_rate,0)
    df = df[df["Approved_Conversion"]==Rate]

  
  with st.sidebar:
    click_choice=st.slider(
    "Clicks", min_value=0, max_value=500,step=10,value=100)
  df=df[df["Clicks"]<(click_choice)]
  
  with st.sidebar:
    spend_choice=st.slider(
      "Budget", min_value=0, max_value=650, step=5, value=100)
  df=df[df["Spent"]<(spend_choice)]
     
  #Plotting
  col1, col2,col3 = st.columns([3,3,3])


    
  with col1:
    #Gauge for impressions
   gauge1 =  go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = df["Impressions"].mean(),
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Impressions"}))
   gauge1.update_layout(
    autosize=False,
    width=400,
    height=350)
   st.plotly_chart(gauge1)

   
   #barplot for age and conversion
   bar1 = px.histogram(df, x="age",y="Approved_Conversion", width=400,height=350,labels={
                     "age": "Age",
                     "sum of Conversion": "Convertion",
                     
                 },
                title="Conversion by Age")
   st.plotly_chart(bar1)
   

   
  
  with col2:
   #Gauge for Conversion
     gauge2 =  go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = df["Total_Conversion"].mean(),
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Interested Audience"}))
     gauge2.update_layout(
    autosize=False,
    width=400,
    height=350)
     st.plotly_chart(gauge2)

     #Funnel 
     figg = go.Figure(go.Funnel(
      y = ["Money Spent", "Clicks","Interested People","Retained"],
      x = [df["Spent"].mean(),df["Clicks"].mean(),df["Total_Conversion"].mean(),df["Approved_Conversion"].mean()]))
     
     figg.update_layout(
    autosize=False,
    width=400,
    height=350,
    title="Process Steps")
     st.plotly_chart(figg)


     
     

  with col3:
    #Gauge for amount Spent 
       gauge3 =  go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = df["Spent"].mean(),
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Total Amount Spent in $"}))
       gauge3.update_layout(
    autosize=False,
    width=400,
    height=350)
       st.plotly_chart(gauge3)
      # Pie chart for conversion by gender
       pie1 = px.pie(df , values="Approved_Conversion", names="gender", width=400, height=350,hole=.3, title="Gender")
       st.plotly_chart(pie1)

       
    
    
        
    
    

#Page 3
if menu_id=="Get your Conversion!":
  st.header("Calculate an estimate of your Conversion Rate")
  st.subheader("Insights to keep in mind before selecting input:")
 
  col4,col5,col6 = st.columns([6,4,4])

  col4.metric(label = "Average Impressions",value=round(df.Impressions.mean()),delta = "1.5")
  col5.metric(label = "Average Cost Per Click", value= round(df.CPC.mean()), delta="1.2")
  col6.metric(label="Average Click Through Rate", value= "0.00016", delta="-1.5")
  st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

   #Cleaning Dataset

  df_clean  = pd.DataFrame(df.fillna(df.mean()))
    
  df_clean.isnull().sum().plot(kind="bar")
    
  df_clean["xyz_campaign_id"] = df["xyz_campaign_id"].astype('object')
    
  df_clean['age'] = df_clean['age'].str.replace("-","")
    
    
    
 
  #Encoding Categorical features    
  df1 = pd.get_dummies(data=df_clean , columns=["gender","age", "xyz_campaign_id"])
    
    
    
    
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_squared_error
  from sklearn.metrics import r2_score
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import cross_val_score
  from sklearn import neighbors
  from math import sqrt
  from sklearn.pipeline import Pipeline
  import pickle    



#Side bar inputs for prediction
  with st.sidebar:
    clickss = st.slider(
    "Clicks", min_value=0, max_value=430,step=10,value=100)
    df1=df1[df1["Clicks"]<(clickss)]
  
   
  with st.sidebar:
    money_spent= st.slider(
    "Budget in $", min_value=0, max_value=650,step=10,value=100)
    df1=df1[df1["Spent"]<(money_spent)]
  
   
  with st.sidebar:
   impressions_spent= st.slider(
      "Impressions", min_value=87, max_value=3100000,step=500,value=1000000)
  df1=df1[df1["Impressions"]<(impressions_spent)]

  with st.sidebar:
   total_people= st.slider(
      "Interested Audience", min_value=0, max_value=60,step=10,value=10)
  df1=df1[df1["Total_Conversion"]<(total_people)]

  with st.sidebar:
   age_rate = df1["age_3034"].unique().tolist()
   Rate_age = st.selectbox("Age range 30-34",age_rate,0)
   df1 = df1[df1["age_3034"]==Rate_age]

  with st.sidebar:
   age_rate2 = df1["age_3539"].unique().tolist()
   Rate_age2 = st.selectbox("Age range 35-39",age_rate2,0)
   df1 = df1[df1["age_3539"]==Rate_age2]

  with st.sidebar:
   age_rate3 = df1["age_4044"].unique().tolist()
   Rate_age3 = st.selectbox("Age range 40-44",age_rate3,0)
   df1 = df1[df1["age_4044"]==Rate_age3]

  with st.sidebar:
   age_rate4 = df1["age_4549"].unique().tolist()
   Rate_age4 = st.selectbox("Age range 45-49",age_rate4,0)
   df1 = df1[df1["age_4549"]==Rate_age4]

  with st.sidebar:
   gender_rate = df1["gender_M"].unique().tolist()
   Rate_gender = st.selectbox("Male?",gender_rate,0)
   df1 = df1[df1["gender_M"]==Rate_gender]

  with st.sidebar:
   gender_rate1 = df1["gender_F"].unique().tolist()
   Rate_gender1 = st.selectbox("Female?",gender_rate1,0)
   df1 = df1[df1["gender_F"]==Rate_gender1]  

  with st.sidebar:
   camp_rate1 = df1["xyz_campaign_id_916"].unique().tolist()
   Campaign_rate = st.selectbox("Campaign 916",camp_rate1,0)
   df1 = df1[df1["xyz_campaign_id_916"]==Campaign_rate]  

  with st.sidebar:
   camp_rate2 = df1["xyz_campaign_id_1178"].unique().tolist()
   Campaign_rate2 = st.selectbox("Campaign 1178",camp_rate2,0)
   df1 = df1[df1["xyz_campaign_id_1178"]==Campaign_rate2]  

  with st.sidebar:
   camp_rate3 = df1["xyz_campaign_id_936"].unique().tolist()
   Campaign_rate3 = st.selectbox("Campaign 936",camp_rate3,0)
   df1 = df1[df1["xyz_campaign_id_936"]==Campaign_rate3] 

  
    
  Scale = StandardScaler()
  df1 = pd.DataFrame(df1)
    
    
  df1 = df1.drop(["ad_id","interest","fb_campaign_id"], axis=1)
    
    
  #Splitting the data after cleaning
  train_set, test_set= train_test_split(df1, test_size=0.4, random_state=42)    
    
  
  #Splitting to X_train,y_train,X_valid,y_valid
  X_train = train_set.drop(["Approved_Conversion"], axis =1)
  y_train = train_set["Approved_Conversion"]
  X_valid = test_set.drop(['Approved_Conversion'], axis=1)
  y_valid = test_set["Approved_Conversion"]
    
    
    
  #Scaling features  
  X_train_scaled = Scale.fit_transform(X_train)
  X_valid_scaled = Scale.transform(X_valid)
    
    
    
  #Initiating Model  
  LR = LinearRegression()
    
  LR.fit(X_train, y_train)
  
  #Predicting using LR  
  np.random.seed(42)
  pred = LR.predict(X_valid)
  lin_mse = mean_squared_error(y_valid , pred)
  lin_rmse = np.sqrt(lin_mse)
  
  #Prediction button
  if st.button("Get Conversion"):
    st.subheader(pred.mean())
      

  #Plotting
  with col4:
    line1 = px.scatter(df, x="Spent",y="Total_Conversion", color="Approved_Conversion",width=550,height=350, title="Budget vs Conversion",labels={
                     "Spent": "Budget",
                     "sum of Conversion": "Convertion"})
    st.plotly_chart(line1)

  with col5:
    line2 = px.bar(df, x="age",y="Approved_Conversion",color="gender", width=350,height=350, title="Conversion by Gender and Age",labels={
                     "age": "Age",
                     "Approved_Conversion": "Convertion"})
    st.plotly_chart(line2)
 
    
  with col6:
    line3 = px.pie(df, values="Approved_Conversion",names="xyz_campaign_id",width=300,height=350,title="Conversion by Campaign")
    st.plotly_chart(line3)

#Page 4
if menu_id == "More EDA":
  col7,col8,col9 = st.columns([3,3,3])

  with col7:
    fig = plt.figure(figsize=(4,2))
    eda1 = sns.boxplot(x="age", y="CTR", data= df).set(title='Click Through Rate by Age')
    st.pyplot(fig)

    eda4 =px.scatter(df, x="Spent",y="Impressions", color="Approved_Conversion",width=500,height=400,title="Amount Spent and Impressions")
    st.plotly_chart(eda4)

  with col8:
     fig = plt.figure(figsize=(4,2))
     eda2 = sns.violinplot(x="gender", y="CPC", data= df).set(title='Cost Per Click by Gender')
     st.pyplot(fig)
     eda5 = px.scatter(df, x="Spent",y="Clicks", color="Approved_Conversion",width=500,height=400,title="Amount Spent and Clicks")
     st.plotly_chart(eda5)
  
     
  with col9:
    
     fig = plt.figure(figsize=(4,2))
     eda3 = sns.violinplot(x="age", y="Approved_Conversion", data=df).set(title="Most Responsive Age Group")
     st.pyplot(fig)


  

  
  
  


# Models that werent chosen: 

###Random Forest Regressor

#RFG = RandomForestRegressor()
#np.random.seed(42)
#RFG.fit(X_train_scaled, y_train)
#pred_1 = RFG.predict(X_valid_scaled)
#forest_mse = mean_squared_error(pred_1, y_valid)
#forest_rmse = np.sqrt(forest_mse)
#forest_rmse

### Cross validation with RFG
#scores = cross_val_score(RFG, X_valid, y_valid,
                         #scoring='neg_mean_squared_error', cv=10)
#LR_scores = np.sqrt(-scores)
#LR_scores.mean()

### KNN  
#rmse_val = [ ]
#k_range = range(1, 20)
#for K in k_range:
  #model= neighbors.KNeighborsRegressor(n_neighbors=K)
  #model.fit(X_train_scaled,y_train)
  #pred2 = model.predict(X_valid_scaled)
  #error = sqrt(mean_squared_error(y_valid,pred2))
  #rmse_val.append(error)
  #print("RMSE value for k=", K, 'is:', error)

### displaying the AUC for every neighbor
#plt.figure(figsize=(12, 6))
#plt.plot(k_range, rmse_val, color='red', linestyle='dashed', marker='o',
         #markerfacecolor='blue', markersize=10)
#plt.title('Knn RMSE for Different K Values')
#plt.xlabel('K Value')
#plt.ylabel('RMSE Score')


###Grid Search
#from sklearn.model_selection import GridSearchCV

#param_grid = [
    #{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8, 10]},
    #{'bootstrap': [False], 'n_estimators': [3, 5, 10], 'max_features': [2,3,4,8]},          
#]



#grid_search = GridSearchCV(RFG, param_grid, cv=5,
   #                        scoring='neg_mean_squared_error',
   #                        return_train_score=True)

#grid_search.fit(X_train_scaled, y_train)




# Best Grid Search Model
#grid_search.best_estimator_
# Performance of the different paramaters 
#cvres = grid_search.cv_results_
#for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
  #print(np.sqrt(-mean_score), params)

# Testing the best model

#final_model = grid_search.best_estimator_

#final_predictions = final_model.predict(X_valid_scaled)
#final_mse = mean_squared_error(y_valid, final_predictions)
#final_rmse = np.sqrt(final_mse)
#final_rmse









