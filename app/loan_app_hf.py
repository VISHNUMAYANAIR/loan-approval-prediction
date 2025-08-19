#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import joblib
import gradio as gr
import pickle

import os, joblib, pickle

# Base directory where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to models + features
rf_model_path = os.path.join(BASE_DIR, "models", "random_forest_model .pkl")
lr_model_path = os.path.join(BASE_DIR, "models", "logistic_model_l1 .pkl")
feature_columns_path = os.path.join(BASE_DIR, "artifacts", "feature_columns .pkl")

# Load models
rf_model = joblib.load(rf_model_path)
lr_model = joblib.load(lr_model_path)

# Load feature columns
with open(feature_columns_path, "rb") as f:
    feature_columns = pickle.load(f)


#define prediction function
def predict_loan_approval(model_choice, *user_inputs):
    #define raw column names (same order as gradio inputs)
    raw_feature_names = ['duration', 'credit_amount','checking_status','credit_history', 'purpose','savings_status',
                         'employment','installment_commitment','personal_status', 'other_parties','residence_since',
                         'property_magnitude','age','other_payment_plans','housing','exising_credits','job', 'own_telephone',
                         'num_dependents', 'foreign_worker']
    
    #convert inputs to DataFrame with correct column names
    user_df=pd.DataFrame([user_inputs], columns= raw_feature_names)
    
    #one-hot encode the inputs
    user_encoded = pd.get_dummies(user_df)
    
    # align with training columns
    #load feature columns used during training
    with open(feature_columns_path, "rb") as f:
        feature_columns = pickle.load(f)
    
    #add missing columns as 0
    for col in feature_columns:
        if col not in user_encoded.columns:
            user_encoded[col]=0
    
    #ensure correct column order
    user_encoded = user_encoded[feature_columns]
    
    #select model
    model=lr_model if model_choice == "Logistic Regression (L1)" else rf_model
    
    #predict
    prediction = model.predict(user_encoded)[0]
    proba = model.predict_proba(user_encoded)[0][1] #probablity of class=1
    
    result = "OK Approved" if prediction==1 else "Sorry Not Approved"
    return f"{result}(Confidence: {proba:.2f})" 


# In[45]:


#dynamically create Gradio input widgets
'''inputs=[]

for col in feature_columns:
    if 'duration' in col or 'amount' in col or 'age' in col:
        inputs.append(gr.Number(label=col))
    else:
        inputs.append(gr.Textbox(label=col))'''
    
#add model dropdown at the top
model_dropdown = gr.Dropdown( choices=['Logistic Regression (L1)', 'Random Forest'],value = 'Random Forest', label = "Select Model")


# In[46]:


#inputs - updated for user clarity

inputs = [gr.Number(label='Duration in Months', minimum= 0, maximum =72, value=12),
         gr.Number(label='Credit Amount', minimum=0,value=1000),
          
         gr.Dropdown(choices=['no checking','<0', '0<X<200', '>=200'], label = 'Checking Account Status'),
         gr.Dropdown(choices=['no credit/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/others'], label = 'Credit History'),
         gr.Dropdown(choices=['radio/tv', 'furniture/equipment', 'new car', 'used car', 'business', 'domestic appliance', 'repairs', 'vacation', 'education'], label = 'Loan Purpose'),
         gr.Dropdown(choices=['no known savings','<100', '500<=X<1000', '>=1000', '100<=X<500'], label = 'Savings Status'),
         gr.Dropdown(choices=['unemployed', '<1', '1<=X<4','4<=X<7', '>=7'], label = 'Employment Duration'),
          
         gr.Number(label='Installment % of Income', minimum=1,maximum=4),
         gr.Dropdown(choices=['male single', 'female div/dep/mar','male div/sep', 'male mar/wid'], label = 'Personal Status'),
         gr.Dropdown(choices=['none', 'guarantor', 'co applicant'], label = 'Other Parties'),
         gr.Number(label='Residence Duration (Years)', minimum=0, value=1),
         gr.Dropdown(choices=['real estate', 'life insurance', 'car', 'no known property'], label = 'Property Type'),
          
         gr.Number(label='Age', minimum=18, maximum=80),
         gr.Dropdown(choices=['none', 'bank', 'stores'], label = 'Other Payment Plans'),
         gr.Dropdown(choices=['own', 'for free', 'rent'], label = 'Housing Type'),
         gr.Number(label='Existing Credits at Bank',minimum=0,maximum=5),
         gr.Dropdown(choices=['skilled', 'unskilled resident', 'high qualif/self emp/mgmt', 'unemp/unskilled non res'], label = 'Job'),
         gr.Dropdown(choices=['yes', 'none'], label = 'Own Telephone ?'),
         gr.Number(label='Number of Liable People',minimum=0,maximum=6),
         gr.Dropdown(choices=['yes', 'no'], label = 'Foreign Worker?')]


# In[47]:


#LAUNCH THE APP
demo = gr.Interface(fn=predict_loan_approval, inputs=[model_dropdown]+inputs, outputs='text',title='Loan Approval Predictor',
                   description = 'Enter applicant details and choose a model to predict loan approval.')

demo.launch(share=True)


# In[49]:


if __name__ == "__main__":
    demo.launch(share=True)


# In[ ]:




