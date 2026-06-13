import pandas as pd
import numpy as np
import streamlit as st
import joblib
import tensorflow
from tensorflow.keras.models import load_model

#page structure

st.title('Welcome to credit card fraud detector')
st.divider()

st.write('Please input details below')

credit_card_amount = st.number_input('Enter Credit Amount', 1, 100000, 50000)

gender = st.selectbox('Select Gender', ['Male', 'Female'])

if gender  == 'Male':
    gender = 1
else:
    gender = 2

education_level = st.selectbox('select education level', ['graduate school', 'university', 'high school', 'others'])

if education_level == 'graduate school':
    education_level = 1
elif education_level == 'university':
    education_level =2
elif education_level == 'high school':
    education_level =3
else:
    education_level =4

marital_status = st.selectbox('select marital status', ['Married', 'Single', 'Other'])

if marital_status == 'Married':
    marital_status = 1
elif marital_status == 'Single':
    marital_status  = 2
else:
    marital_status = 3

age = st.slider('select age', 18, 80, 45 )

total_bill_due = st.number_input('Enter total due amount', 1,10000000)

total_bill_paid = st.number_input('Enter total bill paid', 1, 10000000)

delay_cat = ['pay duly', 
            'payment delay for one month', 
            'payment delay for two months', 
            'payment delay for three months',
            'payment delay for four months',
            'payment delay for five months',
            'payment delay for six months',
            'payment delay for seven months',
            'payment delay for eight months',
            'payment delay for nine months']

Average_repayment_delay = st.selectbox('Select average repayment delays',delay_cat, index= 0 )


if Average_repayment_delay  == 'pay duly':
    Average_repayment_delay = -1
elif Average_repayment_delay == 'payment delay for one month':
    Average_repayment_delay  = 1
elif Average_repayment_delay == 'payment delay for two months':
    Average_repayment_delay  = 2
elif Average_repayment_delay == 'payment delay for three months':
    Average_repayment_delay  = 3
elif Average_repayment_delay == 'payment delay for four months':
    Average_repayment_delay  = 4
elif Average_repayment_delay == 'payment delay for five months':
    Average_repayment_delay  = 5
elif Average_repayment_delay == 'payment delay for six months':
    Average_repayment_delay  = 6
elif Average_repayment_delay == 'payment delay for seven months':
    Average_repayment_delay  = 7
elif Average_repayment_delay == 'payment delay for eight months':
    Average_repayment_delay  = 8
else:
    Average_repayment_delay  = 9



#all input data

input = {'Credit Amount' : credit_card_amount,
         'gender' : gender,
          'Education level': education_level,
          'Marital Status': marital_status,
           "Age" : age,
           'totall bill statement' : total_bill_due,
            'total amount paid': total_bill_paid,
             'avg repayment': Average_repayment_delay}


#loading models

model = load_model('model.h5')

scaler = joblib.load('scaler.pkl')





st.divider()

predict_button = st.button("Click to check credit card fraud risk")

if predict_button:
    #scale values first 

    inputdf = pd.DataFrame(input, index= [0])


    num_features = inputdf[['Credit Amount', 'Age', 'totall bill statement','total amount paid', 'avg repayment']]

    num_features = scaler.transform(num_features)

    inputdf.drop(['Credit Amount', 'Age', 'totall bill statement','total amount paid', 'avg repayment'], axis= 1, inplace= True)

    input = np.concatenate([inputdf, num_features],axis = 1)

    #make prediction with ANN
    prediction = model.predict(input)
    
    if prediction[0][0] > 0.5:
        st.write('Card holder is likely to default')
    else:
        st.write('Card holder is not likely to default')

else:
    st.write('Please enter all values and predict again')

