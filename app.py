import streamlit as st
import numpy as np

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")
from sklearn.linear_model import LinearRegression
y = df["logS"]
x = df.drop('logS', axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
lr = LinearRegression()
lr.fit(x_train, y_train)




def mp(input):
    return lr.predict(input).__getitem__(0)
st.title("Solubility Prediction")
st.write("This is a simple app to predict solubility using Linear Regression.")  
lp = st.number_input("1. LogP")
mw = st.number_input("2. Molecular Weight")
rb = st.number_input("3. Number of Rotatable Bonds")
ar = st.number_input("4. Aromatic Proportion")
if st.button("Predict"):
    st.write("Predicted logS: ", mp(np.array([[lp, mw, rb, ar]])))
