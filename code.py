import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("Zomato Bangalore Restaurants")
st.subheader("About the Project")
st.write("This project focuses on")

df = pd.read_csv('zomato.csv')

