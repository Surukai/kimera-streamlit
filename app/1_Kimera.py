import streamlit as st
import plotly.express as px
import pandas as pd

@st.cache_data()
def getWeapons(file :str) -> pd.DataFrame:
    return pd.read_csv(file)



st.write("Hej")