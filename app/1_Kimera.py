import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

@st.cache_data()
def getWeapons(file :str) -> pd.DataFrame:
    return pd.read_csv(file)

row = 5
cols = 5

Random = np.random.randint(low=0, high=100, size=(row, cols))

df = pd.DataFrame(Random)

st.write("Hej")
st.write(df)

st.plotly_chart(px.scatter(df, x=0, y=1))