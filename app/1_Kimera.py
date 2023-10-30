import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from itertools import product

@st.cache_data()
def getWeapons(file :str) -> pd.DataFrame:
    # TODO: weapons table as .csv
    return pd.read_csv(file)

def d(*dice):
    result = 0
    for d in dice:
        result += np.random.randint(low=1, high=d)
    return result

def d_list(*dice):
    results = [sum(combination) for combination in product(*[range(1, d + 1) for d in dice])]
    return sorted(results)

def d_df(*dice):
    results = d_list(*dice)
    unique_results = list(set(results))  # Get unique results
    unique_results.sort()
    counts = [results.count(result) for result in unique_results]
    total_outcomes = len(results)
    percentages = [(counts[i] / total_outcomes) * 100 for i in range(len(counts))]
    df = pd.DataFrame({'result': unique_results, 'count': counts, 'percentage': percentages})
    return df

df_chart = d_df(8, 8, 6)

st.plotly_chart(px.scatter(df_chart, x='result', y='percentage'))

''' Tutorial
row = 5
cols = 5

Random = np.random.randint(low=0, high=100, size=(row, cols))

df = pd.DataFrame(Random)

st.write("Hej")
st.write(df)

st.plotly_chart(px.scatter(df, x=0, y=1))
'''