import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from itertools import product

@st.cache_data()
def getWeapons(file :str) -> pd.DataFrame:
    # TODO: weapons table as .csv
    return pd.read_csv(file)



# Dice functions

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
    percentages = [(counts[i] / total_outcomes) for i in range(len(counts))]
    df = pd.DataFrame({'result': unique_results, 'count': counts, 'fraction': percentages})
    return df



# Defaults
df_hit = d_df(8, 8)
df_dmg = d_df(8, 8)
armor_layer1 = {'layer': "Reinforced", 'coverage': 4, 'protection': 8}
#armor_layer_2 = {'layer': "Light armor", 'coverage': 8, 'protection': 4}
df_armor = pd.DataFrame([armor_layer1])

# Attack functions
def attackMelee(df_hit=df_hit, df_dmg=df_dmg, guard=8, df_armor=df_armor, tough=8):
    # make df of outcomes:
    # damage outcomes as x-axis
    # fraction/percentage as y-axis
    # vertical stop-line
    # plots: all clean dmgs, all layer dmgs, all total damages

    #deduct guard from df_hit.result
    df_hit.result -= guard
    fraction_blocked = df_hit[df_hit.result <= 0].fraction.sum()
    fraction_layer1 = df_hit[(df_hit.result <= armor_layer1['coverage']) & (0 < df_hit.result)].fraction.sum()
    fraction_clean = df_hit[df_hit.result > armor_layer1['coverage']].fraction.sum()

    st.write("fraction_blocked:", fraction_blocked)
    st.write("fraction_layer1:", fraction_layer1)
    st.write("fraction_clean:", fraction_clean)
    st.write("checksum:", fraction_blocked + fraction_layer1 + fraction_clean)

    df_dmg_layer1 = df_dmg.copy()
    df_dmg_layer1['fraction'] = df_dmg_layer1['fraction'] * fraction_layer1
    df_dmg_layer1['result'] = df_dmg_layer1['result'] - armor_layer1['protection']
    df_dmg_clean = df_dmg.copy()
    df_dmg_clean['fraction'] = df_dmg_clean['fraction'] * fraction_clean

    result_sum = pd.DataFrame()
    result_sum['result'] = pd.concat([df_dmg_clean['result'], df_dmg_layer1['result']]).unique()

    # Initialize the 'fraction_sum' column with zeros of type float
    result_sum['fraction_sum'] = 0.0

    # Calculate the sum of fractions for each 'result' that exists in both DataFrames
    for result in result_sum['result']:
        if result in df_dmg_clean['result'].unique():
            result_sum.loc[result_sum['result'] == result, 'fraction_sum'] += df_dmg_clean[df_dmg_clean['result'] == result]['fraction'].sum()
        if result in df_dmg_layer1['result'].unique():
            result_sum.loc[result_sum['result'] == result, 'fraction_sum'] += df_dmg_layer1[df_dmg_layer1['result'] == result]['fraction'].sum()
    result_sum = result_sum.sort_values(by='result')
    # Create a line plot with df_dmg_clean
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_dmg_clean['result'], y=df_dmg_clean['fraction'], mode='lines', name='df_dmg_clean'))
    # Create a line plot for df_dmg_layer1 and set the legend label to 'df_dmg_layer1' without an underscore
    fig.add_trace(go.Scatter(x=df_dmg_layer1['result'], y=df_dmg_layer1['fraction'], mode='lines', name='df_dmg_layer1'))
    # Create a line plot for the sum of fractions
    fig.add_trace(go.Scatter(x=result_sum['result'], y=result_sum['fraction_sum'], mode='lines', name='Sum of Fractions'))

    # Create vertical dashed lines at 0 and a specific value (e.g., tough)
    fig.add_shape(go.layout.Shape(type="line", x0=0, x1=0, xref="x", y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"), name='Zero'))
    fig.add_shape(go.layout.Shape(type="line", x0=tough, x1=tough, xref="x", y0=0, y1=1, yref="paper", line=dict(color="blue", width=2, dash="dash"), name='Tough Value'))

    # Show the combined plot
    st.plotly_chart(fig)
       

# Specific instructions
st.write("Professional soliders, armed, light armor, mutual combat, one strike:")
attackMelee()



# Old tutorial:
# row = 5
# cols = 5
# Random = np.random.randint(low=0, high=100, size=(row, cols))
# df = pd.DataFrame(Random)
# st.plotly_chart(px.scatter(df, x=0, y=1))
