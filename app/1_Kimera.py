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
layer1 = {'layer': "Guard", 'coverage': 8, 'protection': 8}
layer2 = {'layer': "Light armor", 'coverage': 11, 'protection': 3}
df_armor = pd.DataFrame([layer1, layer2])
#st.write(df_armor)

# Attack functions
def attackMelee(df_hit=df_hit, df_dmg=df_dmg, df_armor=df_armor, tough=8):
    dict_layer_fractions = {}
    dict_outcome = {}
    last_coverage = 0
    for row in df_armor.iterrows():
        layer = row[1]['layer']
        coverage = row[1]['coverage']
        protection = row[1]['protection']
        armor_hit_fraction = df_hit[(df_hit.result <= coverage) & (last_coverage < df_hit.result)].fraction.sum()
        # calculate damage for this layer
        df_dmg_layer = df_dmg.copy()
        df_dmg_layer['result'] -= protection
        df_dmg_layer['fraction'] = df_dmg_layer['fraction'] * armor_hit_fraction
        dict_layer_fractions[layer] = armor_hit_fraction
        dict_outcome[layer] = df_dmg_layer
        last_coverage = coverage
    # that which didn't strike any armor is "clean"
    fraction_clean = df_hit[df_hit.result > last_coverage].fraction.sum()
    df_dmg_clean = df_dmg.copy()
    df_dmg_clean['fraction'] = df_dmg_clean['fraction'] * fraction_clean
    dict_outcome['clean'] = df_dmg_clean

    # Initialize df_sum with a copy of the first DataFrame in dict_outcome
    df_sum = dict_outcome[list(dict_outcome.keys())[0]].copy()
    # Iterate through each key (DataFrame) in dict_outcome
    for key, df in dict_outcome.items():
        if key != list(dict_outcome.keys())[0]:
            # Group the DataFrame by 'result' and sum the 'fraction' values
            grouped = df.groupby('result')['fraction'].sum().reset_index()
            # Merge the grouped DataFrame with df_sum using 'result' as the key
            df_sum = pd.merge(df_sum, grouped, on='result', how='outer', suffixes=('', f'_{key}'))
    # Calculate the sum of 'fraction' columns from different layers per result
    df_sum['fraction'] = df_sum.filter(like='fraction').sum(axis=1)
    # Sort the df_sum DataFrame by the 'result' column
    df_sum = df_sum.sort_values(by='result').reset_index(drop=True)
    dict_outcome['sum'] = df_sum

    fig = go.Figure()
    for key, df in dict_outcome.items():
        fig.add_trace(go.Scatter(x=df['result'], y=df['fraction'], mode='lines', name=key))

    # Create vertical dashed lines at 0 and a specific value (e.g., tough)
    fig.add_shape(go.layout.Shape(type="line", x0=0, x1=0, xref="x", y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"), name='Zero'))
    fig.add_shape(go.layout.Shape(type="line", x0=tough, x1=tough, xref="x", y0=0, y1=1, yref="paper", line=dict(color="blue", width=2, dash="dash"), name='Tough Value'))
    # Show the combined plot
    st.plotly_chart(fig)

    # Calculate 'no_stop' and 'stopped' based on 'tough'
    no_stop = df_sum[df_sum['result'] <= tough]['fraction'].sum()
    stopped = df_sum[tough < df_sum['result']]['fraction'].sum()

    # print distribution of hits:
    struck_armor = 0
    for key, value in dict_layer_fractions.items():
        st.write(f"struck {key}: {value*100:.2f}%")
        struck_armor += value
    st.write(f"clean: {fraction_clean*100:.2f}%")
    st.write(f"checksum struck: {(struck_armor+fraction_clean)*100:.2f}%")
    st.write("")

    # Summarize outcomes:
    st.write(f"No Stop: {no_stop*100:.2f}%")
    st.write(f"Stopped: {stopped*100:.2f}%")       
    st.write(f"checksum damage: {(no_stop+stopped)*100:.2f}%")

# Specific instructions
st.write("Professional soliders, armed, light armor, mutual combat, one strike:")
attackMelee()