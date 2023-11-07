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
@st.cache_data()
def d(*dice):
    result = 0
    for d in dice:
        result += np.random.randint(low=1, high=d)
    return result

@st.cache_data()
def d_list(*dice): # returns a sorted list of all results possible with *dice
    results = [sum(combination) for combination in product(*[range(1, d + 1) for d in dice])]
    return sorted(results)

@st.cache_data()
def d_df(*dice): # returns a dataframe with columns (unique)'result', 'count', and 'fraction'
    list_results = d_list(*dice)
    list_unique_results = list(set(list_results)) # discard duplicates
    list_counts = [list_results.count(result) for result in list_unique_results] # count occurance of each unique result
    total_counts = len(list_results)
    list_fractions = [(list_counts[i] / total_counts) for i in range(len(list_counts))] # calculate fraction of each unique result
    df = pd.DataFrame({'result': list_unique_results, 'count': list_counts, 'fraction': list_fractions})
    return df

# Defaults
df_hit = d_df(8, 8).rename(columns={'result': 'hit'})
df_dmg = d_df(8, 8)
df_crit = df_dmg.copy()
df_crit['result'] = df_crit['result'] + 6

# Sidebar
with st.sidebar:
    l1 = st.container()
    l2 = st.container()
    l2.subheader("Layer 2")
    layer2_coverage = l2.select_slider("L1Coverage", range(20))
    layer2_protection = l2.select_slider("L1Protection", range(10))
    l1.subheader("Layer 1")
    layer1_coverage = l1.select_slider("Coverage", range((layer2_coverage if layer2_coverage > 0 else 2)))
    layer1_protection = l1.select_slider("Protection", range(20))

layer1 = {'layer': "Reinforced", 'coverage': layer1_coverage, 'protection': layer1_protection}
layer2 = {'layer': "Light armor", 'coverage': layer2_coverage, 'protection': layer2_protection}
df_armor = pd.DataFrame([layer1, layer2])


# Attack functions
def attackMelee(df_hit=df_hit, df_dmg=df_dmg, df_crit=df_crit, guard = 3, block = 12, df_armor=df_armor, tough=8):
    # TODO: generalize to resolve ranged attacks with different arguments

    # resolve Guard; no crit section as crits by definition cannot strike guard
    df_hit.hit -= guard
    fraction_guard = df_hit[df_hit.hit < 1].fraction.sum()
    df_dmg_guard = df_dmg.copy()
    df_dmg_guard['result'] -= block
    df_dmg_guard['fraction'] = df_dmg_guard['fraction'] * fraction_guard    
    dict_outcome = {'guard': df_dmg_guard}

    # resolve armor layers
    dict_layer_fractions = {}
    last_coverage = 0 # lowest coverage that struck last layer
    for row in df_armor.iterrows():
        layer = row[1]['layer']
        coverage = row[1]['coverage']
        protection = row[1]['protection']
        df_hit_layer = df_hit[(df_hit.hit <= coverage) & (last_coverage < df_hit.hit)]
        armor_hit_fraction = df_hit_layer.fraction.sum()
        dict_layer_fractions[layer] = armor_hit_fraction
        df_dmg_layer = df_dmg.copy()
        if (9 < df_hit_layer.hit.max()): # if any hits on the layer are above 10, calculate crits for this layer
            # calculate fractions of non-crits and crits
            layer_nocrit_fraction = df_hit_layer[df_hit_layer.hit < 10].fraction.sum()
            layer_crit_fraction = df_hit_layer[9 < df_hit_layer.hit].fraction.sum()
            # apply fractions to damage and crit tables
            df_dmg_layer['result'] -= protection
            df_dmg_layer['fraction'] = df_dmg_layer['fraction'] * armor_hit_fraction * layer_nocrit_fraction
            df_crit_layer = df_crit.copy()
            df_crit_layer['result'] -= protection
            df_crit_layer['fraction'] = df_crit_layer['fraction'] * armor_hit_fraction * layer_crit_fraction
            # merge damage and crit tables
            merged_df = df_dmg_layer.merge(df_crit_layer, on='result', suffixes=('_dmg', '_crit'), how='outer').fillna(0)
            # sum 'count' and 'fraction' columns
            merged_df['count'] = merged_df['count_dmg'] + merged_df['count_crit']
            merged_df['fraction'] = merged_df['fraction_dmg'] + merged_df['fraction_crit']
            df_dmg_layer = merged_df[['result', 'count', 'fraction']]
        else: # calculate regular damage only for this layer
            df_dmg_layer['result'] -= protection
            df_dmg_layer['fraction'] = df_dmg_layer['fraction'] * armor_hit_fraction
        dict_outcome[layer] = df_dmg_layer
        last_coverage = coverage

    # that which didn't strike any armor is "clean"
    df_dmg_clean = df_dmg.copy()
    df_hit_clean = df_hit[(df_hit.hit > last_coverage)]
    fraction_clean = df_hit_clean.fraction.sum()
    if 9 < df_hit_clean.hit.max(): # check for crits:
        # calculate RELATIVE fractions of non-crits and crits
        clean_counts = df_hit_clean['count'].sum()
        clean_nocrit_counts = df_hit_clean[df_hit_clean.hit < 10]['count'].sum()
        clean_nocrit_fraction = clean_nocrit_counts / clean_counts
        clean_crit_fraction = 1 - clean_nocrit_fraction
        # apply fractions to damage and crit tables
        df_dmg_clean['fraction'] = df_dmg_clean['fraction'] * clean_nocrit_fraction * fraction_clean
        df_crit_clean = df_crit.copy()
        df_crit_clean['fraction'] = df_crit_clean['fraction'] * clean_crit_fraction * fraction_clean
        merged_df = df_dmg_clean.merge(df_crit_clean, on='result', suffixes=('_dmg', '_crit'), how='outer').fillna(0)
        # sum 'count' and 'fraction' columns
        merged_df['count'] = merged_df['count_dmg'] + merged_df['count_crit']
        merged_df['fraction'] = merged_df['fraction_dmg'] + merged_df['fraction_crit']
        df_dmg_clean = merged_df[['result', 'count', 'fraction']]
    else:            
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
    fig.add_shape(go.layout.Shape(type="line", x0=1, x1=1, xref="x", y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"), name='Zero'))
    fig.add_shape(go.layout.Shape(type="line", x0=tough, x1=tough, xref="x", y0=0, y1=1, yref="paper", line=dict(color="blue", width=2, dash="dash"), name='Tough Value'))
    # Show the combined plot
    st.plotly_chart(fig)

    # write distribution of hit outcomes:
    struck_armor = 0
    st.write(f"parried (guard): {fraction_guard*100:.2f}%")
    for key, value in dict_layer_fractions.items():
        st.write(f"struck {key}: {value*100:.2f}%")
        struck_armor += value
    st.write(f"struck clean: {fraction_clean*100:.2f}%")
    st.write(f"checksum struck: {(fraction_guard+struck_armor+fraction_clean)*100:.2f}%")
    st.write("")

    # Calculate 'no_stop' and 'stopped' based on 'tough'
    no_stop = df_sum[df_sum['result'] <= tough]['fraction'].sum()
    stopped = df_sum[tough < df_sum['result']]['fraction'].sum()

    # write distribution of damage outcomes:
    st.write(f"No Stop: {no_stop*100:.2f}%")
    st.write(f"Stopped: {stopped*100:.2f}%")       
    st.write(f"checksum stoppage: {(no_stop+stopped)*100:.2f}%")

# Specific instructions
attackMelee()