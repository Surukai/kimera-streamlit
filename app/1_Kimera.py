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
df_crit = d_df(8, 8, 12)

guard = 4
block = 12
tough = 8

layer1 = {'layer': "armor - Reinforced", 'coverage': 2, 'protection': 4}
layer2 = {'layer': "armor - Light", 'coverage': 6, 'protection': 2}


# Sidebar
with st.sidebar:
    l1 = st.container()
    l2 = st.container()
    l1.subheader(layer1['layer'])
    layer1_coverage = l1.slider("Coverage", 0, 20, layer1['coverage'])
    layer1_protection = l1.slider("Protection", 0, 20, layer1['protection'])
    l2.subheader(layer2['layer'])
    layer2_coverage = l2.slider("Coverage", 0, 20, layer2['coverage'])
    layer2_protection = l2.slider("Protection", 0, 20, layer2['protection'])

layer1['coverage'] = layer1_coverage
layer1['protection'] = layer1_protection
layer2['coverage'] = layer2_coverage
layer2['protection'] = layer2_protection
df_armor = pd.DataFrame([layer1, layer2])


def merge_dfs(dict_dfs):
    list_merge = []
    # Iterate dict_hit_dmg and append them to list_df_merged
    for key, df in dict_dfs.items():
        if 'result' in df.columns:
            list_merge.append(df[['result', 'count', 'fraction']])
    # Concatenate the DataFrames in the list
    df_merged = pd.concat(list_merge)
    df_merged = df_merged.groupby('result', as_index=False).sum()
    return df_merged[['result', 'count', 'fraction']]


def attack(df_hit=df_hit, df_dmg=df_dmg, df_crit=df_crit, df_armor=df_armor, guard=None, block=None, frame=None, cover=None, distance=0, verbose=False):
    '''
    df_hit: hit distribution DataFrame with columns 'hit', 'count', and 'fraction'
    df_dmg: damage distribution DataFrame with columns 'result', 'count', and 'fraction'
    df_crit: critical damage distribution DataFrame with columns 'result', 'count', and 'fraction'
    df_armor: armor DataFrame with columns 'layer', 'coverage', and 'protection'
    guard: for melee attacks. Reduces hit rolls; 0 or lower are blocked
    block: (two uses) Reduces damage of blocked attacks in melee, and reduces damage of ranged attacks that hit the cover
    frame: (working name) difficulty to hit with ranged attacks, regardless of cover. The smaller the target, the greater the frame value
    cover: additional armor layer - the greater of cover value of environment and Stealth attribute
    distance (in m): if the distance is greater than 50m, the GM can apply a bonus to the frame value
        this could be as simple as +5 up to 100m, when it beomces +5,
        or as precise as +1 per 10m over 50m.
        As a typical frame is 5, the typical penalty would be exactly 1/10 of the distance: 6 at 60m, 7 at 70m, etc (minimum of 5).
    '''
    df_hit = df_hit.copy()
    crit_threshold = 9 # results 10 or more are crits
    melee = bool(frame is None)

    if verbose:
        if melee:
            st.write(f"melee attack {df_hit['hit'].min()}-{df_hit['hit'].max()}, dmg {df_dmg.result.min()}-{df_crit.result.max()} . . . vs . . . guard {guard}, block {block}")
        else:
            st.write(f"ranged attack {df_hit['hit'].min()}-{df_hit['hit'].max()}, dmg {df_dmg.result.min()}-{df_crit.result.max()} . . . vs . . . frame {frame}, cover {cover}, block {block}, distance {distance}m")

    if melee: # resolve guard for melee attack
        df_hit.hit -= guard
    else: # resolve frame and distance for ranged attack
        range_penalty = (frame + max(0, int(distance/10-5)))
        df_hit.hit -= range_penalty
        if verbose:
            st.write(f"frame {frame}, distance {distance}; penalty {range_penalty}")

    # initiate general dicts of outcomes
    dict_hit_dmg = {}
    dict_armor = {}
    dict_clean = {}
    dict_block = {}
    dict_cover = {}

    for row in df_hit.iterrows():
        hit = int(row[1]['hit'])
        crit = bool(crit_threshold < hit)
        hit_fraction = row[1]['fraction']
        #fetch relevant damage table
        if crit:
            hit_df_dmg = df_crit.copy()
        else:
            hit_df_dmg = df_dmg.copy()
        hit_df_dmg['fraction'] *= hit_fraction
        # resolve block for melee attacks
        if melee:
            if hit < 1:
                hit_df_dmg['result'] -= block
                dict_block[str(hit)] = hit_df_dmg
        # resolve cover for ranged attacks
        elif (cover is not None) and (hit <= cover):
            hit_df_dmg['result'] -= block
            dict_cover[str(hit)] = hit_df_dmg
        # resolve armor layers: best protection that is struck
        df_layers_struck = df_armor[(hit <= df_armor.coverage) & (0 < df_armor.coverage)]
        if not df_layers_struck.empty:
            best_protection_row = df_layers_struck.loc[df_layers_struck['protection'].idxmax()]
            best_protection = best_protection_row['protection']
            best_layer = best_protection_row['layer']
            hit_df_dmg['result'] -= best_protection
            dict_armor[best_layer] = hit_df_dmg
        else:
            dict_clean[str(hit)] = hit_df_dmg
        dict_hit_dmg[str(hit)] = hit_df_dmg
        #st.write(f"hit: {hit}, crit: {crit}, fraction: {hit_fraction}, best_protection: {best_protection}, result range: {hit_df_dmg['result'].min()} to {hit_df_dmg['result'].max()}")

    # compile dict of outcomes
    dict_outcome = {'sum': merge_dfs(dict_hit_dmg)}
    for key, df in dict_armor.items():
        dict_outcome[key] = df
    if dict_clean:
        dict_outcome['clean'] = merge_dfs(dict_clean)
    if dict_block:
        dict_outcome['block'] = merge_dfs(dict_block)
    if dict_cover:
        dict_outcome['cover'] = merge_dfs(dict_cover)
    return dict_outcome


def report(dict_outcome, tough):
    # Calculate 'no_stop' and 'stopped' based on 'tough'
    df = dict_outcome['sum']
    no_stop = df[df['result'] <= tough]['fraction'].sum()
    stopped = df[tough < df['result']]['fraction'].sum()
    st.write(f"Stopped: {stopped*100:.2f}% ({no_stop*100:.2f}% failure)")

    fig = go.Figure() # Plotting
    for key, df in dict_outcome.items():
        if df is not None:
            fig.add_trace(go.Scatter(x=df['result'], y=df['fraction'], mode='lines', name=key))
    # Create vertical dashed lines at 1 and a specific value (e.g. tough)
    fig.add_shape(go.layout.Shape(type="line", x0=1, x1=1, xref="x", y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"), name='One'))
    fig.add_shape(go.layout.Shape(type="line", x0=tough, x1=tough, xref="x", y0=0, y1=1, yref="paper", line=dict(color="blue", width=2, dash="dash"), name='Tough'))
    st.plotly_chart(fig) # Show the combined plot



# Specific instructions, crit: dice vs +10
df_hit = d_df(12, 12).rename(columns={'result': 'hit'})
report(dict_outcome=attack(df_hit=df_hit, df_crit=df_crit, frame=5, cover=2, block=4, verbose=True), tough=tough)
df_crit = df_dmg.copy()
df_crit['result'] = df_crit['result'] + 10
report(dict_outcome=attack(df_hit=df_hit, df_crit=df_crit, frame=5, cover=2, block=4, verbose=True), tough=tough)

#report(dict_outcome=attack(guard=6, block=12, verbose=True), tough=tough)