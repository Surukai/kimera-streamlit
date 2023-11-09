import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from itertools import product
import colorsys

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
    # TODO: check dice vs set of supported dice; break up high numbers into supported dice
    list_results = d_list(*dice)
    list_unique_results = list(set(list_results)) # discard duplicates
    list_counts = [list_results.count(result) for result in list_unique_results] # count occurance of each unique result
    total_counts = len(list_results)
    list_fractions = [(list_counts[i] / total_counts) for i in range(len(list_counts))] # calculate fraction of each unique result
    df = pd.DataFrame({'result': list_unique_results, 'count': list_counts, 'fraction': list_fractions})
    return df



# PC Attack test defaults
df_hit = d_df(8, 8).rename(columns={'result': 'hit'})
df_dmg = d_df(8, 8)
df_crit = d_df(8, 8, 12)

guard = 4
block = 12
tough = 8

# PC Defend test defaults
df_guard = d_df(8, 8)
df_tough_guard = d_df(8, 8, 8)
df_dodge = d_df(8, 8)
df_tough = d_df(8)
df_cover = d_df(8)
df_frame = d_df(10)
hit = 8
dmg = 8
crit_dmg = dmg + 10

# Armor defaults ()
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
    Resolves PC attack (rolled) vs NPC (static)
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
    Returns a dict of DataFrames with columns 'result', 'count', and 'fraction', describing the outcomes of the attack
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
    dict_crit = {}

    for row in df_hit.iterrows():
        hit = int(row[1]['hit'])
        critical = bool(crit_threshold < hit)
        hit_fraction = row[1]['fraction']
        #fetch relevant damage table
        if critical:
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
        if critical:
            dict_crit[str(hit)] = hit_df_dmg
        #st.write(f"hit: {hit}, critical: {critical}, fraction: {hit_fraction}, best_protection: {best_protection}, result range: {hit_df_dmg['result'].min()} to {hit_df_dmg['result'].max()}")

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
    if dict_crit:
        dict_outcome['crit'] = merge_dfs(dict_crit)
    return dict_outcome



def defend(df_guard=None, df_tough_guard=None, df_dodge=None, df_tough=df_tough, hit=hit, dmg=dmg, crit_dmg=crit_dmg, df_armor=df_armor, df_frame=None, df_cover=None, distance=0, verbose=False):
    '''
    Resolves PC defense (rolled) against an NPC attack (static)
    (params)
    distance (in m): if the distance is greater than 50m, the GM can apply a bonus to the frame value
        this could be as simple as +5 up to 100m, when it beomces +5,
        or as precise as +1 per 10m over 50m.
        As a typical frame is 5, the typical penalty would be exactly 1/10 of the distance: 6 at 60m, 7 at 70m, etc (minimum of 5).
    Returns a dict of DataFrames with columns 'result', 'count', and 'fraction', describing the outcomes of the attack
    '''
    st.write(f"PC defense {df_guard['result'].min()}-{df_guard['result'].max()}, Block {df_tough_guard['result'].min()}-{df_tough_guard['result'].max()}, Tough {df_tough['result'].min()}-{df_tough['result'].max()}({int(df_tough['result'].max()/2)}) . . . vs . . . hit {hit}, dmg {dmg}-{crit_dmg}")
    
    dict_guard_tough = {}
    dict_block = {}
    dict_armor = {}
    dict_clean = {}
    dict_crit = {}

    for row in df_guard.iterrows():
        guard = int(row[1]['result'])
        guard_fraction = row[1]['fraction']
        diff = hit-guard
        critical = bool(9 < diff)
        guard_dmg = dmg
        if diff < 0:
            guard_df_tough = df_tough_guard.copy()
            guard_df_tough['result'] = guard_dmg - guard_df_tough['result']
            guard_df_tough['fraction'] *= guard_fraction
            dict_block[str(guard)] = guard_df_tough
        else:
            if critical:
                guard_dmg = crit_dmg
            guard_df_tough = df_tough.copy()
            guard_df_tough['result'] = guard_dmg - guard_df_tough['result']
            guard_df_tough['fraction'] *= guard_fraction
            df_layers_struck = df_armor[(diff < df_armor.coverage)]
            if not df_layers_struck.empty:
                best_protection_row = df_layers_struck.loc[df_layers_struck['protection'].idxmax()]
                best_protection = best_protection_row['protection']
                best_layer = best_protection_row['layer']
                guard_df_tough['result'] -= best_protection
                dict_armor[best_layer] = guard_df_tough
            else:
                #st.write(f"{hit}-{guard}={diff} No armor layer struck")
                dict_clean[str(guard)] = guard_df_tough
            if critical:
                dict_crit[str(guard)] = guard_df_tough
        dict_guard_tough[str(guard)] = guard_df_tough

    # compile dict of outcomes
    dict_outcome = {'sum': merge_dfs(dict_guard_tough)}
    for key, df in dict_armor.items():
        dict_outcome[key] = df
    if dict_clean:
        dict_outcome['clean'] = merge_dfs(dict_clean)
    if dict_block:
        dict_outcome['block'] = merge_dfs(dict_block)
    if dict_crit:
        dict_outcome['crit'] = merge_dfs(dict_crit)
    return dict_outcome



def report(dict_outcome, defend=False, tough=None):
    '''
    defend flag: active defense is interpreted differently: an active roll has to BEAT the passive; not just equal it
    '''
    if tough is not None: # Calculate 'no_stop' and 'stopped' based on 'tough'
        df = dict_outcome['sum']
        if defend:
            no_effect = df[df['result'] < 0]['fraction'].sum()
            staggered = df[(0 <= df['result']) & (df['result'] < tough)]['fraction'].sum()
            stopped = df[tough <= df['result']]['fraction'].sum()
            st.write(f"Stopped: {stopped*100:.2f}% - Staggered {staggered*100:.2f}% - ({no_effect*100:.2f}% failure)")
        else:
            no_effect = df[df['result'] < tough]['fraction'].sum()
            stopped = df[tough <= df['result']]['fraction'].sum()
            st.write(f"Stopped: {stopped*100:.2f}% - ({no_effect*100:.2f}% failure)")

    fig = go.Figure() # Plotting
    num_keys = len(dict_outcome)
    first_color = (255, 255, 255)
    # generate a spectrum of colors
    colors = [first_color] + [
        tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / num_keys, 1, 1))
        for i in range(num_keys - 1)]
    # Generate a spectrum of colors
    for index, (key, df) in enumerate(dict_outcome.items()):
        if df is not None:
            line_color = f'rgba({colors[index][0]}, {colors[index][1]}, {colors[index][2]}, 0.5)'
            fig.add_trace(go.Scatter(x=df['result'], y=df['fraction'], mode='lines', name=key, line=dict(color=line_color)))

    # Create vertical dashed lines at 1 and, if available, a specific value (e.g. tough)
    fig.add_shape(go.layout.Shape(type="line", x0=1-defend, x1=1-defend, xref="x", y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"), name='One'))
    if tough is not None:
        fig.add_shape(go.layout.Shape(type="line", x0=tough-defend, x1=tough-defend, xref="x", y0=0, y1=1, yref="paper", line=dict(color="blue", width=2, dash="dash"), name='Tough'))

    st.plotly_chart(fig) # Show the combined plot



# Instructions
test_attack = False
test_melee = True
test_dodge = False

if test_attack: #crit: dice vs +10
    df_hit = d_df(8, 8).rename(columns={'result': 'hit'})
    if test_melee:
        report(dict_outcome=attack(df_hit=df_hit, df_crit=df_crit, guard=8, block=12, verbose=True), tough=12)
    else:
        report(dict_outcome=attack(df_hit=df_hit, df_crit=df_crit, frame=5, cover=12, block=20, verbose=True), tough=8)

    df_crit = df_dmg.copy()
    df_crit['result'] = df_crit['result'] + 10
    if test_melee:
        report(dict_outcome=attack(df_hit=df_hit, df_crit=df_crit, guard=8, block=12, verbose=True), tough=12)
    else:
        report(dict_outcome=attack(df_hit=df_hit, df_crit=df_crit, frame=5, cover=12, block=20, verbose=True), tough=8)
else: # Defense test
    df_hit = d_df(8,6,8,6).rename(columns={'result': 'hit'})
    df_crit = df_dmg.copy()
    df_crit['result'] = df_crit['result'] + 10
    hit=14
    dmg=8
    crit_dmg=18
    report(dict_outcome=defend(df_guard=df_guard, df_tough_guard=df_tough_guard, df_dodge=df_dodge, df_tough=df_tough, hit=hit, dmg=dmg, crit_dmg=crit_dmg, df_armor=df_armor, df_frame=df_frame, df_cover=df_cover, distance=0, verbose=True), defend=True, tough=tough/2)
    report(dict_outcome=attack(df_hit=df_hit, df_crit=df_crit, guard=8, block=12, verbose=True), tough=8)
