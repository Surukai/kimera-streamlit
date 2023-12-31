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
KIMERA_DICE = {4, 6, 8, 10, 12}

@st.cache_data()
def d(*dice):
    result = 0
    for d in dice:
        result += np.random.randint(low=1, high=d)
    return result

#@st.cache_data()
def split_dice(value):
    if value == 14:
        return [8, 6]
    elif value == 16:
        return [8, 8]
    elif value == 18:
        return [10, 8]
    elif value == 20:
        return [10, 10]
    elif value == 22:
        return [12, 10]
    elif value == 24:
        return [12, 12]
    else:
        st.write(f"WARNING! split_dice: d{value} is not a Kimera dice")
        return [value] # return original

#@st.cache_data()
def d_list(*dice):
    list_dice = []
    for d in dice:
        while d > 24:
            d -= 12
            list_dice.append(12)
    if 12 < d:
        list_dice.extend(split_dice(d))
    else:
        list_dice.append(d)
    # check if all dice are supported
    for d in list_dice:
        if d not in KIMERA_DICE:
            st.write(f"WARNING! d_list: d{d} is not a Kimera dice")
    results = [sum(combination) for combination in product(*[range(1, d + 1) for d in list_dice])]
    return sorted(results)

#@st.cache_data()
def d_df(*dice): # returns a dataframe with columns (unique)'result', 'count', and 'fraction'
    # TODO: check dice vs set of supported dice; break up high numbers into supported dice
    list_results = d_list(*dice)
    list_unique_results = list(set(list_results)) # discard duplicates
    list_counts = [list_results.count(result) for result in list_unique_results] # count occurance of each unique result
    total_counts = len(list_results)
    list_fractions = [(list_counts[i] / total_counts) for i in range(len(list_counts))] # calculate fraction of each unique result
    df = pd.DataFrame({'result': list_unique_results, 'count': list_counts, 'fraction': list_fractions})
    return df

#@st.cache_data()
def d_dict(*dice): # ddf: dict where key conserves the dice rolled
    return {str(dice): d_df(*dice)}



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


def attack(df_hit=None, df_dmg=None, df_crit=None, df_armor=None, guard=None, block=None, frame=None, cover=None, distance=0, verbose=False):
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

    #cycle through HIT rolls (minus penalty) and add the corresponding damage tables to dict(s)
    for row in df_hit.iterrows():
        diff = int(row[1]['hit']) # the hit roll currently checked
        critical = bool(crit_threshold < diff)
        hit_fraction = row[1]['fraction']
        #fetch relevant damage table
        if critical:
            hit_df_dmg = df_crit.copy()
        else:
            hit_df_dmg = df_dmg.copy()
        hit_df_dmg['fraction'] *= hit_fraction
        # resolve block for melee attacks
        if melee:
            if diff < 1: # successful block
                protection = block
                hit_df_dmg.loc[hit_df_dmg['result'] <= guard, 'result'] = 0 # superior blocks
                hit_df_dmg.loc[guard < hit_df_dmg['result'], 'result'] -= protection # regular blocks
                dict_block[str(diff)] = hit_df_dmg
        # resolve cover for ranged attacks
        elif (cover is not None) and (diff <= cover):
            hit_df_dmg['result'] -= block
            dict_cover[str(diff)] = hit_df_dmg
        # resolve armor layers: best protection that is struck
        if 0 < diff: # only resolve armor for connected hits
            df_layers_struck = df_armor[(diff <= df_armor.coverage) & (0 < df_armor.coverage)]
            if not df_layers_struck.empty:
                best_protection_row = df_layers_struck.loc[df_layers_struck['protection'].idxmax()]
                protection = best_protection_row['protection']
                best_layer = best_protection_row['layer']
                hit_df_dmg['result'] -= protection
                dict_armor[best_layer] = hit_df_dmg
            else:
                dict_clean[str(diff)] = hit_df_dmg
        dict_hit_dmg[str(diff)] = hit_df_dmg
        if critical:
            dict_crit[str(diff)] = hit_df_dmg
        #st.write(f"diff: {diff}, critical: {critical}, fraction: {hit_fraction}, protection: {protection}, result range: {hit_df_dmg['result'].min()} to {hit_df_dmg['result'].max()}")

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



def defend(df_guard=None, df_tough_block=None, df_dodge=None, ddf_tough=None, hit=None, dmg=None, crit_dmg=None, df_armor=None, df_frame=None, df_cover=None, distance=0, verbose=False):
    '''
    Resolves PC defense (rolled) against an NPC attack (static)
    (params)
    distance (in m): if the distance is greater than 50m, the GM can apply a bonus to the frame value
        this could be as simple as +5 up to 100m, when it beomces +5,
        or as precise as +1 per 10m over 50m.
        As a typical frame is 5, the typical penalty would be exactly 1/10 of the distance: 6 at 60m, 7 at 70m, etc (minimum of 5).
    Returns a dict of DataFrames with columns 'result', 'count', and 'fraction', describing the outcomes of the attack
    '''
    if verbose:
        st.write(f"PC defense {df_guard['result'].min()}-{df_guard['result'].max()}, Block {df_tough_block['result'].min()}-{df_tough_block['result'].max()}, Tough {df_tough['result'].min()}-{df_tough['result'].max()}({int(df_tough['result'].max()/2)}) . . . vs . . . hit {hit}, dmg {dmg}-{crit_dmg}")
    
    dict_sum = {}
    dict_block = {}
    dict_armor = {}
    dict_clean = {}
    dict_crit = {}
    list_d_tough = [int(value) for value in str(list(ddf_tough.keys())[0]).strip('()').split(',') if value]
    df_tough = list(ddf_tough.values())[0]
    # cycle through GUARD rolls and add the corresponding damage tables to dict(s)
    for row in df_guard.iterrows():
        guard = int(row[1]['result'])
        guard_fraction = row[1]['fraction']
        diff = hit-guard
        critical = bool(9 < diff)
        if diff < 0: # successful block
            df_dmg_taken = pd.DataFrame({ # damage taken when blocking
                'count': df_tough_block['count'], # fairly pointless column, but needed for merge_dfs
                'fraction': df_tough_block['fraction'] * guard_fraction,
                'result': dmg - df_tough_block['result']})
            if dmg < guard: #superior block
                df_dmg_taken['result'] = -1
            else:
                st.write(f"blocked! guard: {guard}, diff: {diff}, dmg {dmg}")
            dict_block[str(guard)] = df_dmg_taken
        else: # connected hit
            if critical:
                c_dmg = crit_dmg
            else:
                c_dmg = dmg
            df_dmg_taken = pd.DataFrame({ # damage taken when not blocking
                'count': df_tough['count'], # fairly pointless column, but needed for merge_dfs
                'fraction': df_tough['fraction'] * guard_fraction,
                'result': c_dmg - df_tough['result']})
            # resolve armor layers: best protection that is struck
            df_layers_struck = df_armor[(diff < df_armor.coverage)]
            if not df_layers_struck.empty:
                best_protection_row = df_layers_struck.loc[df_layers_struck['protection'].idxmax()]
                protection = best_protection_row['protection']
                best_layer = best_protection_row['layer']
                # reverseengineer toughness roll with added protection
                list_d = list_d_tough.copy()
                list_d.append(protection)
                df_armor_tough = d_df(*list_d)
                df_dmg_taken = pd.DataFrame({ # damage taken when blocking
                'count': df_armor_tough['count'],
                'fraction': df_armor_tough['fraction'] * guard_fraction,
                'result': dmg - df_armor_tough['result']})
                dict_armor[best_layer] = df_dmg_taken
            else:
                dict_clean[str(guard)] = df_dmg_taken
            if critical:
                dict_crit[str(guard)] = df_dmg_taken
        dict_sum[str(guard)] = df_dmg_taken

    # compile dict of outcomes
    dict_outcome = {'sum': merge_dfs(dict_sum)}
    for key, df in dict_armor.items():
        dict_outcome[key] = df
    if dict_clean:
        dict_outcome['clean'] = merge_dfs(dict_clean)
    if dict_block:
        dict_outcome['block'] = merge_dfs(dict_block)
    if dict_crit:
        dict_outcome['crit'] = merge_dfs(dict_crit)
    return dict_outcome



def colorList(num): # generate a spectrum of colors, first one is white
    first_color = (255, 255, 255)
    list_colors = [first_color] + [
        tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / num, 1, 1))
        for i in range(num - 1)]
    return list_colors



def report(dict_outcome, defend=False):
    '''
    defend flag: active defense is interpreted differently: an active roll has to BEAT the passive; not just equal it
    '''
    fig = go.Figure()  # Plotting
    colors = colorList(len(dict_outcome))

    df = dict_outcome['sum']
    if defend:
        threshold = 4
        no_effect = df[df['result'] < 0]['fraction'].sum()
        staggered = df[(0 <= df['result']) & (df['result'] < threshold)]['fraction'].sum()
        stopped = df[threshold <= df['result']]['fraction'].sum()
        st.write(f"Effect rating {(stopped+(staggered/2))*100:.2f}% - - - Stopped: {stopped*100:.2f}% - Staggered {staggered*100:.2f}% - ({no_effect*100:.2f}% failure)")
    else:
        threshold = 5
        no_effect = df[df['result'] < threshold]['fraction'].sum()
        stopped = df[threshold <= df['result']]['fraction'].sum()
        st.write(f"Effect rating: {stopped*100:.2f}% - - - ({no_effect*100:.2f}% failure)")

    fig.add_shape(go.layout.Shape(type="line", x0=1-defend, x1=1-defend, xref="x", y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"), name='One'))
    fig.add_shape(go.layout.Shape(type="line", x0=threshold, x1=threshold, xref="x", y0=0, y1=1, yref="paper", line=dict(color="blue", width=2, dash="dash"), name='Tough'))

    for index, (key, df) in enumerate(dict_outcome.items()):
        if df is not None:
            if len(df) == 1:
                fig.add_trace(go.Scatter(x=df['result'], y=df['fraction'], mode='markers', name=key, marker=dict(color=f'rgba({colors[index][0]}, {colors[index][1]}, {colors[index][2]}, 0.5)')))
            else:
                line_color = f'rgba({colors[index][0]}, {colors[index][1]}, {colors[index][2]}, 0.5)'
                fig.add_trace(go.Scatter(x=df['result'], y=df['fraction'], mode='lines', name=key, line=dict(color=line_color)))

    st.plotly_chart(fig)



def compare(dict_dfs):
    fig = go.Figure() # Plotting
    colors = colorList(len(dict_dfs))

    for index, (key, df_raw) in enumerate(dict_dfs.items()):
        if df_raw is not None:
            df = df_raw.copy()
            if key.startswith('attack'):
                df['result'] = df['result'] - 1
            fail = df[df['result'] < 4]['fraction'].sum()
            stopped = df[4 <= df['result']]['fraction'].sum()
            st.write(f"{key} outcome: Kill {stopped*100:.2f}% - - - ({fail*100:.2f}% failure), {df['result'].min()} - {df['result'].max()}")

            line_color = f'rgba({colors[index][0]}, {colors[index][1]}, {colors[index][2]}, 0.5)'
            fig.add_trace(go.Scatter(x=df['result'], y=df['fraction'], mode='lines', name=key, line=dict(color=line_color)))

    fig.add_shape(go.layout.Shape(type="line", x0=0, x1=0, xref="x", y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"), name='One'))
    if tough is not None:
        fig.add_shape(go.layout.Shape(type="line", x0=tough, x1=tough, xref="x", y0=0, y1=1, yref="paper", line=dict(color="blue", width=2, dash="dash"), name='Tough'))

    st.plotly_chart(fig) # Show the combined plot



#################### test parameters ####################

verbose = False

# test type
test_attack = True
test_defend = True
test_melee = True
test_dodge = False

# basic parameters
hit = 8
dmg = 8
crit_dmg = dmg + 10
guard = 8
block = 12
tough = 8
frame = 5

# PC Attack test (derived parameters)
df_hit = d_df(hit, hit).rename(columns={'result': 'hit'})
df_dmg = d_df(dmg, dmg)
df_crit = df_dmg.copy()
df_crit['result'] = df_crit['result'] + (crit_dmg-dmg)

# PC Defend test (derived parameters)
df_guard = d_df(guard, guard)
df_tough_block = d_df(8, 8, 8)
df_dodge = d_df(8, 8)
ddf_tough = d_dict(tough)
df_cover = d_df(8)
df_frame = d_df(frame*2)


##################### test sequence #####################

df_test = d_df(8, 8, 8)
df_wrong = d_df(24)

if test_attack:
    if test_melee:
        dict_attack = attack(df_hit=df_hit, df_dmg=df_dmg, df_crit=df_crit, df_armor=df_armor, guard=8, block=12, verbose=verbose)
if test_defend:
    if test_melee:
        dict_defend = defend(df_guard=df_guard, df_tough_block=df_tough_block, df_dodge=df_dodge, ddf_tough=ddf_tough, hit=hit, dmg=dmg, crit_dmg=crit_dmg, df_armor=df_armor, df_frame=df_frame, df_cover=df_cover, distance=0, verbose=verbose)

dict_dfs = {'attack': dict_attack['sum'], 'defend': dict_defend['sum']}
compare(dict_dfs)
report(dict_outcome=dict_attack)
report(dict_outcome=dict_defend, defend=True)