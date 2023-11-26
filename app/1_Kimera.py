import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from itertools import product
import colorsys
from kiutils import *

# Ensure session variables are alive
ini_session()

@st.cache_data()
def getWeapons(file :str) -> pd.DataFrame:
    # TODO: weapons table as .csv
    return pd.read_csv(file)

dice_mappings = {
    14: [8, 6],
    16: [8, 8],
    18: [10, 8],
    20: [10, 10],
    22: [12, 10],
    24: [12, 12]}

# Dice functions

#@st.cache_data()
def int2d(value):
    list_dice = []
    while value > 24: #add a d12 for every 12 over 24
        value -= 12
        list_dice.append(12)
    if value in dice_mappings: #convert the rest to Kimera dice
        list_dice.extend(dice_mappings[value])
    else:
        if value not in {4, 6, 8, 10, 12}:
            st.write(f"WARNING! int2d: d{value} is not a Kimera dice")
        list_dice.extend([value])
    return list_dice

#@st.cache_data()
def d(*dice):
    result = 0
    for d in dice:
        result += np.random.randint(low=1, high=d)
    return result

#@st.cache_data()
def d_list(*dice): # returns a sorted list of all results possible with *dice
    results = [sum(combination) for combination in product(*[range(1, d + 1) for d in dice])]
    return sorted(results)

#@st.cache_data()
def d_df(*dice): # returns a dataframe with columns (unique)'result', 'count', and 'fraction'
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


# convenience functions
def merge_dfs(dict_dfs):
    list_merge = []
    # Iterate dict_dfs and append non-empty DataFrames to list_merge
    for key, df in dict_dfs.items():
        if 'result' in df.columns and not df.empty:
            list_merge.append(df[['result', 'count', 'fraction']])
    # Check if list_merge is not empty before concatenating
    if list_merge:
        # Concatenate the DataFrames in the list
        df_merged = pd.concat(list_merge)
        df_merged = df_merged.groupby('result', as_index=False).sum()
        return df_merged[['result', 'count', 'fraction']]
    else:
        # Return an empty DataFrame as a placeholder
        return pd.DataFrame(columns=['result', 'count', 'fraction'])

def colorList(num): # generate a spectrum of colors, first one is white
    first_color = (255, 255, 255)
    list_colors = [first_color] + [
        tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / num, 1, 1))
        for i in range(num - 1)]
    return list_colors


# Plotting functions
def report(dict_outcome, defend=False):
    '''
    defend flag: active defense is interpreted differently: an active roll has to BEAT the passive; not just equal it
    '''
    fig = go.Figure()  # Plotting
    colors = colorList(len(dict_outcome))
    df = dict_outcome['sum']
    if defend:
        df['result'] = df['result'] + 1 # compensation for damage 0 being a successful attack for defense
        st.write("Active Defense:")
    else:
        st.write("Active Attack:")

    fig.add_shape(go.layout.Shape(type="line", x0=0.5, x1=0.5, xref="x", y0=0, y1=1, yref="paper", line=dict(color="gray", width=2, dash="dash"), name='Stagger'))
    fig.add_shape(go.layout.Shape(type="line", x0=5.5, x1=5.5, xref="x", y0=0, y1=1, yref="paper", line=dict(color="blue", width=2, dash="dash"), name='Out'))

    for index, (key, df) in enumerate(dict_outcome.items()):
        if df is not None:
            if len(df) == 1 or key == 'sum':
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
            if key.startswith('Defend'):
                df['result'] = df['result'] + 1 # compensation for damage 0 being a successful attack for defense
            no_effect = df[df['result'] < 0]['fraction'].sum()
            staggered = df[(0 <= df['result']) & (df['result'] < 4)]['fraction'].sum()
            stopped = df[4 <= df['result']]['fraction'].sum()
            st.write(f"{key} outcome: Out {stopped*100:.2f}% - Stun {staggered*100:.2f}% - Failure {no_effect*100:.2f}%")
            fig.add_trace(go.Scatter(x=df['result'], y=df['fraction'], mode='markers', name=key, marker=dict(color=f'rgba({colors[index][0]}, {colors[index][1]}, {colors[index][2]}, 0.5)')))
            
    fig.add_shape(go.layout.Shape(type="line", x0=0.5, x1=0.5, xref="x", y0=0, y1=1, yref="paper", line=dict(color="gray", width=2, dash="dash"), name='Stagger'))
    fig.add_shape(go.layout.Shape(type="line", x0=5.5, x1=5.5, xref="x", y0=0, y1=1, yref="paper", line=dict(color="blue", width=2, dash="dash"), name='Out'))

    st.plotly_chart(fig) # Show the combined plot



# Attack and Defense functions
#@st.cache_data()
def attack(df_hit=None, df_dmg=None, df_crit=None, df_armor=None, guard=None, block=None, tough=None, frame=None, cover=None, distance=0, verbose=False):
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

    if melee:
        st.write(f"PC melee attack: Hit {df_hit['hit'].min()}-{df_hit['hit'].max()} vs Guard {guard}, dmg {df_dmg['result'].min()}-{df_crit['result'].max()} vs BlockTough {block+tough} and Tough {tough}")
    else:
        st.write(f"PC ranged attack: ")

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
        hit_df_dmg['result'] -= tough
        # resolve block for melee attacks
        if melee:
            if diff < 1: # successful block
                protection = block
                if superblock:
                    hit_df_dmg.loc[hit_df_dmg['result'] <= guard, 'result'] = 0 # superior blocks
                    hit_df_dmg.loc[guard < hit_df_dmg['result'], 'result'] -= protection # regular blocks
                else:
                    hit_df_dmg['result'] -= protection
                dict_block[str(diff)] = hit_df_dmg
                median_dmg = int(hit_df_dmg.loc[hit_df_dmg['fraction'].idxmax()].result)
                str_outcome = f"blocked: {median_dmg} dmg"
        # resolve cover for ranged attacks
        elif (cover is not None) and (diff <= cover):
            hit_df_dmg['result'] -= block
            dict_cover[str(diff)] = hit_df_dmg
        # resolve armor layers: best protection that is struck
        if 0 < diff: # only resolve armor for connected hits
            df_layers_struck = df_armor[((diff <= df_armor.coverage) & (0 < df_armor.coverage)) | (df_armor.coverage == 0)]
            if not df_layers_struck.empty:
                best_protection_row = df_layers_struck.loc[df_layers_struck['protection'].idxmax()]
                best_layer = best_protection_row['layer']
                protection = best_protection_row['protection']
                if 0 < protection:
                    hit_df_dmg['result'] -= protection
                    dict_armor[best_layer] = hit_df_dmg
                    median_dmg = int(hit_df_dmg.loc[hit_df_dmg['fraction'].idxmax()].result)
                    str_outcome = f"struck {best_layer}, protection {protection}: {median_dmg} dmg"
                else:
                    median_dmg = int(hit_df_dmg.loc[hit_df_dmg['fraction'].idxmax()].result)
                    dict_clean[str(diff)] = hit_df_dmg
                    str_outcome = f"clean: {median_dmg} dmg"
            else:
                dict_clean[str(diff)] = hit_df_dmg
                median_dmg = int(hit_df_dmg.loc[hit_df_dmg['fraction'].idxmax()].result)
                str_outcome = f"clean: {median_dmg} dmg"
        dict_hit_dmg[str(diff)] = hit_df_dmg
        if critical:
            dict_crit[str(diff)] = hit_df_dmg
            median_dmg = int(hit_df_dmg.loc[hit_df_dmg['fraction'].idxmax()].result)
            str_outcome = f"{str_outcome} (critical): {median_dmg} dmg"
        if verbose:
            st.write(f"Attack: diff {diff}, {str_outcome}")
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


#@st.cache_data()
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
    # separate ddf_tough into list_d_tough (local tough_dice) and df_tough
    list_d_tough = [int(value) for value in str(list(ddf_tough.keys())[0]).strip('()').split(',') if value]
    df_tough = list(ddf_tough.values())[0]

    st.write(f"PC melee defense: Guard {df_guard['result'].min()}-{df_guard['result'].max()} vs Hit {hit}, BlockTough {df_tough_block['result'].min()}-{df_tough_block['result'].max()} and Tough {df_tough['result'].min()}-{df_tough['result'].max()} vs dmg {dmg}-{crit_dmg}")
    dict_sum = {}
    dict_block = {}
    dict_armor = {}
    dict_clean = {}
    dict_crit = {}
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
            median_dmg = int(df_dmg_taken.loc[df_dmg_taken['fraction'].idxmax()].result)
            str_outcome = f"blocked ({block}): {median_dmg} dmg"
            if superblock:
                if dmg < guard: #superior block
                    df_dmg_taken['result'] = -1
                    str_outcome = "super blocked: -1 dmg"
            dict_block[str(guard)] = df_dmg_taken
        else: # connected hit
            if critical:
                c_dmg = crit_dmg
            else:
                c_dmg = dmg
            df_dmg_taken = pd.DataFrame({ # damage taken when not blocking
                'count': df_tough['count'],
                'fraction': df_tough['fraction'] * guard_fraction,
                'result': c_dmg - df_tough['result']})
            # resolve armor layers: best protection that is struck
            df_layers_struck = df_armor[(diff < df_armor.coverage) | (df_armor.coverage == 0)]
            if not df_layers_struck.empty:
                best_protection_row = df_layers_struck.loc[df_layers_struck['protection'].idxmax()]
                best_layer = best_protection_row['layer']
                protection = best_protection_row['protection']
                # reverseengineer toughness roll with added protection
                if 0 < protection:
                    list_d = list_d_tough.copy()
                    list_d.append(protection*2)
                    df_armor_tough = d_df(*list_d)
                    df_dmg_taken = pd.DataFrame({ # damage taken when blocking
                    'count': df_armor_tough['count'],
                    'fraction': df_armor_tough['fraction'] * guard_fraction,
                    'result': c_dmg - df_armor_tough['result']})
                    dict_armor[best_layer] = df_dmg_taken
                    median_dmg = int(df_dmg_taken.loc[df_dmg_taken['fraction'].idxmax()].result)
                    str_outcome = f"struck {best_layer}, protection {protection}: {median_dmg} dmg"
                else:
                    dict_clean[str(guard)] = df_dmg_taken
                    median_dmg = int(df_dmg_taken.loc[df_dmg_taken['fraction'].idxmax()].result)
                    str_outcome = f"clean: {median_dmg} dmg"
            else:
                dict_clean[str(guard)] = df_dmg_taken
                median_dmg = int(df_dmg_taken.loc[df_dmg_taken['fraction'].idxmax()].result)
                str_outcome = f"clean: {median_dmg} dmg"
            if critical:
                dict_crit[str(guard)] = df_dmg_taken
                median_dmg = int(df_dmg_taken.loc[df_dmg_taken['fraction'].idxmax()].result)
                str_outcome = f"{str_outcome} (critical): {median_dmg} dmg"
        dict_sum[str(guard)] = df_dmg_taken
        if verbose:
            st.write(f"guard {guard} vs {hit}, {str_outcome}")
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



# Sidebar

# Armor defaults
layer1 = {'layer': "armor - Reinforced", 'coverage': 2, 'protection': 0}
layer2 = {'layer': "armor - Light", 'coverage': 6, 'protection': 0}

with st.sidebar:
    slide_hit = st.container()
    slide_dmg = st.container()
    slide_guard = st.container()
    slide_block = st.container()
    slide_tough = st.container()
    hit = slide_hit.slider("Hit", 4, 36, 8, step=2)
    dmg = slide_dmg.slider("Damage", 4, 36, 8, step=2)
    guard = slide_guard.slider("Guard", 0, 36, 8, step=2)
    block = slide_block.slider("Block", 0, 36, 12, step=2)
    tough = slide_tough.slider("Tough", 4, 36, 8, step=2)

    l1 = st.container()
    l2 = st.container()
    l1.subheader(layer1['layer'])
    layer1_coverage = l1.slider("Coverage 1", 0, 20, layer1['coverage'], key="layer1_coverage", step=2)
    layer1_protection = l1.slider("Protection 1", 0, 20, layer1['protection'], key="layer1_protection", step=2)
    l2.subheader(layer2['layer'])
    layer2_coverage = l2.slider("Coverage 2", 0, 20, layer2['coverage'], key="layer2_coverage", step=2)
    layer2_protection = l2.slider("Protection 2", 0, 20, layer2['protection'], key="layer2_protection", step=2)

layer1['coverage'] = layer1_coverage
layer1['protection'] = layer1_protection
layer2['coverage'] = layer2_coverage
layer2['protection'] = layer2_protection
df_armor = pd.DataFrame([layer1, layer2])



#################### test parameters ####################

verbose = st.session_state.verbose
superblock = st.session_state.superblock

# test type
test_attack = st.session_state.test_attack
test_defend = st.session_state.test_defend
test_melee = st.session_state.test_melee
test_dodge = st.session_state.test_dodge

crit_bonus = st.session_state.crit_bonus
frame = st.session_state.frame # difficulty to hit with ranged attacks, regardless of cover. The smaller the target, the greater the frame value

hit_dice = sum([int2d(hit) for _ in range(2)], [])
dmg_dice = sum([int2d(dmg) for _ in range(2)], [])
guard_dice = sum([int2d(guard) for _ in range(2)], [])
tough_dice = sum([int2d(tough) for _ in range(2)], [])
block_dice = sum([int2d(block) for _ in range(2)], []) + tough_dice

# PC Attack test (derived parameters)
df_hit = d_df(*hit_dice).rename(columns={'result': 'hit'})
df_dmg = d_df(*dmg_dice)
df_crit = df_dmg.assign(result=lambda x: x['result'] + (crit_bonus))

# PC Defend test (derived parameters)
df_guard = d_df(*guard_dice)
df_tough_block = d_df(*block_dice)
ddf_tough = d_dict(*tough_dice)
# Not used yet
df_dodge = d_df(*sum([int2d(12) for _ in range(2)], []))
df_cover = d_df(12)
df_frame = d_df(10)
df_standard = d_df(8,8)
df_standard_crit = df_standard.assign(result=lambda x: x['result'] + (crit_bonus))


##################### test sequence #####################
st.write(f"hit {hit}, dmg {dmg}, guard {guard}, block {block}, tough {tough}")
dict_attack = attack(df_hit=df_hit, df_dmg=df_dmg, df_crit=df_crit, df_armor=df_armor, guard=guard, block=block, tough=tough, verbose=verbose)
#dict_Standard_Standard = attack(df_hit=df_standard, df_dmg=df_standard, df_crit=df_standard_crit, df_armor=df_armor, guard=8, block=12, tough=8, verbose=verbose)
#dict_Slider_Standard = attack(df_hit=df_hit, df_dmg=df_dmg, df_crit=df_crit, df_armor=df_armor, guard=8, block=12, tough=8, verbose=verbose)
dict_defend = defend(df_guard=df_guard, df_tough_block=df_tough_block, df_dodge=df_dodge, ddf_tough=ddf_tough, hit=hit, dmg=dmg, crit_dmg=dmg+crit_bonus, df_armor=df_armor, df_frame=df_frame, df_cover=df_cover, distance=0, verbose=verbose)
dict_dfs = {'Attack': dict_attack['sum'], 'Defend': dict_defend['sum']}
#dict_dfs = {'Attack_Standard': dict_Standard_Standard['sum'], 'Attack_Slider': dict_Slider_Standard['sum']}
compare(dict_dfs)
#report(dict_outcome=dict_attack)
#report(dict_outcome=dict_defend, defend=True)
