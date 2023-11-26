import streamlit as st
from kiutils import *

ini_session()

st.header("Inställningar")
st.session_state.verbose = st.checkbox("Verbose", value=st.session_state.verbose)

st.session_state.superblock = st.checkbox("superblock", value=st.session_state.superblock)

# test type
st.session_state.test_attack = st.checkbox("test_attack", value=st.session_state.test_attack)
st.session_state.test_defend = st.checkbox("test_defend", value=st.session_state.test_defend)
st.session_state.test_melee = st.checkbox("test_melee", value=st.session_state.test_melee)
st.session_state.test_dodge = st.checkbox("test_dodge", value=st.session_state.test_dodge)

st.session_state.crit_bonus = st.number_input("Crit bonus", value=st.session_state.crit_bonus)
st.session_state.frame = st.number_input("Frame", value=st.session_state.frame)