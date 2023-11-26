import streamlit as st

def ini_session():
    if 'verbose' not in st.session_state:
        st.session_state.verbose = False
        st.session_state.verbose = False
        st.session_state.superblock = False

        # test type
        st.session_state.test_attack = True
        st.session_state.test_defend = True
        st.session_state.test_melee = True
        st.session_state.test_dodge = False

        st.session_state.crit_bonus = 10
        st.session_state.frame = 5 # difficulty to hit with ranged attacks, regardless of cover. The smaller the target, the greater the frame value
