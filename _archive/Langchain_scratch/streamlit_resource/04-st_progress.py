# PROGRESS BAR

import streamlit as st
import time
st.write('Starting and extensive computation ...')
latest_iteration = st.empty()

progress_text = 'Operation in progress. Please wait ...'
my_bar = st.progress(0, text=progress_text)
time.sleep(2)

for i in range(100):
    my_bar.progress(i+1)
    latest_iteration.text(f'Iteration {i+1}')
    time.sleep(0.1)

st.write('We are done!! :+1:')

# Run it: streamlit run .\file.py
