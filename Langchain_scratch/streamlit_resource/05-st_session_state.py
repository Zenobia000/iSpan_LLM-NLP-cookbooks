# SESSION STATE

import streamlit as st

st.title('Streamlit Session')
st.write(st.session_state)

if 'counter' not in st.session_state:
    st.session_state['counter'] = 0
else:
    st.session_state.counter += 1

st.write(f'Counter: {st.session_state.counter}')

button = st.button('Update state')
if 'clicks' not in st.session_state:
    st.session_state['clicks'] = 0

if button:
    st.session_state['clicks'] += 1
    f'After pressing button {st.session_state}'

number = st.slider('Value', 1, 10, key='my_slider')
st.write(st.session_state)
st.write(number)

# Run it: streamlit run .\file.py
