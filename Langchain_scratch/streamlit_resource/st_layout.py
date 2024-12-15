# SIDEBAR

import streamlit as st
my_select_box = st.sidebar.selectbox('Select', ['US', 'UK', 'DE', 'FR'])
my_slider = st.sidebar.slider('Temperature')

# COLUMNS
left_column, right_column = st.columns(2)

import random
data = [random.random() for _ in range(100)]

with left_column:
    st.subheader('A linechart')
    st.line_chart(data)

right_column.subheader('Data')
right_column.write(data[:10])

col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

col1.markdown('Hello Streamlit World!!')
col2.write(data[5:10])

with col3:
    st.header('A cat')
    st.image('https://static.streamlit.io/examples/cat.jpg')

# EXPANDER
with st.expander('Click to expand'):
    st.bar_chart({'Data':[random.randint(2, 10) for _ in range(25)]})
    st.write('This is an image of a dog.')
    st.image('https://static.streamlit.io/examples/dog.jpg')

# Run it: streamlit run .\file.py
