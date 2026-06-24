# Displaying data on the screen:
# 1. st.write()
# 2. Magic 🙂

import streamlit as st
import pandas as pd

st.title('Hello Streamlit World! :100:')

st.write('We learn Streamlit!')

l1 = [1, 2, 3]
st.write(l1)

l2 = list('abc')
d1 = dict(zip(l1, l2))
st.write(d1)

# using magic
'Displaying using magic :smile:'

df = pd.DataFrame({
    'first_column': [1, 2, 3, 4],
    'second_column': [10, 20, 30, 40]
})

df # st.write(df)

# Run it: streamlit run .\file.py
