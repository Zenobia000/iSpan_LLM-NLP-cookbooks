import streamlit as st
import time
import pandas as pd


# set page title
st.title('Helloooooo Streamlit World!!!')

my_select_box = st.sidebar.selectbox('Select country:', list(['US', 'UK', 'DE', 'FR', 'JP']) )
my_slider = st.sidebar.slider('Temperature C')
st.sidebar.write(f'Temperature F: {my_slider * 1.8 + 32}')

def miles_to_km():
    st.session_state.km = st.session_state.miles * 1.609

def km_to_miles():
    st.session_state.miles = st.session_state.km * 0.621

col1, buff, col2 = st.columns([2, 0.2, 2])

with col1:
    miles = st.number_input('Miles:', key='miles', on_change=miles_to_km)


with col2:
    km = st.number_input('Km:', key='km', on_change=km_to_miles)


if "photo" not in st.session_state:
    st.session_state["photo"] = "not done"


def change_photo_state():
    st.session_state["photo"] = "done"



col1, col2, col3 = st.columns([1, 1, 0.5])  # second column is 2 times larger
with col1:
    df = pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    })
    df

# upload photo
uploaded_photo = col2.file_uploader("Upload a photo:", on_change=change_photo_state)

# take a photo using the camera
camera_photo = col2.camera_input("Take a photo", on_change=change_photo_state)


with col3:
    # display a list
    l1 = [1, 2, 3]
    st.write(l1)

    # display a dict
    l2 = list('abc')
    d1 = dict(zip(l1, l2))
    st.write(d1)

if st.session_state["photo"] == "done":
    # progress bar
    progress_bar = col2.progress(0)
    for i in range(100):
        time.sleep(0.03)
        progress_bar.progress(i)

    col2.success("Photo uploaded succesfully!")

    with st.expander("Click to read more"):
        st.write("Hi, here are more details about the topic you are interested in")

        if uploaded_photo is None:
            st.image(camera_photo)
        else:
            st.image(uploaded_photo)

# Run it: streamlit run .\file.py
