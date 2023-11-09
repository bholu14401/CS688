"""
    First Streamlit App ( https://streamlit.io/gallery )
        streamlit run dev/ex_1.py
        st.write() used to display many data types
"""
import pandas as pd
import streamlit as st
import apps.project_functions as pf



@st.cache
def load_data(path):  # Downloads file & caches it in memory
    dataset = pd.read_csv(path)
    return dataset


with st.sidebar:
    st.subheader('About the Dashboard')
    st.markdown(' Stremlit UI for calling App endpoint.')
    st.header('Example Request')
    query = st.text_input('Query', 'Is there any investment guide for the stock market in India?')
    num_results = st.text_input('Number of Results Returned', 2)
    num_results = int(num_results)
    submitted = st.button("Submitt")


st.title('Streamlit App')  # Or st.write('# First Streamlit App')
st.markdown(""" 
    Stremlit Example Request Dashboard 
     """)


st.header('Returned Response')

if submitted:
    results = pf.get_result(query, num_results)
    st.write(results)


