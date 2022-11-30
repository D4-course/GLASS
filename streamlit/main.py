import streamlit as st
import requests
import asyncio

url = "http://127.0.0.1:8000/"

st.title("Subgraph predictions")
st.subheader("Enter nodes of subgraph : ")
submit = None

with st.form("Form", clear_on_submit=False):
    subgraph = st.text_input("Enter subgraph nodes")
    submit = st.form_submit_button("Submit Values")

if True:
    requests.post(url+"post_subgraph", subgraph)

    if st.button('Load models'):
        d1      = requests.get(url+"load_models")
    if st.button('Get subgraph'):
        data    = requests.get(url+"get_subgraph")
        st.write("Values : ")
        st.write(data.content)
    if st.button('Get density predictions'):
        pe      = requests.get(url+"get_density_prediction")
        st.write("Density predictions :")
        st.write(pe.content)
    if st.button('Get ppi predictions'):
        pe      = requests.get(url+"get_ppi_prediction")
        st.write("PPI predictions :")
        st.write(pe.content)
