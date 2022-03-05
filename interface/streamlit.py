import streamlit as st
import requests
import json

def search(query):
    params = {'query': query}
    response = requests.post(
        "http://localhost:5001/search",
        json=params
    )
    try:
        result = response.json()['docs']
    except:
        result = response.json()
    return result

def interface():
    st.header("Annual Report Intelligent Search")

    query = st.text_input("Introduce a query to search")

    if query:
        with st.spinner("Sending request..."):
            results = search(query)

        if 'detail' in results:
            st.error("Error: {}".format(results['detail'][0]))
        else:
            st.write(results)

interface()