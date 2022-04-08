import streamlit as st
import requests
import json

def search(query, top_k):
    params = {'query': query, 'top_k': top_k}
    response = requests.post(
        "http://localhost:5001/search",
        json=params
    )
    try:
        result = response.json()['docs']
    except:
        result = response.json()
    return result

def pretty_print(results):
    if 'detail' in results:
        st.error("Error: {}".format(results['detail'][0]))
    else:
        # Print results in streamlit in a nice format
        for result in results:
            st.markdown("#### Text")
            st.write(result['text'])
            st.metric("Score", result['score'])
            st.write("Document name: ", result['tags']['parent_text'])
            st.write("---")

def interface():
    st.header("Annual Report Intelligent Search")

    query = st.text_input("Introduce a query to search")
    top_k = st.number_input("Number of results to return", value=5)

    if st.button("Search"):
        with st.spinner("Sending request..."):
            results = search(query, top_k)

        st.markdown("## Results")
        pretty_print(results)

interface()