import streamlit as st
import requests
import pandas as pd
import os
from train_search_history import load_search_history, get_related_searches, process_search, train_fasttext_model


GOOGLE_SEARCH_API_KEY = "AIzaSyD30IVyVndnxcAPuGIsVWySTvkEovj_PzM"
SEARCH_ENGINE_ID = "810a40fa74b054e3c"
IMAGE_PATH = "L:\\search_engine\\image.webp"

def get_google_search_results(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_SEARCH_API_KEY}&cx={SEARCH_ENGINE_ID}"
    try:
        response = requests.get(url).json()
        search_results = []
        if "items" in response:
            for item in response["items"][:10]:
                search_results.append(f":--: [{item['title']}]({item['link']})")
        return search_results if search_results else ["No results found."]
    except Exception as e:
        return [f"API Error: {str(e)}"]


st.title("🚀 HFS Search Engine")
st.image(IMAGE_PATH, width=600)


search_history = load_search_history()

search_query = st.text_input("Search here...")


if search_query:
    suggestions = get_related_searches(search_query, search_history, train_fasttext_model())
    if suggestions:
        selected_query = st.selectbox("🔽 Previous Searches:", suggestions, index=0)
        if st.button("Use Selected Query"):
            search_query = selected_query


if st.button("Search"):
    if search_query.strip():
        st.write(f"🔍 Searching for: **{search_query}**")
        results = get_google_search_results(search_query)
        st.subheader("🔗 Search Results:")
        for result in results:
            st.markdown(result)
        process_search(search_query)  
    else:
        st.warning("Please enter a search query.")
