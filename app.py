import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Anime Recommendation System",
    layout="wide"
)

st.title("🎌 Anime Recommendation System")
st.caption("Data loading test")

# -------------------------
# Load data from GitHub
# -------------------------

@st.cache_data
def load_data():
    anime_url = "https://raw.githubusercontent.com/065010-AmanMalhi/anime-recommendation-system/main/anime.csv"
    synopsis_url = "https://raw.githubusercontent.com/065010-AmanMalhi/anime-recommendation-system/main/anime_with_synopsis.csv"

    anime = pd.read_csv(anime_url)
    synopsis = pd.read_csv(synopsis_url)

    return anime, synopsis


anime, synopsis = load_data()

# -------------------------
# Sanity check
# -------------------------

st.subheader("Dataset check")
st.write("Anime rows:", anime.shape[0])
st.write("Synopsis rows:", synopsis.shape[0])

