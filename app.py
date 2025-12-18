import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Anime Recommendation System",
    layout="wide"
)

st.title("🎌 Anime Recommendation System")
st.caption("Data cleaning & sampling check")

GROUP_ID = 81036
SAMPLE_SIZE = 10001

# -------------------------
# Load data from GitHub
# -------------------------

@st.cache_data
def load_raw_data():
    anime_url = "https://raw.githubusercontent.com/065010-AmanMalhi/anime-recommendation-system/main/anime.csv"
    synopsis_url = "https://raw.githubusercontent.com/065010-AmanMalhi/anime-recommendation-system/main/anime_with_synopsis.csv"

    anime = pd.read_csv(anime_url)
    synopsis = pd.read_csv(synopsis_url)

    return anime, synopsis


# -------------------------
# Clean + sample data
# -------------------------

@st.cache_data
def prepare_anime_df():
    anime, synopsis = load_raw_data()

    # Merge
    anime_df = anime.merge(
        synopsis[['MAL_ID', 'sypnopsis']],
        on='MAL_ID',
        how='inner'
    )

    # Replace "Unknown" strings
    anime_df = anime_df.replace("Unknown", np.nan)

    # Keep required columns
    anime_df = anime_df[
        [
            'MAL_ID',
            'Name',
            'Genres',
            'Studios',
            'Aired',
            'Score',
            'Episodes',
            'Members',
            'Type',
            'sypnopsis'
        ]
    ]

    # Enforce non-null synopsis
    anime_df = anime_df.dropna(subset=['sypnopsis'])

    # Episodes → numeric
    anime_df['Episodes'] = pd.to_numeric(
        anime_df['Episodes'],
        errors='coerce'
    ).fillna(1).astype(int)

    # Genres
    anime_df['Genres'] = anime_df['Genres'].fillna("Unknown Genre")

    # Display-friendly fills
    anime_df['Studios'] = anime_df['Studios'].fillna("Studio information not available")
    anime_df['Aired'] = anime_df['Aired'].fillna("Release date not available")

    # Sampling (FINAL SIZE)
    anime_df = anime_df.sample(
        n=SAMPLE_SIZE,
        random_state=GROUP_ID
    ).reset_index(drop=True)

    return anime_df


anime_df = prepare_anime_df()

# -------------------------
# Sanity checks
# -------------------------

st.subheader("Final dataset status")
st.write("Final shape:", anime_df.shape)

st.subheader("Sample rows")
st.dataframe(anime_df.head(5))
