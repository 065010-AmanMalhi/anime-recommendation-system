import pandas as pd
import numpy as np
import streamlit as st
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="Anime Recommendation System",
    layout="wide"
)

st.title("🎌 Anime Recommendation System")
st.caption("Story-driven recommendations · Built with analytics, not stereotypes")

GROUP_ID = 81036
SAMPLE_SIZE = 10001

# --------------------------------------------------
# Load + prepare data
# --------------------------------------------------

@st.cache_data
def load_and_prepare_data():
    anime_url = "https://raw.githubusercontent.com/065010-AmanMalhi/anime-recommendation-system/main/anime.csv"
    synopsis_url = "https://raw.githubusercontent.com/065010-AmanMalhi/anime-recommendation-system/main/anime_with_synopsis.csv"

    anime = pd.read_csv(anime_url)
    synopsis = pd.read_csv(synopsis_url)

    anime_df = anime.merge(
        synopsis[['MAL_ID', 'sypnopsis']],
        on='MAL_ID',
        how='inner'
    )

    # Standardize missing values
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
    ].dropna(subset=['sypnopsis'])

    # Episodes → numeric
    anime_df['Episodes'] = pd.to_numeric(
        anime_df['Episodes'],
        errors='coerce'
    ).fillna(1).astype(int)

    # Score → numeric
    anime_df['Score'] = pd.to_numeric(
        anime_df['Score'],
        errors='coerce'
    )

    # Display-friendly fills
    anime_df['Genres'] = anime_df['Genres'].fillna("Unknown Genre")
    anime_df['Studios'] = anime_df['Studios'].fillna("Studio information not available")
    anime_df['Aired'] = anime_df['Aired'].fillna("Release date not available")

    # Final sampling
    anime_df = anime_df.sample(
        n=SAMPLE_SIZE,
        random_state=GROUP_ID
    ).reset_index(drop=True)

    return anime_df


anime_df = load_and_prepare_data()

# --------------------------------------------------
# Text embeddings + similarity
# --------------------------------------------------

@st.cache_resource
def build_similarity(df):
    df['Genres'] = df['Genres'].str.replace(',', ' ', regex=False)
    df['text_features'] = df['Genres'] + ' ' + df['sypnopsis']

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=6000
    )

    tfidf_matrix = tfidf.fit_transform(df['text_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    title_to_index = pd.Series(
        df.index,
        index=df['Name'].str.lower()
    )

    return cosine_sim, title_to_index


cosine_sim, title_to_index = build_similarity(anime_df)

# --------------------------------------------------
# Recommendation functions
# --------------------------------------------------

def recommend_anime(title, top_n=8):
    title = title.lower()

    if title not in title_to_index:
        return pd.DataFrame()

    idx = title_to_index[title]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    indices = [i[0] for i in scores[1:top_n+1]]

    return anime_df.iloc[indices]


def beginner_recommendations(top_n=10):
    beginners = anime_df[
        anime_df['Type'].isin(['TV', 'Movie']) &
        (anime_df['Episodes'] >= 6) &
        (anime_df['Episodes'] <= 50) &
        (~anime_df['Genres'].str.contains('Music', case=False, na=False)) &
        (
            (anime_df['Score'] >= 7.5) |
            (anime_df['Members'] >= anime_df['Members'].quantile(0.75))
        )
    ].copy()

    beginners['score_filled'] = beginners['Score'].fillna(0)
    beginners['popularity_norm'] = beginners['Members'] / beginners['Members'].max()

    beginners['final_score'] = (
        0.6 * beginners['score_filled'] +
        0.4 * beginners['popularity_norm']
    )

    beginners = beginners.sort_values(
        by='final_score',
        ascending=False
    )

    return beginners.head(top_n)

# --------------------------------------------------
# Poster fetching (MAL via Jikan)
# --------------------------------------------------

@st.cache_data
def fetch_anime_image(mal_id):
    try:
        url = f"https://api.jikan.moe/v4/anime/{mal_id}"
        response = requests.get(url, timeout=5).json()
        return response['data']['images']['jpg']['large_image_url']
    except:
        return None

def get_ui_anime_list(df, top_n=200):
    ui_df = df.copy()

    # Keep only proper anime
    ui_df = ui_df[ui_df['Type'].isin(['TV', 'Movie'])]
    ui_df = ui_df[ui_df['Episodes'] >= 6]

    # Remove weird titles
    ui_df = ui_df[
        ui_df['Name'].str.match(
            r'^[A-Za-z0-9\s:;\'\-!,\.]+$',
            na=False
        )
    ]

    # Prefer popular anime
    ui_df = ui_df.sort_values(
        by='Members',
        ascending=False
    )

    return ui_df['Name'].head(top_n).tolist()


# --------------------------------------------------
# UI components
# --------------------------------------------------

def anime_card(row):
    col1, col2 = st.columns([1, 3])

    with col1:
        poster = fetch_anime_image(row['MAL_ID'])
        if poster:
            st.image(poster, use_column_width=True)

    with col2:
        st.markdown(f"### {row['Name']}")
        score = "N/A" if pd.isna(row['Score']) else round(row['Score'], 2)

        st.markdown(f"⭐ **Score:** {score}")
        st.markdown(f"🎭 **Genres:** {row['Genres']}")
        st.markdown(f"🎬 **Studio:** {row['Studios']}")
        st.markdown(f"📺 **Episodes:** {row['Episodes']}")
        st.markdown(f"📅 **Aired:** {row['Aired']}")

        st.markdown(
            f"<p style='color:#cfcfcf'>{row['sypnopsis'][:300]}...</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")

# --------------------------------------------------
# Sidebar + App logic
# --------------------------------------------------

st.sidebar.header("Explore")

mode = st.sidebar.radio(
    "Choose recommendation type",
    ["Similar Anime", "New to Anime"]
)
st.sidebar.subheader("Refine Similar Results")

# Genre filter
all_genres = get_all_genres(anime_df)
selected_genres = st.sidebar.multiselect(
    "Preferred Genres",
    options=all_genres
)

# Rating filter
min_score = st.sidebar.slider(
    "Minimum Rating",
    0.0, 10.0, 6.5, 0.5
)

# Episode range filter
episode_range = st.sidebar.slider(
    "Episode Range",
    1, 100, (6, 50)
)

if mode == "Similar Anime":
    selected_anime = st.selectbox(
        "Select an anime you like",
        ui_anime_list = get_ui_anime_list(anime_df, top_n=200)

selected_anime = st.selectbox(
    "Select an anime you like",
    ui_anime_list
)

    )

    if st.button("Recommend Similar Anime"):
        recs = recommend_anime(selected_anime)

        if recs.empty:
            st.warning("Selected anime not found in dataset.")
        else:
            if st.button("Recommend Similar Anime"):
    recs = recommend_anime(selected_anime)

    if recs.empty:
        st.warning("Selected anime not found in dataset.")
    else:
        # Rating filter
        recs = recs[
            (recs['Score'].isna()) |
            (recs['Score'] >= min_score)
        ]

        # Episode filter
        recs = recs[
            (recs['Episodes'] >= episode_range[0]) &
            (recs['Episodes'] <= episode_range[1])
        ]

        # Genre filter
        if selected_genres:
            recs = recs[
                recs['Genres'].apply(
                    lambda g: any(genre in g for genre in selected_genres)
                )
            ]

        if recs.empty:
            st.info("No anime matched the selected filters.")
        else:
            for _, row in recs.iterrows():
                anime_card(row)


if mode == "New to Anime":
    st.subheader("🌱 Beginner-Friendly Anime")

    if st.button("Show Beginner Recommendations"):
        beginners = beginner_recommendations(10)

        for _, row in beginners.iterrows():
            anime_card(row)

# Add Filters

def get_all_genres(df):
    genres = set()
    for g in df['Genres'].dropna():
        for genre in g.split(','):
            genres.add(genre.strip())
    return sorted(genres)

