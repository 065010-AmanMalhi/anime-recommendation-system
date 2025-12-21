import pandas as pd
import numpy as np
import streamlit as st
import requests
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.markdown("""
<style>
/* Dynamic gradient background */
.stApp {
    background: linear-gradient(
        120deg,
        #0e1117,
        #141824,
        #0e1117
    );
    background-size: 300% 300%;
    animation: gradientShift 12s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Make cards stand out slightly */
div[data-testid="stVerticalBlock"] {
    background-color: rgba(28, 31, 38, 0.75);
    border-radius: 12px;
    padding: 1rem;
}

/* Sidebar polish */
section[data-testid="stSidebar"] {
    background-color: rgba(20, 24, 36, 0.95);
}
</style>
""", unsafe_allow_html=True)

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

@st.cache_data
def load_rating_aggregates():
    ratings_url = "https://raw.githubusercontent.com/065010-AmanMalhi/anime-recommendation-system/main/anime_rating_aggregates.csv"
    return pd.read_csv(ratings_url)
anime_df = load_and_prepare_data()
rating_agg = load_rating_aggregates()

anime_df = anime_df.merge(
    rating_agg,
    on='MAL_ID',
    how='left'
)

anime_df['rating_confidence'] = anime_df['rating_confidence'].fillna(0)
anime_df['rating_count'] = anime_df['rating_count'].fillna(0)
anime_df['avg_rating'] = anime_df['avg_rating'].fillna(anime_df['Score'])


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
    candidates = anime_df[
        anime_df['Type'].isin(['TV', 'Movie']) &
        (anime_df['Episodes'] >= 6) &
        (anime_df['Episodes'] <= 50) &
        (~anime_df['Genres'].str.contains('Music', case=False, na=False))
    ].copy()

    candidates['score_norm'] = candidates['Score'].fillna(0) / 10
    candidates['popularity_norm'] = (
        candidates['Members'] / candidates['Members'].max()
    )

    candidates['community_norm'] = (
        candidates['rating_confidence'] /
        (candidates['rating_confidence'].max() + 1)
    )

    candidates['final_score'] = (
        0.3 * candidates['score_norm'] +
        0.3 * candidates['popularity_norm'] +
        0.4 * candidates['community_norm']
    )

    candidates = candidates.sort_values(
        by='final_score',
        ascending=False
    )

    return candidates.head(top_n)

def popularity_rating_scatter(df, title):

    df = df.copy()

    df["rating_band"] = pd.cut(
        df["Score"],
        bins=[0, 7.5, 8.5, 10],
        labels=["Decent", "Strong", "Elite"]
    )

    fig = px.scatter(
        df,
        x="Members",
        y="Score",
        size="Members",
        color="rating_band",
        color_discrete_map={
            "Elite": "#FFD700",
            "Strong": "#2ECC71",
            "Decent": "#3498DB"
        },
        hover_name="Name",
        title=title,
        template="plotly_dark",
        opacity=0.8
    )

    fig.update_layout(
        xaxis_title="Number of Users",
        yaxis_title="Rating",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)



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

def get_all_genres(df):
    genres = set()
    for g in df['Genres'].dropna():
        for genre in g.split(','):
            genres.add(genre.strip())
    return sorted(genres)

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

        # Community rating signal
        if row.get('rating_count', 0) > 0:
            st.markdown(
                f"👥 **Rated by:** {int(row['rating_count']):,} users"
            )

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
   ["Similar Anime", "Recommend Anime"]

)
st.sidebar.subheader("Refine Similar Results")

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

    ui_anime_list = get_ui_anime_list(anime_df, top_n=200)

    selected_anime = st.selectbox(
        "Select an anime you like",
        ui_anime_list
    )

    if st.button("Recommend Similar Anime"):
        recs = recommend_anime(selected_anime)

        if recs.empty:
            st.info("No anime matched the selected filters.")
        else:
            for _, row in recs.iterrows():
                anime_card(row)

            st.subheader("⚔️ How these recommendations compare")
            popularity_rating_scatter(
                recs,
                "Similar Anime: Popularity vs Rating"
            )


if mode == "Recommend Anime":

    st.subheader("🐉 Beginner-Friendly Anime")

    if st.button("Show Beginner Recommendations"):

        beginners = beginner_recommendations(10)

        for _, row in beginners.iterrows():
            anime_card(row)

        st.subheader("🧠 Why these anime are recommended")
        popularity_rating_scatter(
            beginners,
            "Beginner Picks: Popularity vs Rating"
        )





