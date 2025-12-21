# 🎌 Anime Recommendation System

A content-driven anime recommendation system built using Python and deployed with Streamlit.  
The application provides personalized anime suggestions based on story similarity, popularity, and viewer engagement — designed to challenge the stereotype that anime recommendations are simplistic or genre-only.

---

## 🚀 Live Application
🔗 **Streamlit App:**  
[https://anime-recommendation-system-cmpwcehjxfuxhhhudxuxym.streamlit.app/]

🔗 **GitHub Repository:**  
[https://github.com/065010-AmanMalhi/anime-recommendation-system]

---

## 🧠 Project Objective

The goal of this project is to design and deploy a recommendation system that:
- Goes beyond basic genre filtering
- Uses textual similarity of anime descriptions and themes
- Incorporates popularity and rating confidence
- Delivers recommendations through an interactive, user-friendly interface

The system is especially designed to:
- Help **new viewers** discover beginner-friendly anime
- Allow **existing fans** to find similar anime they might enjoy

---

## 🗂 Dataset Details

- **Source:** Kaggle (https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)
- **Final Sample Size:** 10,001 records  
- **Sampling Method:** Random sampling using Group ID as seed  
- **Group ID Used:** 81036 (used consistently across the project)

### Key Features Used
- Anime name
- Synopsis / description
- Genres
- Studio
- Number of episodes
- Average score
- Number of members (popularity proxy)

---

## ⚙️ Recommendation Approach

### 1️⃣ Content-Based Similarity
- Anime descriptions and genres are analyzed to identify thematic similarity between titles.
- Similarity is computed using **cosine similarity** on text-based representations.

### 2️⃣ Popularity & Rating Confidence
- Recommendations are refined using:
  - Viewer count (number of members)
  - Average ratings
- This helps avoid recommending obscure or low-engagement titles.

### 3️⃣ Beginner-Friendly Logic
For users new to anime, recommendations are filtered based on:
- Moderate episode count
- High viewer engagement
- Strong or consistent ratings
- Popular, well-received titles

---

## 🖥 Application Features

### 🔍 Find Similar Anime
- Select an anime you like
- Get similar recommendations based on story and theme
- Apply filters like rating and episode range
- Visual comparison using interactive charts

### 🌱 New to Anime
- Curated beginner-friendly recommendations
- Easy-to-understand explanations for why each anime is suggested
- Focus on accessibility and engagement

### 🎬 Trailer Links
- Each anime card includes a direct link to its official trailer on YouTube
- Helps users quickly explore before committing to watch

### 🖼 Anime Posters
- Posters fetched dynamically using an external anime API
- Graceful fallback image when posters are unavailable

---

## 📊 Visualization & UI

- Interactive charts built using **Plotly**
- Dark-themed, dynamic background for a modern look
- Hover effects and clean card-based layout
- Designed to feel closer to a real product than a static academic demo

---

## 🛠 Tech Stack

- **Python**
- **Pandas & NumPy** – data handling
- **Scikit-learn** – similarity computation
- **Streamlit** – deployment and UI
- **Plotly** – interactive visualizations
- **External Anime API** – poster retrieval

---

## 📌 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
