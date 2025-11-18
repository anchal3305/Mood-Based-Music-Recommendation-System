import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import random
import warnings
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# CONFIG + CACHING
st.set_page_config(page_title="Mood Music — Premium (Spotify Style)", layout="wide")

@st.cache_data(ttl=3600)
def load_tracks(path: str = "processed_spotify_tracks.csv") -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_safe_classifier():
    # CPU-friendly classifier that usually loads reliably
    try:
        from transformers import pipeline
        model_name = "michellejieli/emotion_text_classifier"
        return pipeline("text-classification", model=model_name, top_k=None, device=-1)
    except Exception:
        return None


# LOAD DATA & ARTIFACTS
df = load_tracks()

# optional artifacts (may not exist)
try:
    scaler = joblib.load("scaler.joblib")
except Exception:
    scaler = None

try:
    feature_cols = joblib.load("feature_cols.joblib")
except Exception:
    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = feature_cols[:12]

feature_matrix = None
if scaler is not None and len(feature_cols) > 0:
    cols_ok = [c for c in feature_cols if c in df.columns]
    if len(cols_ok) > 0:
        feature_matrix = scaler.transform(df[cols_ok])


# MOOD MAPS & KEYWORDS
cluster_to_mood = {
    0: "angry", 1: "melancholic", 2: "energetic",
    3: "happy", 4: "calm", 5: "sad", 6: "romantic"
}
mood_to_cluster = {m: c for c, m in cluster_to_mood.items()}

keyword_map = {
    "energetic": "energetic", "energy": "energetic", "excited": "energetic", "hype": "energetic",
    "relax": "calm", "relaxed": "calm", "relaxing": "calm", "calm": "calm", "chill": "calm", "peace": "calm",
    "romantic": "romantic", "love": "romantic", "date": "romantic",
    "sad": "sad", "cry": "sad", "lonely": "sad", "depressed": "sad",
    "angry": "angry", "mad": "angry", "annoyed": "angry",
    "nostalgia": "melancholic", "nostalgic": "melancholic", "thinking": "melancholic"
}

emotion_map = {
    "joy": "happy", "amusement": "happy", "excitement": "energetic", "enthusiasm": "energetic",
    "calm": "calm", "contentment": "calm", "relief": "calm",
    "sadness": "sad", "grief": "sad",
    "anger": "angry", "annoyance": "angry",
    "fear": "melancholic", "nervousness": "melancholic", "disgust": "sad",
    "love": "romantic"
}

classifier = load_safe_classifier()


# HELPERS
def detect_mood_from_text(text: str) -> str:
    txt = str(text or "").lower().strip()
    if not txt:
        return "neutral"
    # 1) keyword priority
    for k, m in keyword_map.items():
        if k in txt:
            return m
    # 2) classifier fallback
    try:
        if classifier is None:
            return "neutral"
        preds = classifier(txt)[0]
        best = max(preds, key=lambda x: x["score"])
        label = best["label"].lower()
        return emotion_map.get(label, "neutral")
    except Exception:
        return "neutral"

def get_album_image(track_url: str):
    try:
        if not isinstance(track_url, str) or not track_url.strip():
            return None
        r = requests.get(f"https://open.spotify.com/oembed?url={track_url}", timeout=4)
        return r.json().get("thumbnail_url")
    except Exception:
        return None

def recommend_by_mood(mood_label: str, top_n: int = 10, shuffle: bool = False, genre_filter=None, sort_by=None) -> pd.DataFrame:
    if mood_label not in mood_to_cluster:
        return pd.DataFrame()
    cid = mood_to_cluster[mood_label]
    cluster_df = df[df["cluster"] == cid].copy()
    if genre_filter:
        cluster_df = cluster_df[cluster_df["genre"].isin(genre_filter)]
    cluster_df = cluster_df.drop_duplicates(subset=["name", "artist", "track_url"])
    if cluster_df.empty:
        return pd.DataFrame()
    # similarity using provided features (if available)
    if feature_matrix is not None:
        idxs = cluster_df.index.tolist()
        vecs = feature_matrix[idxs]
        center = vecs.mean(axis=0)
        sims = cosine_similarity(center.reshape(1, -1), vecs)[0]
        cluster_df["similarity"] = sims
        cluster_df = cluster_df.sort_values("similarity", ascending=False)
    else:
        cluster_df["similarity"] = 0.0
    if sort_by and sort_by in df.columns:
        cluster_df = cluster_df.sort_values(sort_by, ascending=False)
    candidates = cluster_df.head(max(top_n * 4, 50))
    if shuffle:
        candidates = candidates.sample(frac=1, random_state=random.randint(1, 9999))
    out = candidates.head(top_n).reset_index(drop=True)
    out = out.drop_duplicates(subset=["track_url", "name", "artist"]).reset_index(drop=True)
    return out

def trending_playlists(n=6):
    playlists = {}
    if "popularity" in df.columns:
        top_overall = df.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_url"]).head(n)
        playlists["Trending Now"] = top_overall
        mood_counts = df['cluster'].map(cluster_to_mood).value_counts().head(3).index
        for mood in mood_counts:
            cid = mood_to_cluster.get(mood)
            if cid is None:
                continue
            pl = df[df["cluster"] == cid].sort_values("popularity", ascending=False).drop_duplicates(subset=["track_url"]).head(n)
            playlists[f"Top {mood.title()}"] = pl
    else:
        for m in list(cluster_to_mood.values())[:4]:
            subset = df[df["cluster"].map(cluster_to_mood) == m].drop_duplicates(subset=["track_url"])
            if not subset.empty:
                playlists[f"Curated {m.title()}"] = subset.sample(n=min(n, len(subset)))
    return playlists


# PREMIUM UI CSS (Spotify-like + Sidebar gradient)
st.markdown("""
<style>
/* global */
html, body, [class*="css"] {
  background-color: #0b0b0b !important;
  color: #eaeaea !important;
  font-family: 'Poppins', sans-serif;
}

/* sidebar gradient */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(
        180deg,
        #07130c 0%,
        #0c2418 35%,
        #08301f 70%,
        #06140a 100%
    ) !important;
    padding-top: 22px;
}
[data-testid="stSidebar"] * {
    color: #e8f5e9 !important;
}

/* sidebar logo area */
.sidebar-logo { display:flex; justify-content:center; margin-bottom:18px; }
.sidebar-logo img { width:130px; opacity:0.95; }

/* hero header */
.spotify-hero {
  display:flex; align-items:center; gap:16px;
  padding:20px; border-radius:14px;
  background: linear-gradient(135deg,#0e1112,#0b0b0b);
  border:1px solid rgba(255,255,255,0.03);
  box-shadow: 0 10px 40px rgba(0,0,0,0.55);
  margin-bottom:18px;
}

/* mood buttons */
.mood-btn {
  background:#121212; border:1px solid rgba(255,255,255,0.03);
  color:#eaeaea; padding:12px 18px; border-radius:12px;
  transition: all .12s ease;
}
.mood-btn:hover { background:#1DB954; color:#07120f; transform:translateY(-3px); }

/* spotify-card */
.spotify-card {
  background: linear-gradient(135deg,#0f0f10,#0b0b0b);
  border-radius:12px; padding:16px; margin-bottom:18px;
  border:1px solid rgba(255,255,255,0.03); transition:all .12s;
}
.spotify-card:hover { box-shadow: 0 12px 40px rgba(29,185,84,0.06); transform:translateY(-3px); }
.spotify-small { color:#bdbdbd; font-size:14px; }
</style>
""", unsafe_allow_html=True)


# SIDEBAR (logo + controls)
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <img src="https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_White.png" />
        </div>
        """,
        unsafe_allow_html=True
    )
    st.header("Controls")
    use_buttons = st.checkbox("Use quick mood buttons", value=True)
    allow_text = st.checkbox("Allow typed feeling", value=True)
    st.markdown("---")
    if "genre" in df.columns:
        genres = sorted(df["genre"].dropna().unique().tolist())
        genre_filter = st.multiselect("Filter by genre (optional)", options=genres, default=None)
    else:
        genre_filter = None
    st.markdown("---")
    sort_options = ["similarity"] + [c for c in ["energy", "valence", "tempo", "popularity"] if c in df.columns]
    sort_by = st.selectbox("Sort recommendations by", options=sort_options, index=0)
    view_mode = st.selectbox("View mode", options=["Card view", "Compact view"])
    st.markdown("---")
    more_btn = st.button("Show more songs (shuffle)")


# HEADER / HERO
st.markdown(f"""
<div class="spotify-hero">
    <div>
        <h1 style="margin:0; font-size:32px; letter-spacing:-0.4px;">🎵 Mood-Based Music Recommendation System</h1>
        <div style="color:#bfe4cf; margin-top:6px;">Spotify-style UI • Mood detection • Curated recommendations</div>
    </div>
</div>
""", unsafe_allow_html=True)


# INPUT: buttons row + text input below
selected_mood = None

st.markdown("### Quick moods")
if use_buttons:
    cols = st.columns(7, gap="small")
    moods = ["happy", "sad", "calm", "energetic", "romantic", "angry", "melancholic"]
    emojis = {"happy":"😊","sad":"😔","calm":"😌","energetic":"⚡","romantic":"💘","angry":"😡","melancholic":"🌙"}
    for i, m in enumerate(moods):
        with cols[i]:
            if st.button(f"{emojis[m]}  {m.title()}", key=f"m_{m}"):
                selected_mood = m

st.write("")  # spacing

st.markdown("### Or type your feeling")
typed = st.text_input("Type how you feel (e.g. 'feeling relaxed', 'excited for party')")

if allow_text and st.button("Detect Mood from Text"):
    selected_mood = detect_mood_from_text(typed)

# fallback: auto-detect but do not override button press
if selected_mood is None and allow_text and typed.strip():
    selected_mood = detect_mood_from_text(typed)

if not selected_mood:
    st.info("Choose a mood (buttons) or type and press 'Detect Mood from Text'.")
    st.stop()

st.markdown(f"#### Detected / Selected mood: **{selected_mood.upper()}**")
st.write("")

# RECOMMENDATIONS
TOP_N = 10
results = recommend_by_mood(selected_mood, top_n=TOP_N, shuffle=more_btn, genre_filter=genre_filter,
                            sort_by=(None if sort_by == "similarity" else sort_by))

if results.empty:
    st.warning("No songs found for this mood + filters.")
    st.stop()

if view_mode == "Card view":
    for _, row in results.iterrows():
        track_url = row.get("track_url", "") or ""
        track_id = track_url.split("/")[-1].split("?")[0] if track_url else ""
        img = get_album_image(track_url) or ""
        artist = row.get("artist", "Unknown")
        name = row.get("name", "Unknown")
        mood_label = row.get("mood", selected_mood)

        card_html = f"""
        <div class="spotify-card" style="display:flex; gap:16px; align-items:center;">
            <div style="flex:0 0 140px;">
                <img src="{img}" width="140" style="border-radius:8px; object-fit:cover;" />
            </div>
            <div style="flex:1;">
                <h3 style="margin:0 0 6px 0; font-weight:700;">🎵 {name}</h3>
                <div class="spotify-small" style="margin-bottom:8px;">👤 {artist} • Mood: <b>{mood_label}</b></div>
                {f"<iframe src='https://open.spotify.com/embed/track/{track_id}' width='100%' height='80' frameborder='0' allow='encrypted-media'></iframe>" if track_id else ""}
            </div>
        </div>
        """
        components.html(card_html, height=180, scrolling=False)

else:
    for _, row in results.iterrows():
        track_url = row.get("track_url", "") or ""
        track_id = track_url.split("/")[-1].split("?")[0] if track_url else ""
        st.markdown(f"**{row.get('name','Unknown')}** — *{row.get('artist','Unknown')}*  • Mood: **{row.get('mood',selected_mood)}**")
        if track_id:
            components.html(f"<iframe src='https://open.spotify.com/embed/track/{track_id}' width='100%' height='80' frameborder='0'></iframe>", height=90)


# TRENDING / CURATED
st.markdown("---")
st.markdown("### Trending / Curated Playlists")
pls = trending_playlists(n=4)
cols = st.columns(len(pls))
i = 0
for title, p_df in pls.items():
    with cols[i]:
        st.markdown(f"**{title}**")
        for _, rr in p_df.head(4).iterrows():
            name = rr.get('name', '')
            artist = rr.get('artist', '')
            turl = rr.get('track_url', '')
            thumb = get_album_image(turl) or ''
            small_html = f"""
            <div style='display:flex;gap:10px;align-items:center;margin-bottom:8px;'>
                <img src='{thumb}' width='64' style='border-radius:8px;object-fit:cover;'>
                <div style='flex:1;'>
                    <div style='font-weight:600;color:#e6efe9'>{name}</div>
                    <div style='color:#9fb8a8;font-size:13px'>{artist}</div>
                </div>
            </div>
            """
            st.markdown(small_html, unsafe_allow_html=True)
    i += 1
    if i >= len(cols):
        i = 0

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)