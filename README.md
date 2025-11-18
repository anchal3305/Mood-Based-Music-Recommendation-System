🎵 Mood-Based Music Recommendation System

AI-powered music recommender that understands your emotions.

This project predicts the user's mood using text/emotion detection and recommends songs using machine learning (K-Means Clustering) based on audio features such as valence, energy, danceability, etc.

The system suggests Spotify songs that best match the user's emotional state — such as happy, sad, energetic, romantic, calm, angry, and melancholic.

📸 Demo

(Add screenshots here after pushing images)

/demo
   ├── homepage.png <img width="953" height="559" alt="Screenshot 2025-11-18 112215" src="https://github.com/user-attachments/assets/cf33b4e3-929c-47f9-8b70-b29a9833fb31" />
   ├── mood_input.png <img width="953" height="558" alt="Screenshot 2025-11-18 112428" src="https://github.com/user-attachments/assets/2cb3ec55-cd91-4692-9828-48c60a2a4471" />
   └── recommendations.png <img width="953" height="563" alt="Screenshot 2025-11-18 112514" src="https://github.com/user-attachments/assets/3a3721d6-6ccf-4f1a-9c1d-2d82e90af195" />
   └── from buttons (choosing Romantic as the option) <img width="951" height="559" alt="Screenshot 2025-11-18 112706" src="https://github.com/user-attachments/assets/a7e29c2a-74ea-4fc0-a53d-8dc83d756786" />
   <img width="953" height="567" alt="Screenshot 2025-11-18 112753" src="https://github.com/user-attachments/assets/76b21c66-9a91-48ce-8875-527de803d994" />

🚀 Features

✔ Detects user mood from text input (using NLP)

✔ Classifies mood into 7 emotional categories

✔ Recommends songs based on mood using ML

✔ Uses K-Means clustering to group songs into mood clusters

✔ Works with Spotify track URLs

✔ Clean and simple UI (Streamlit / VSC-based app)

✔ Lightweight & fast recommendations

🧠 How the ML Model Works

1. Dataset Preparation
   
The dataset includes Spotify tracks with features:

-valence

-energy

-danceability

-tempo

-liveness

-acousticness

-instrumentalness

-loudness

-speechiness

These audio features reflect the emotion of a track.

3. Feature Scaling
   
StandardScaler is used:

scaler = StandardScaler()

X = scaler.fit_transform(df[feature_cols])

Why?
To ensure all audio features contribute equally in clustering.


4. Choosing K-Means Clustering (k = 7)
   
K-Means groups songs into clusters by similarity in emotional features.

Each cluster represents a mood:

Cluster	Mood

0	Angry

1	Melancholic

2	Energetic

3	Happy

4	Calm

5	Sad

6	Romantic

🎯 Why K-Means?

Because:

It is perfect for unsupervised problems like mood grouping

Works well with continuous numeric data (Spotify audio features)

Very fast and scalable

Easy to interpret and visualize

Great baseline for music segmentation

🛑 Why Not Other Algorithms?

Algorithm	Reason Not Used

SVM	Needs labelled data (we don’t have mood labels for songs)

Random Forest	Supervised only; unsuitable for clustering emotions

KNN	Slow for large datasets and not ideal for natural clusters

DBSCAN	Struggles with high-dimensional feature space

Neural Networks	Overkill for this problem

➡ K-Means was the best fit because our goal is grouping songs by mood, not classification.

🗂 Folder Structure
mood-music-recommendation/
│
├── app.py                   # Frontend UI

├── mood_based_music_recommendation_system (1).py      # Recommendation logic

├── scaler.joblib            # Scaler model

├── kmeans.joblib            # Clustering model

├── id_to_idx.joblib         # ID mapping

├── feature_cols.joblib      # Feature list

├── processed_spotify_tracks.csv

│

├── requirements.txt

└── README.md

🔧 Installation & Running the Project
1️⃣ Clone the Repository

git clone https://github.com/anchal3305/mood-music-recommendation.git

cd mood-music-recommendation

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the App

python app.py

🎤 How Mood Detection Works

User enters text like:

"I feel very sad today 😢"

Keyword detection checks for direct emotional words

If no keyword → Transformer model detects emotion

Emotion is mapped to one of the 7 moods

Recommendations are fetched using centroid-based similarity

🎧 Recommendation Logic

def recommend_by_mood(mood, top_n=10):

    cluster_id = mood_to_cluster[mood]
    
    mood_center = feature_matrix[df['cluster']==cluster_id].mean(axis=0)
    
    return top_n_from_vector(mood_center, top_n)

🛠 Tech Stack

Python

Scikit-learn

Transformers (HuggingFace)

Pandas, NumPy

Streamlit UI

Joblib (for saving models)

Spotify Dataset

📈 Future Improvements

Add face-expression-based mood detection

Add audio-based mood detection

Add genre filters

Create mobile app version

Connect real Spotify API for real-time playback

🤝 Contributing
Pull requests are welcome!
For major changes, open an issue first to discuss.

⭐ Acknowledgements
- Spotify Dataset
  
- Scikit-Learn
  
- HuggingFace Transformers
  
- Streamlit
