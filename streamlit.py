import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics.pairwise import cosine_similarity

from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import CoClustering

import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎶"
)

def load_data():
    count_df = pd.read_csv('count_data.csv')
    song_df = pd.read_csv('song_data.csv')
    count_with_song = pd.merge(count_df,song_df.drop_duplicates(['song_id']),on='song_id', how='left')

    le = LabelEncoder()

    count_with_song['user_id'] = le.fit_transform(count_with_song['user_id'])
    count_with_song['song_id'] = le.fit_transform(count_with_song['song_id'])

    users = count_with_song.user_id
    ratings_count = dict()

    for user in users:
        if user in ratings_count:
            ratings_count[user] += 1
        else:
            ratings_count[user] = 1    

    RATINGS_CUTOFF = 90

    remove_users = []

    for user, num_ratings in ratings_count.items():
        if num_ratings < RATINGS_CUTOFF:
            remove_users.append(user)

    count_with_song = count_with_song.loc[ ~ count_with_song.user_id.isin(remove_users)]

    songs = count_with_song.song_id

    ratings_count = dict()

    for song in songs:
        if song in ratings_count:
            ratings_count[song] += 1
        else:
            ratings_count[song] = 1    

    RATINGS_CUTOFF = 120
    remove_songs = []

    for song, num_ratings in ratings_count.items():
        if num_ratings < RATINGS_CUTOFF:
            remove_songs.append(song)

    df_final= count_with_song.loc[ ~ count_with_song.song_id.isin(remove_songs)]
    df_final.drop(df_final.index[df_final['play_count'] > 5], inplace= True)

    return df_final, song_df


def recommend_by_popular(df_final, song_df):
    average_count = df_final.groupby('song_id')['play_count'].mean()
    play_freq = df_final.groupby('song_id')['play_count'].count()
    final_play = pd.DataFrame({'avg_count':average_count, 'play_freq':play_freq})

    recommendations = final_play[final_play['play_freq'] > 100]
    recommendations = recommendations.sort_values(by = 'avg_count', ascending = False)

    top_song_indices = recommendations.index[:10]
    
    top_song_titles = song_df.loc[top_song_indices, 'title'].tolist()
    
    return top_song_titles
    

def get_recommendations(data, user_id, top_n, algo):

    recommendations = []
    user_item_interactions_matrix = data.pivot_table(index = 'user_id', columns = 'song_id', values = 'play_count')
    non_interacted_songs = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()

    for item_id in non_interacted_songs:
        est = algo.predict(user_id, item_id).est
        recommendations.append((item_id, est))

    recommendations.sort(key = lambda x: x[1], reverse = True)

    return recommendations[:top_n]

def ranking_songs(recommendations, final_rating):
  ranked_songs = final_rating.loc[[items[0] for items in recommendations]].sort_values('play_freq', ascending = False)[['play_freq']].reset_index()
  ranked_songs = ranked_songs.merge(pd.DataFrame(recommendations, columns = ['song_id', 'predicted_ratings']), on = 'song_id', how = 'inner')
  ranked_songs['corrected_ratings'] = ranked_songs['predicted_ratings'] - 1 / np.sqrt(ranked_songs['play_freq'])
  ranked_songs = ranked_songs.sort_values('corrected_ratings', ascending = False)
  
  return ranked_songs

def tokenize(text):
    text = re.sub(r"[^a-zA-Z]"," ", text.lower())
    tokens = word_tokenize(text)
    words = [word for word in tokens if word not in stopwords.words('english')]  # Using stopwords of english
    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]

    return text_lems

def recommendations_content( df_final, title):
    df_features = df_final
    df_features['text'] = df_features['title'] + ' ' + df_features['release'] + ' ' + df_features['artist_name']
    df_features = df_features[['user_id', 'song_id', 'play_count', 'title', 'text']]

    df_features = df_features.drop_duplicates(subset = ['title'])

    df_features = df_features.set_index('title')
    indices = pd.Series(df_features.index)

    nltk.download('omw-1.4')
    tfidf = TfidfVectorizer(tokenizer = tokenize)

    songs_tfidf = tfidf.fit_transform(df_features['text'].values).toarray()

    similar_songs = cosine_similarity(songs_tfidf, songs_tfidf)
    
    recommended_songs = []

    idx = indices[indices == title].index[0]

    score_series = pd.Series(similar_songs[idx]).sort_values(ascending = False)

    top_10_indexes = list(score_series.iloc[1 : 11].index)
    print(top_10_indexes)

    for i in top_10_indexes:
        recommended_songs.append(list(df_features.index)[i])

    return recommended_songs

def main():
    st.title("Music Recommendation system")

    st.sidebar.success("Select a page under!")

    menu = ["Popularity - based recommendation system","User-User similarity-based collaborative filtering","Item-Item similarity-based collaborative filtering",
            "Model based collaborative filtering / Matrix factorization", "Clustering -  based recommendation system", "Content based recommendation system"]
    choice = st.sidebar.selectbox("Recommend by: ",menu)

    df_final, song_df = load_data()

    unique_users_df = df_final.drop_duplicates(subset='user_id')
    unique_titles_df = df_final.drop_duplicates(subset='title')
    
    average_count = df_final.groupby('song_id')['play_count'].mean()

    play_freq = df_final.groupby('song_id')['play_count'].count()

    final_play = pd.DataFrame({'avg_count':average_count, 'play_freq':play_freq})

    reader = Reader(rating_scale= (0,5)) 

    data = Dataset.load_from_df(df_final[['user_id','song_id', 'play_count']], reader) # Taking only "user_id","song_id", and "play_count"

    trainset, testset = train_test_split(data, test_size= 0.4, random_state = 42) # Taking test_size = 0.4

    def get_title(song_id):
        return df_final[df_final['song_id'] == song_id]['title'].values[0]


    if choice == "Popularity - based recommendation system":

        st.subheader("TOP 10 MOST POPULARITY SONGS:")
        
        result_1 = recommend_by_popular(df_final, song_df)
        i=0
        for song in result_1:
            i += 1

            st.write(i," ",song)

    elif choice == "User-User similarity-based collaborative filtering":

        sim_options = {'name': 'pearson_baseline',
               'user_based': True}
        sim_user_user = KNNBasic(sim_options = sim_options, verbose = False, random_state = 1) 
        sim_user_user.fit(trainset)

        st.subheader("Recommend by User similarity-based collaborative filtering")
        option = st.selectbox("Select User ID", unique_users_df['user_id'])

        if st.button("Recommend"):

            search_name = option

            recommendations_user = get_recommendations(df_final,search_name, 10, sim_user_user)
            ranking_2 = ranking_songs(recommendations_user, final_play)
            
            i=0
            for song in ranking_2.iterrows():
                i += 1
                title = get_title(song[1]['song_id'])
                st.write(i, " ", title)
    
    elif choice == "Item-Item similarity-based collaborative filtering":
        
        sim_options = {'name': 'pearson_baseline',
                    'user_based': False}
        sim_item_item = KNNBasic(sim_options = sim_options, verbose = False, random_state = 1) 
        sim_item_item.fit(trainset)

        st.subheader("Recommend by Item similarity-based collaborative filtering")
        option = st.selectbox("Select User ID", unique_users_df['user_id'])

        if st.button("Recommend"):
            search_name = option
            recommendations_item = get_recommendations(df_final,search_name, 10, sim_item_item)
            ranking_3 = ranking_songs(recommendations_item, final_play)

            i=0
            for song in ranking_3.iterrows():
                i += 1
                title = get_title(song[1]['song_id'])
                st.write(i, " ", title)

    elif choice == "Model based collaborative filtering / Matrix factorization":
        
        svd = SVD(n_epochs = 30, lr_all = 0.01, reg_all = 0.2, random_state = 1)
        svd.fit(trainset)

        st.subheader("Recommend by Collaborative filtering / Matrix factorization")
        option = st.selectbox("Select User ID", unique_users_df['user_id'])

        if st.button("Recommend"):
            search_name = option
            svd_recommendations = get_recommendations(df_final, search_name, 10, svd)
            ranking_4 = ranking_songs(svd_recommendations, final_play)

            i=0
            for song in ranking_4.iterrows():
                i += 1
                title = get_title(song[1]['song_id'])
                st.write(i, " ", title)
    
    elif choice == "Clustering -  based recommendation system":

        clust = CoClustering(n_cltr_u = 5,n_cltr_i = 5, n_epochs = 10, random_state = 1)
        clust.fit(trainset)

        st.subheader("Recommend by Clustering -  based recommendation system")
        option = st.selectbox("Select User ID", unique_users_df['user_id'])

        if st.button("Recommend"):

            search_name = option
            clustering_recommendations = get_recommendations(df_final, search_name, 10, clust)
            ranking_5 = ranking_songs(clustering_recommendations, final_play)

            i=0
            for song in ranking_5.iterrows():
                i += 1
                title = get_title(song[1]['song_id'])
                st.write(i, " ", title)

    elif choice == "Content based recommendation system":
        
        clust = CoClustering(n_cltr_u = 5,n_cltr_i = 5, n_epochs = 10, random_state = 1)
        clust.fit(trainset)

        st.subheader("Recommend by Content based recommendation system")
        option = st.selectbox("Select Title Song", unique_titles_df['title'])

        if st.button("Recommend"):
            search_name = option
            result_6 = recommendations_content(df_final,search_name)

            i=0
            for song in result_6:
                i += 1

                st.write(i," ",song)


if __name__ == '__main__':
    main()
