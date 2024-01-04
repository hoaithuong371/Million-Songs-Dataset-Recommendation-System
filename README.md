<h1 align='center'><b> Million Songs Data - Recommendation System </b> </h1>

**Thành viên thực hiện:**

*  20280020:  Huỳnh Việt Dũng
*  20280094:  Lê Hoài Thương
*  20280095:  Nguyễn Ngọc Anh Thy

**Giảng viên môn học:** Huỳnh Thanh Sơn

#### Web app by streamlit.io: https://million-songs-dataset-recommendation-system-resys.streamlit.app/

## **Why this is important?**

**The context:** 

With the advent of technology, societies have become more efficient with their lives. At the same time, however, individual human lives have also become more fast-paced and distracted, leaving little time to explore artistic pursuits. Also, technology has made significant advancements in the ability to coexist with art and general entertainment. It has in fact made it easier for humans with a shortage of time to find and consume good content.

Almost every internet-based company's revenue relies on the time consumers spend on it's platform. These companies need to be able to figure out what kind of content is needed in order to increase customer time spent and make their experience better. Therefore, one of the key challenges for these companies is figuring out what kind of content their customers are most likely to consume.

Also most people enjoy listening to music and the challenge of recommending music to a user is easy to understand for a non technical audience.


**The objectives:** 

The objective of this project is **to build a recommendation system to predict the top_n songs for a user based on the likelihood of listening to those songs.**

This project showcases my ability to rapidly learn and develop machine learning solutions while laying the foundation for building a robust production-grade tool that demonstrates the complete end-to-end machine learning process. In the future, I aim to deploy an interactive tool as a demonstration for prospective employers.



## **Data Dictionary**

The core data is the Taste Profile Subset released by the Echo Nest as part of the Million Song Dataset. There are two files in this dataset. The first file contains the details about the song id, titles, release, artist name, and the year of release. The second file contains the user id, song id, and the play count of users.

### song_data

song_id - A unique id given to every song

title - Title of the song

Release - Name of the released album

Artist_name - Name of the artist 

year - Year of release

### count_data

user _id - A unique id given to the user

song_id - A unique id given to the song

play_count - Number of times the song was played

## **Data Source**
http://millionsongdataset.com/

**Why this dataset?**

- It is freely available to the public.
- It is a large enough dataset for the purpose of this project.

## **Approach**

* Load and understand the data

* Data cleaning and feature engineering, some steps taken inclde:

     - I  combined the datasets to create a final dataset for our analysis
     - For easier analysis  I encoded user_id and song_id columns
     - I filtered the data such that the data for analysis contains users who have listened to a good count of songs
     - I also filtered the data for songs that have been listened to a good number of times

* Exploratory Data Analysis

* I built recommendation systems using 6 different algorithms:

     - Rank/Popularity - based recommendation system
     - User-User similarity-based collaborative filtering
     - Item-Item similarity-based collaborative filtering
     - Model based collaborative filtering / Matrix factorization
     - Clustering -  based recommendation system
     - Content based recommendation system


* To demonstrate clustering-based recommendation systems, the surprise library was used.


* Grid search cross-validation was used to tune hyperparameters and improve model perfomance.


* I used RMSE, precision@k, recall@k and  F_1 score to evaluate model perfomance.


* In the future I hope to improve the performance of these models using hyperparameter tuning.

![image](https://github.com/hoaithuong371/Million-Songs-Dataset-Recommendation-System/assets/116288034/0cafbdb6-1f19-431c-a875-4849dc2dc4eb)

![image](https://github.com/hoaithuong371/Million-Songs-Dataset-Recommendation-System/assets/116288034/b3d330c8-09f6-47de-bc7c-a07bc13e21c4)

### Observations and Insights:
The majority of recommendations are of similar artists and songs, it implies that the resulting recommendation system is working well

## **Conclusion and Recommendations:** 

### **Key Observations and Insights:**

*   The **count_df** data has 2,000,000 entries and 3 columns of data types; Unnamed: 0: int64, user_id: object, song_id: object, play_count: int64.


*   The **song_df** data has 1,000,000 entries and 5 columns of data types; song_id: object, title: object, release: object, artist_name: object, year: int64. 

*  The title and release columns have a few missing values. Some of the years are missing.

*   The **df_final** data frame has 117,876 entries with 7 columns and no missing values. The columns and data types are ;  user_id : int64, song_id : int64, play_count : int64, title : object, release : object, artist_name : object and year : int64

*   There are 3,155 unique user_id and 563 unique song_id meaning we could have upto 3155*563 = 1,776,265 interactions but we only have 117,876, this implies, the data is sparse.


*   The user 61472 has interacted with the most songs, 243 times. But still there is a possibility of 3155 - 243 = 2912 more interactions as we have 3155 unique users. For those 2912 remaining users I build recommendation systems to predict which songs are most likely to be played by the user.

*   A **popularity - based recommendation system** is built using play_count, this is helpful where there could be a cold start problem.


*   While comparing the play_count of two songs, it is not only the play_counts that describe the likelihood of the user to that song. Along with the play_count the number of users who have played that song also becomes important to consider. Due to this, I calculated the "corrected_ratings" for each song. Commonly, the higher the "play_count" of a song the more it is liked by users. To interpret the above concept, a song played 4 times by each of 3 people is less liked in comparison to a song played 3 times by ech of 50 people.

*   In the above 'corrected_rating', there is a quantity **1 / np.sqrt(n)** which can be added or subtracted to vary how optimistic predictions are. This can be used to boast a song and vice versa in a production environment.


*   **Model-based Collaborative Filtering** is a **personalized recommendation system**, the recommendations are based on the past behavior of the user and it is not dependent on any additional information. We use **latent features** (features that explain the patterns and relationships) to find recommendations for each user.


*   In the **clustering-based recommendation system**, we explore the similarities and differences in people's tastes in songs based on how they rate different songs. We cluster similar users together and recommend songs to a user based on play_counts from other users in the same cluster.


*   In the **Content Based Recommendation System**, other features ("title", "release", "artist_name") are used to make predictions instead of play_count.

### **Comparison of the various techniques and their relative performance:**



*   In this project, I built recommendation systems using 6 different algorithms

*   Rank/Popularity - based recommendation system can be helpful in the case of cold start problems (when you don't have data points say for a new user to a platform)

*   User-User similarity-based collaborative filtering and Item-Item similarity-based collaborative filtering are used


*   RMSE, precision@k and recall@k, and F1_Score@k are used to evaluate the model performance.

*   I used GridSearchCV to tuning different hyperparameters to improve model performance, in the future I aim to experiment with several hyperparameters to obtain superior model perfomance.


*   For User-User similarity-based collaborative filtering, based on F_1 score, tuning the model improved its performance in comparison to the baseline model. Also the RMSE of the model has gone down as compared to the model before hyperparameter tuning.


*   The optimized model for Item - Item similarity - based collaborative filtering gives an F_1 score of 0.506, this is lower in comparison to the user user similarity-based collaborative filtering (F_1 score = 0.525). Also the predicted play_count for user_id 6958 is lower compared to the user user similarity-based collaborative filtering recommendation system.

 

*   For the Item - Item similarity - based collaborative filtering -  after tuning hyperparameters, the F_1 score of the tuned model is much better than the baseline model. Also, there is a fall in the RMSE value after tuning the hyperparameters. Hence the tuned model is doing better than the baseline model.

*   With  the Model Based Collaborative Filtering - Matrix Factorization, the tuned model shows a slightly better F_1 score than the baseline model, also the RMSE  gones down. Hence the tuned model seems to do better than the baseline model.

 
*   In clustering-based recommendation system, we explore the similarities and differences in people's tastes in songs based on how they rate different songs. We cluster similar users together and recommend songs to a user based on play_counts from other users in the same cluster.



*    In the clustering-based recommendation system, we see that the F_1 score for the tuned co-clustering model is comparable with the F_1 score for the baseline Co-clustering model. The model performance did not improve by much.

*   In the content based recommendation system, majority of our recommendations are of similar artists and songs, it implies that the resulting recommendation system is working well.


### **Proposal for the future solution design and outlook:**


*   A hybrid recommendation system consisting of the explored recommendation systems is proposed so as to suit users and the platform needs. This would give more robust recommendations. The recommendation systems to be used will include;
 
    - Rank/Popularity - based recommendation system
    - User-User similarity-based collaborative filtering
    - Item-Item similarity-based collaborative filtering
    - Model based collaborative filtering / Matrix factorization
    - Clustering -  based recommendation system
    - Content based recommendation system

*   The popularity - based recommendation system would be helpful in the case of cold start problems.

*   The model-based collaborative filtering/ matrix factorization is advantageous as it can handle sparse data since it can predict missing ratings by estimating the values based on the learned latent factors. It can also capture complex patterns and relationships in user-item interactions, making it suitable for providing personalized recommendations.

*   The content based recommendation system, uses other features ( "title", "release", "artist_name") instead of play_count, with additional text data this will be helpful


*   Future hyperparameter tuning is required to improve model performance

*   In hope to build and deploy a production grade tool that show case the entire machine learning process, I will periodically re train the models with new data

*   Some  tools to further explore in the future include: 
    - Github Actions - to run notebooks
    - Neptune.ai - experiment tracking
    - Hopsworks.ai - feature Store and Model Registry

