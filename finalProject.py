import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from flask import Flask

app = Flask(__name__)


ratings_file = "ml-latest-small/ratings.csv"
df = pd.read_csv(ratings_file)

user_ids = df["userId"].unique().tolist()
u2u_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded = {i: x for i, x in enumerate(user_ids)}
movie_ids = df["movieId"].unique().tolist()
m2m_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded = {i: x for i, x in enumerate(movie_ids)}
df["user"] = df["userId"].map(u2u_encoded)
df["movie"] = df["movieId"].map(m2m_encoded)
df["rating"] = df["rating"].values.astype(np.float32)

num_users = len(u2u_encoded)
num_movies = len(movie_encoded)
min_rating = min(df["rating"])
max_rating = max(df["rating"])



df = df.sample(frac=1, random_state=42)
x = df[["user", "movie"]].values

y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
val_indices = train_indices + int(0.1 * df.shape[0])
x_train, x_val,x_test, y_train, y_val,y_test = (
    x[:train_indices],
    x[train_indices:val_indices],
    x[val_indices:],
    y[:train_indices],
    y[train_indices:val_indices],
    y[val_indices:],
)



class CollaborativeFilteringModel(keras.Model):
    def __init__(self, users, movies, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.users = users
        self.movies = movies
        self.user_embedding = layers.Embedding(
            users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_embedding = layers.Embedding(
            movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_b = layers.Embedding(users, 1)
        self.movie_b = layers.Embedding(movies, 1)

    def call(self, inputs):
        user_v = self.user_embedding(inputs[:, 0])
        user_b = self.user_b(inputs[:, 0])
        movie_v = self.movie_embedding(inputs[:, 1])
        movie_b = self.movie_b(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_v, movie_v, 2)
        
        x = dot_user_movie + user_b + movie_b
        
        return tf.nn.sigmoid(x)


model = CollaborativeFilteringModel(num_users, num_movies, 50)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=3,
    verbose=1,
    validation_data=(x_val, y_val),
)

movie_df = pd.read_csv("ml-latest-small/movies.csv")
df_train = df[:train_indices]


def get_movies(df,user_id):
  movies_watched_by_user = df_train[df_train.userId == user_id]
  movies_not_watched = movie_df[
      ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
  ]["movieId"]
  
  movies_not_watched = list(
      set(movies_not_watched).intersection(set(m2m_encoded.keys()))
  )
  movies_not_watched = [[m2m_encoded.get(x)] for x in movies_not_watched]
  
  user_encoder = u2u_encoded.get(user_id)
  user_movie_array = np.hstack(
      ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
  )
  
  ratings = model.predict(user_movie_array).flatten()
  top_ratings_indices = ratings.argsort()[-10:][::-1]
  recommended_movie_ids = [
      movie_encoded.get(movies_not_watched[x][0]) for x in top_ratings_indices
  ]

  
  recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
  
  return recommended_movies

@app.route('/')
def home():
   return "Welcome to movie recommender"

@app.route('/get_recommendations/<user_id>')
def getmovie(user_id):
   u_id = int(user_id)
   return {"Recommendations":get_movies(df_train,u_id).values.tolist()}

if __name__ == '__main__':
   app.run(host = '0.0.0.0')