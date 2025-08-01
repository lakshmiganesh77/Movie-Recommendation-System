# 🎬 Movie Recommendation System

A **Python-based movie recommender** that uses **TF-IDF vectorization** and **cosine similarity** to suggest similar movies based on content features.

---

## 📌 Concept

**Cosine similarity** measures the angle between two n-dimensional vectors. It is calculated as:

```
Cosine Similarity = (A ⋅ B) / (||A|| * ||B||)
```

This helps compare how similar two movie feature vectors are.

---

## 🛠️ Tech Stack and Libraries

* `pandas` ** for data handling**
* `numpy` **for numerical computations**
* `TfidfVectorizer` **from** `sklearn` **for text feature extraction**
* `cosine_similarity` **from **`sklearn` **to measure similarity**
* `difflib` **to match input movie names to dataset entries**

---

## 🔍 How it Works

1. **Load and preprocess dataset** (`movies.csv`)
2. **Select relevant features**:

   * `genres`, `keywords`, `tagline`, `cast`, `director`
3. **Clean missing data**
4. **Combine features into a single string per movie**
5. **Vectorize using TF-IDF**
6. **Compute similarity matrix**
7. **Take user input**, find best match, and recommend top 30 similar movies

---

## 🧪 Example Code Snippet

```python
from sklearn.metrics.pairwise import cosine_similarity
vec1 = [1,1,0,1,1]
vec2 = [0,1,0,1,1]
print(cosine_similarity([vec1, vec2]))
```

## 🧠 Consolidated Logic
```python
movie_name = input('Enter your favourite movie name: ')
list_of_all_titles = movies_data['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
print('Movies suggested for you:\n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i < 30:
        print(i, '.', title_from_index)
        i += 1
```

---

## 🧠 Full Program

```python
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read dataset
movies_data = pd.read_csv('movies.csv')
print(movies_data.head())
print(movies_data.shape)

# Select relevant features
selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)
print(movies_data.info())
print(movies_data.isna().sum())
print(movies_data[selected_features].head())
print(movies_data[selected_features].isna().sum())

# Fill nulls
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

print(movies_data.head())

# Combine features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Vectorize
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors.shape)
print(feature_vectors)

# Similarity matrix
similarity = cosine_similarity(feature_vectors)
print(similarity)
print(similarity.shape)

# Input movie
movie_name = input('Enter your favourite movie name: ')
list_of_all_titles = movies_data['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

# Output top 30
print('Movies suggested for you:\n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i < 30:
        print(i, '.', title_from_index)
        i += 1
```

---

## ✅ Output

The program recommends the **top 30 most similar movies** to the one entered by the user based on content similarity.

---

## 📁 Dataset

The model uses a dataset (`movies.csv`) containing fields like:

* Movie title
* Genres
* Keywords
* Tagline
* Cast
* Director

---

## 📌 Note

* No GUI or frontend required.
* Ideal for learning **NLP**, **vectorization**, and **recommendation systems**.
* Can be extended with visualizations (e.g., pie charts, word clouds).
