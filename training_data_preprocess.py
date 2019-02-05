import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm,tqdm_notebook

users_ratings_columns = 'user id | item id | rating | timestamp'.split(' | ')
movie_info_columns = '''movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'''.split(' | ')
users_info_columns = 'user id | age | gender | occupation | zip code'.split(' | ')

users_ratings_url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
movie_info_url  = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.item'
users_info_url    = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.user'
genre_info_url  = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.genre'
ocp_info_url    = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.occupation'

ratings = pd.read_csv(users_ratings_url, delimiter='\t', header=None, names=users_ratings_columns, encoding='latin-1')

movies = pd.read_csv(movie_info_url, delimiter='|', header=None, names=movie_info_columns, encoding='latin-1')

users = pd.read_csv(users_info_url, delimiter='|', header=None, names=users_info_columns, encoding='latin-1')

genres = pd.read_csv(genre_info_url, delimiter='|', header=None, names=['genre_id', 'genre_code'], encoding='latin-1')
genres = genres[['genre_id']]

occupations = pd.read_csv(ocp_info_url, delimiter='|', header=None, names=['ocp_id'], encoding='latin-1')
occupations = occupations[['ocp_id']]

movies = movies.drop(['video release date', 'IMDb URL', 'movie title'], axis=1)

from datetime import datetime
years = []
movies = movies.dropna() #drop all the rows that at least on element is missing

for ind, x in enumerate(movies['release date']):
    data = datetime.strptime(x,'%d-%b-%Y')
    dates=list(data.timetuple())
    years.append(dates[0])

years_series = pd.Series(years)
movies['release-year'] = years_series.values

movies = movies.drop(['release date'], axis=1)

gender_to_binary = list(map(lambda x:0 if (x=='M') else 1,users['gender'].values))
col = pd.Series(gender_to_binary)
users['gender'] = col.values

occupation_factor = pd.factorize(users['occupation'], sort=False, order=None, na_sentinel=-1, size_hint=None)

users['ocupation'] = list(occupation_factor[0])

users = users.drop(['occupation'], axis = 1)

agelist = []
for val in users.age:
    agelist.append(val // 10) # get only the int before decimal point so ages will be for 1-10
col = pd.Series(agelist)
users['age'] = col.values

ratings = ratings.drop(['timestamp'], axis=1)

tmp = ratings.copy()
tmp['rating'] = tmp['rating'].map(lambda x: int(x >= 3))
tmp.rename(columns={'rating': 'target'}, inplace=True)

ratings_final = tmp.copy()

movies.to_csv('processed-data/BSAS-data/movies.csv', index=False)
users.to_csv('processed-data/BSAS-data/users.csv', index=False)
ratings_final.to_csv('processed-data/BSAS-data/ratings_final.csv', index=False)

df_final = ((users.merge(ratings_final, how='inner', on='user id'))\
.merge(movies, how='inner', left_on='item id', right_on='movie id'))\
.drop(['item id', 'movie id', 'user id'], axis=1)

df_final.to_csv('processed-data/BSAS-data/train.csv', index=False)