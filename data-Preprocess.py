import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm_notebook

users_ratings_columns = 'user id | item id | rating | timestamp'.split(' | ')
movies_info_columns = '''movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'''.split(' | ')
users_info_columns = 'user id | age | gender | occupation | zip code'.split(' | ')

##
#ΑΥΤΟ ΜΠΟΡΟΥΜΕ ΝΑ ΤΟ ΑΛΛΑΞΟΥΜΕ ΜΕ ΤΑ path ΤΟΥ ΥΠΟΛΟΓΙΣΤΗ ΜΑΣ ΑΠΛΑ ΓΙΑ ΝΑ ΜΗΝ ΒΑΖΟΥΜΕ ΚΑΘΕ ΦΟΡΑ ΔΙΑΦΟΡΕΤΙΚΑ ΣΤΟ ΤΕΛΟΣ ΤΟ ΑΛΛΑΖΟΥΜΕ
##
users_ratings_url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
movies_info_url  = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.item'
users_info_url    = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.user'
genres_info_url  = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.genre'
ocuppation_info_url    = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.occupation'




##
#ΑΠΟΘΗΚΕΥΟΥΜΕ ΤΑ ΔΕΔΟΜΕΝΑ ΑΠΟ ΤΑ csv ΠΡΟΣΩΡΙΝΑ ΧΩΡΙΖΟΝΤΑΣ ΤΑ ΟΠΩΣ ΑΝΑΦΕΡΑΜΕ ΣΤΗΝ ΑΡΧΗ
##
ratings = pd.read_csv(users_ratings_url, delimiter='\t', header=None, names=users_ratings_columns, encoding='latin-1')

movies = pd.read_csv(movies_info_url, delimiter='|', header=None, names=movies_info_columns, encoding='latin-1')

users = pd.read_csv(users_info_url, delimiter='|', header=None, names=users_info_columns, encoding='latin-1')

genres = pd.read_csv(genres_info_url, delimiter='|', header=None, names=['genre_id', 'genre_code'], encoding='latin-1')
genres = genres[['genre_id']]

ocuppation = pd.read_csv(ocuppation_info_url, delimiter='|', header=None, names=['ocuppation_id'], encoding='latin-1')
ocuppation = ocuppation[['ocuppation_id']]


#join καθε item με το αντιστοιχο movie
ratings_complete = ratings.merge(movies, left_on='item id', right_on='movie id')

ratings_complete = ratings_complete.drop(['item id','timestamp','movie title','release date','video release date','IMDb URL'],axis = 1)

#διαγραφη των ταινιων με βαθμολογια κατω απο 3
ratings_complete.drop(ratings_complete[ratings_complete.rating < 3].index, inplace=True)


ratings_complete = ratings_complete.drop(['rating','movie id'],axis =1)

sum_list = []
for user in tqdm_notebook(ratings_complete['user id'].unique()):
    tmp = ratings_complete.loc[ratings_complete['user id'] == user] #find the columns and rows that contain only the specified user id

    tmplst = []
    tmplst.append(user)#add the user rating genres in a temporary list

    sums = tmp.iloc[:,1:20].sum(axis=0)#find the second to last column with the 1 and 0 of genres and sum all the genres of the user per column
    print(tmplst)#user id
    for sumamount in sums:
        tmplst.append(sumamount)
    sum_list.append(tmplst)
cols = [col for col in ratings_complete.columns]
df = pd.DataFrame(sum_list, columns=cols)
x = df.values[:, 1:] #returns a numpy array
minimax = preprocessing.MinMaxScaler()
x_minimax_scaled = minimax.fit_transform(x.T).T

x_stdscl = preprocessing.scale(x, axis=1)

np.save('processed-data/user-feature-set-orig.npy', x)
np.save('processed-data/user-feature-set-minimax.npy', x_minimax_scaled)
np.save('processed-data/user-feature-set-stdscl.npy', x_stdscl)