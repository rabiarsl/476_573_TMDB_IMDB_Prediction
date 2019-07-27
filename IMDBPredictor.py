import numpy as np
import pandas as pd
import ast
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer

# Read train and test data
train = pd.read_csv('train_with_imdb.csv', index_col=0)
test = pd.read_csv('test_with_imdb.csv', index_col=0)

# Fix some values
train.loc[train['id'] == 90,'budget'] = 30000000
train.loc[train['id'] == 118,'budget'] = 60000000
train.loc[train['id'] == 149,'budget'] = 18000000
train.loc[train['id'] == 464,'budget'] = 20000000
train.loc[train['id'] == 470,'budget'] = 13000000
train.loc[train['id'] == 513,'budget'] = 930000
train.loc[train['id'] == 797,'budget'] = 8000000
train.loc[train['id'] == 819,'budget'] = 90000000
train.loc[train['id'] == 850,'budget'] = 90000000
train.loc[train['id'] == 1007,'budget'] = 2
train.loc[train['id'] == 1112,'budget'] = 7500000
train.loc[train['id'] == 1131,'budget'] = 4300000
train.loc[train['id'] == 1359,'budget'] = 10000000
train.loc[train['id'] == 1542,'budget'] = 1
train.loc[train['id'] == 1570,'budget'] = 15800000
train.loc[train['id'] == 1571,'budget'] = 4000000
train.loc[train['id'] == 1714,'budget'] = 46000000
train.loc[train['id'] == 1721,'budget'] = 17500000
train.loc[train['id'] == 1885,'budget'] = 12
train.loc[train['id'] == 2091,'budget'] = 10
train.loc[train['id'] == 2268,'budget'] = 17500000
train.loc[train['id'] == 2491,'budget'] = 6
train.loc[train['id'] == 2602,'budget'] = 31000000
train.loc[train['id'] == 2612,'budget'] = 15000000
train.loc[train['id'] == 2696,'budget'] = 10000000
train.loc[train['id'] == 2801,'budget'] = 10000000
train.loc[train['id'] == 335,'budget'] = 2
train.loc[train['id'] == 348,'budget'] = 12
train.loc[train['id'] == 470,'budget'] = 13000000
train.loc[train['id'] == 513,'budget'] = 1100000
train.loc[train['id'] == 640,'budget'] = 6
train.loc[train['id'] == 696,'budget'] = 1
train.loc[train['id'] == 797,'budget'] = 8000000
train.loc[train['id'] == 850,'budget'] = 1500000
train.loc[train['id'] == 1199,'budget'] = 5
train.loc[train['id'] == 1282,'budget'] = 9
train.loc[train['id'] == 1347,'budget'] = 1
train.loc[train['id'] == 1755,'budget'] = 2
train.loc[train['id'] == 1801,'budget'] = 5
train.loc[train['id'] == 1918,'budget'] = 592
train.loc[train['id'] == 2033,'budget'] = 4
train.loc[train['id'] == 2118,'budget'] = 344
train.loc[train['id'] == 2252,'budget'] = 130
train.loc[train['id'] == 2256,'budget'] = 1
train.loc[train['id'] == 2696,'budget'] = 10000000
test.loc[test['id'] == 3033,'budget'] = 250
test.loc[test['id'] == 3051,'budget'] = 50
test.loc[test['id'] == 3084,'budget'] = 337
test.loc[test['id'] == 3224,'budget'] = 4
test.loc[test['id'] == 3594,'budget'] = 25
test.loc[test['id'] == 3619,'budget'] = 500
test.loc[test['id'] == 3831,'budget'] = 3
test.loc[test['id'] == 3935,'budget'] = 500
test.loc[test['id'] == 4049,'budget'] = 995946
test.loc[test['id'] == 4424,'budget'] = 3
test.loc[test['id'] == 4460,'budget'] = 8
test.loc[test['id'] == 4555,'budget'] = 1200000
test.loc[test['id'] == 4624,'budget'] = 30
test.loc[test['id'] == 4645,'budget'] = 500
test.loc[test['id'] == 4709,'budget'] = 450
test.loc[test['id'] == 4839,'budget'] = 7
test.loc[test['id'] == 3125,'budget'] = 25
test.loc[test['id'] == 3142,'budget'] = 1
test.loc[test['id'] == 3201,'budget'] = 450
test.loc[test['id'] == 3222,'budget'] = 6
test.loc[test['id'] == 3545,'budget'] = 38
test.loc[test['id'] == 3670,'budget'] = 18
test.loc[test['id'] == 3792,'budget'] = 19
test.loc[test['id'] == 3881,'budget'] = 7
test.loc[test['id'] == 3969,'budget'] = 400
test.loc[test['id'] == 4196,'budget'] = 6
test.loc[test['id'] == 4221,'budget'] = 11
test.loc[test['id'] == 4222,'budget'] = 500
test.loc[test['id'] == 4285,'budget'] = 11
test.loc[test['id'] == 4319,'budget'] = 1
test.loc[test['id'] == 4639,'budget'] = 10
test.loc[test['id'] == 4719,'budget'] = 45
test.loc[test['id'] == 4822,'budget'] = 22
test.loc[test['id'] == 4829,'budget'] = 20
test.loc[test['id'] == 4969,'budget'] = 20
test.loc[test['id'] == 5021,'budget'] = 40
test.loc[test['id'] == 5035,'budget'] = 1
test.loc[test['id'] == 5063,'budget'] = 14
test.loc[test['id'] == 5119,'budget'] = 2
test.loc[test['id'] == 5214,'budget'] = 30
test.loc[test['id'] == 5221,'budget'] = 50
test.loc[test['id'] == 4903,'budget'] = 15
test.loc[test['id'] == 4983,'budget'] = 3
test.loc[test['id'] == 5102,'budget'] = 28
test.loc[test['id'] == 5217,'budget'] = 75
test.loc[test['id'] == 5224,'budget'] = 3
test.loc[test['id'] == 5469,'budget'] = 20
test.loc[test['id'] == 5840,'budget'] = 1
test.loc[test['id'] == 5960,'budget'] = 30
test.loc[test['id'] == 6506,'budget'] = 11
test.loc[test['id'] == 6553,'budget'] = 280
test.loc[test['id'] == 6561,'budget'] = 7
test.loc[test['id'] == 6582,'budget'] = 218
test.loc[test['id'] == 6638,'budget'] = 5
test.loc[test['id'] == 6749,'budget'] = 8
test.loc[test['id'] == 6759,'budget'] = 50
test.loc[test['id'] == 6856,'budget'] = 10
test.loc[test['id'] == 6858,'budget'] = 100
test.loc[test['id'] == 6876,'budget'] = 250
test.loc[test['id'] == 6972,'budget'] = 1
test.loc[test['id'] == 7079,'budget'] = 8000000
test.loc[test['id'] == 7150,'budget'] = 118
test.loc[test['id'] == 6506,'budget'] = 118
test.loc[test['id'] == 7225,'budget'] = 6
test.loc[test['id'] == 7231,'budget'] = 85
test.loc[test['id'] == 5222,'budget'] = 5
test.loc[test['id'] == 5322,'budget'] = 90
test.loc[test['id'] == 5350,'budget'] = 70
test.loc[test['id'] == 5378,'budget'] = 10
test.loc[test['id'] == 5545,'budget'] = 80
test.loc[test['id'] == 5810,'budget'] = 8
test.loc[test['id'] == 5926,'budget'] = 300
test.loc[test['id'] == 5927,'budget'] = 4
test.loc[test['id'] == 5986,'budget'] = 1
test.loc[test['id'] == 6053,'budget'] = 20
test.loc[test['id'] == 6104,'budget'] = 1
test.loc[test['id'] == 6130,'budget'] = 30
test.loc[test['id'] == 6301,'budget'] = 150
test.loc[test['id'] == 6276,'budget'] = 100
test.loc[test['id'] == 6473,'budget'] = 100
test.loc[test['id'] == 6842,'budget'] = 30

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

# Dictionary columns of data
def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
    return df

train = text_to_dict(train)
test = text_to_dict(test)

# Convert columns to list
list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_countries = list(train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_languages = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_cast_genders = list(train['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)
list_of_cast_characters = list(train['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)
list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_crew_jobs = list(train['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)
list_of_crew_genders = list(train['crew'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)
list_of_crew_departments = list(train['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)

# There is a column named belongs_to_collection we take collection name.
# If there isn't any collection, has_collection has_collection = 0.
# Normal format: [{'id': 313576, 'name': 'Hot Tub Time Machine...}]
train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)
test['collection_name'] = test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
test['has_collection'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

# We took collection information, so drop this column
train = train.drop(['belongs_to_collection'], axis=1)
test = test.drop(['belongs_to_collection'], axis=1)

# Holds genre number and genre names
# Normal format: [{'id': 35, 'name': 'Comedy'}]
train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)
train['all_genres'] = train['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]
for g in top_genres:
    train['genre_' + g] = train['all_genres'].apply(lambda x: 1 if g in x else 0)

test['num_genres'] = test['genres'].apply(lambda x: len(x) if x != {} else 0)
test['all_genres'] = test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_genres:
    test['genre_' + g] = test['all_genres'].apply(lambda x: 1 if g in x else 0)

train = train.drop(['genres'], axis=1)
test = test.drop(['genres'], axis=1)

# Holds number of production companies and their names.
# Normal format: [{'name': 'Bold Films', 'id': 2266}, {'name': ...]
train['num_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)
train['all_production_companies'] = train['production_companies'].apply(
    lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(15)]
for g in top_companies:
    train['production_company_' + g] = train['all_production_companies'].apply(lambda x: 1 if g in x else 0)

test['num_companies'] = test['production_companies'].apply(lambda x: len(x) if x != {} else 0)
test['all_production_companies'] = test['production_companies'].apply(
    lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_companies:
    test['production_company_' + g] = test['all_production_companies'].apply(lambda x: 1 if g in x else 0)

train = train.drop(['production_companies', 'all_production_companies'], axis=1)
test = test.drop(['production_companies', 'all_production_companies'], axis=1)

# Holds number of production countries and their names.
# Normal format: [{'iso_3166_1': 'US', 'name': 'United States o...]
train['num_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)
train['all_countries'] = train['production_countries'].apply(
    lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(15)]
for g in top_countries:
    train['production_country_' + g] = train['all_countries'].apply(lambda x: 1 if g in x else 0)

test['num_countries'] = test['production_countries'].apply(lambda x: len(x) if x != {} else 0)
test['all_countries'] = test['production_countries'].apply(
    lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_countries:
    test['production_country_' + g] = test['all_countries'].apply(lambda x: 1 if g in x else 0)

train = train.drop(['production_countries', 'all_countries'], axis=1)
test = test.drop(['production_countries', 'all_countries'], axis=1)

# Holds number of spoken languages and their names.
# Normal format: [{'iso_639_1': 'en', 'name': 'English'}]
train['num_languages'] = train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)
train['all_languages'] = train['spoken_languages'].apply(
    lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(15)]
for g in top_languages:
    train['language_' + g] = train['all_languages'].apply(lambda x: 1 if g in x else 0)

test['num_languages'] = test['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)
test['all_languages'] = test['spoken_languages'].apply(
    lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_languages:
    test['language_' + g] = test['all_languages'].apply(lambda x: 1 if g in x else 0)

train = train.drop(['spoken_languages', 'all_languages'], axis=1)
test = test.drop(['spoken_languages', 'all_languages'], axis=1)

# Holds number of keywords and their names.
# Normal format: [{'id': 4379, 'name': 'time travel'}, {'id': 9...]
train['num_Keywords'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)
train['all_Keywords'] = train['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]
for g in top_keywords:
    train['keyword_' + g] = train['all_Keywords'].apply(lambda x: 1 if g in x else 0)

test['num_Keywords'] = test['Keywords'].apply(lambda x: len(x) if x != {} else 0)
test['all_Keywords'] = test['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_keywords:
    test['keyword_' + g] = test['all_Keywords'].apply(lambda x: 1 if g in x else 0)

train = train.drop(['Keywords', 'all_Keywords'], axis=1)
test = test.drop(['Keywords', 'all_Keywords'], axis=1)

# Holds number of number of cast and their names.
# Normal format: [{'cast_id': 4, 'character': 'Lou', 'credit_id': '52fe4ee7c3a36847f82afae7', 'gender': 2,
#  'id': 52997, 'name': 'Rob Corddry', 'order': 0, 'profile_path': '/k2zJL0V1nEZuFT08xUdOd3ucfXz.jpg'}]
# 0-> unspecified 1-> woman 2-> man
train['num_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)
top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]
for g in top_cast_names:
    train['cast_name_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)
train['genders_1_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
train['genders_2_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]
for g in top_cast_characters:
    train['cast_character_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)

test['num_cast'] = test['cast'].apply(lambda x: len(x) if x != {} else 0)
for g in top_cast_names:
    test['cast_name_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)
test['genders_1_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
test['genders_2_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
for g in top_cast_characters:
    test['cast_character_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)

train = train.drop(['cast'], axis=1)
test = test.drop(['cast'], axis=1)

# Holds number of number of crew and their names.
# Normal format: [{'cast_id': 4, 'character': 'Lou', 'credit_id': '52fe4ee7c3a36847f82afae7', 'gender': 2,
#  'id': 52997, 'name': 'Rob Corddry', 'order': 0, 'profile_path': '/k2zJL0V1nEZuFT08xUdOd3ucfXz.jpg'}]
# 0-> unspecified 1-> woman 2-> man
train['num_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)
top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]
for g in top_crew_names:
    train['crew_name_' + g] = train['crew'].apply(lambda x: 1 if g in str(x) else 0)

top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)]
for j in top_crew_jobs:
    train['jobs_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))
top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(15)]
for j in top_crew_departments:
    train['departments_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j]))

test['num_crew'] = test['crew'].apply(lambda x: len(x) if x != {} else 0)
for g in top_crew_names:
    test['crew_name_' + g] = test['crew'].apply(lambda x: 1 if g in str(x) else 0)

for j in top_crew_jobs:
    test['jobs_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))
for j in top_crew_departments:
    test['departments_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j]))

train = train.drop(['crew'], axis=1)
test = test.drop(['crew'], axis=1)

# Taking IMDB score as double
train['imdb_score'] = train['imdb_scores'].apply(lambda x: np.double(x.split('/')[0]))
test['imdb_score'] = test['imdb_scores'].apply(lambda x: np.double(x.split('/')[0]))

train = train.drop(['imdb_scores'], axis=1)
test = test.drop(['imdb_scores'], axis=1)

# Since there are very large and very small values, the approximation process
train['log_budget'] = np.log1p(train['budget'])
test['log_budget'] = np.log1p(test['budget'])

# Setting has home page or not. If there exist any home_page link 1, else 0
train['has_homepage'] = 0
train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1
test['has_homepage'] = 0
test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1

# If release date is null, give a constant value.
test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/98'
def fix_date(x):
    year = x.split('/')[2]
    if int(year) <= 19:
        return x[:-2] + '20' + year
    else:
        return x[:-2] + '19' + year


# Converting date to exact format
train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))
test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))
train['release_date'] = pd.to_datetime(train['release_date'])
test['release_date'] = pd.to_datetime(test['release_date'])


def process_date(df):
    date_parts = ["year", "month", 'weekofyear', 'quarter']
    for part in date_parts:
        part_col = 'release_date' + "_" + part
        df[part_col] = getattr(df['release_date'].dt, part).astype(int)

    return df


train = process_date(train)
test = process_date(test)

train = train.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status'], axis=1)
test = test.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status'], axis=1)
train = train.drop(['revenue'], axis=1)

train = train.loc[:,~train.columns.duplicated()]
test = test.loc[:,~test.columns.duplicated()]


# Encodes string values to integer
for col in ['original_language', 'collection_name', 'all_genres']:
    le = preprocessing.LabelEncoder()
    le.fit(list(train[col].fillna('')) + list(test[col].fillna('')))
    train[col] = le.transform(train[col].fillna('').astype(str))
    test[col] = le.transform(test[col].fillna('').astype(str))

train_texts = train[['title', 'tagline', 'overview', 'original_title']]
test_texts = test[['title', 'tagline', 'overview', 'original_title']]

# Counts word number in in given columns
for col in ['title', 'tagline', 'overview', 'original_title']:
    train = train.drop(col, axis=1)
    test = test.drop(col, axis=1)

X = train.drop(['id', 'imdb_score'], axis=1)
y = train['imdb_score']
X_test = test.drop(['id', 'imdb_score'], axis=1)
y_test = test['imdb_score']

# Using TFIDF to score and weight words
for col in train_texts.columns:
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 analyzer='word',
                                 token_pattern=r'\w{1,}',
                                 ngram_range=(1, 2),
                                 min_df=10)

    vectorizer.fit(list(train_texts[col].fillna('')) + list(test_texts[col].fillna('')))
    train_col_text = vectorizer.transform(train_texts[col].fillna(''))
    test_col_text = vectorizer.transform(test_texts[col].fillna(''))
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=2000)
    model = linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=folds)

    scores = []
    oof = np.zeros(train_col_text.shape[0])
    prediction = np.zeros(test_col_text.shape[0])
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_col_text)):
        X_train, X_valid = train_col_text[train_index], train_col_text[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        model.fit(X_train, y_train)
        y_pred_valid = model.predict(X_valid).reshape(-1, )
        score = mean_squared_error(y_valid, y_pred_valid)
        y_pred = model.predict(test_col_text)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)
        prediction += y_pred
    prediction /= n_fold

    X[col + '_oof'] = oof
    X_test[col + '_oof'] = prediction


trainAdditionalFeatures = pd.read_csv('TrainAdditionalFeatures.csv')
testAdditionalFeatures = pd.read_csv('TestAdditionalFeatures.csv')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X['imdb_id'] = train['imdb_id']
X_test['imdb_id'] = test['imdb_id']
del train, test

X = pd.merge(X, trainAdditionalFeatures, how='left', on=['imdb_id'])
X_test = pd.merge(X_test, testAdditionalFeatures, how='left', on=['imdb_id'])

X = X.drop(['imdb_id'], axis=1)
X_test = X_test.drop(['imdb_id'], axis=1)


def add_extra_features(df):
    df['budget_to_runtime'] = df['budget'] / df['runtime']
    df['runtime_to_mean_year'] = df['runtime'] / df.groupby("release_date_year")["runtime"].transform('mean')
    df['popularity_to_mean_year'] = df['popularity'] / df.groupby("release_date_year")["popularity"].transform('mean')
    df['budget_to_mean_year'] = df['budget'] / df.groupby("release_date_year")["budget"].transform('mean')
    df['log_budget_to_year'] = df['log_budget'] / df['release_date_year']
    df['budget_to_mean_year_to_year'] = df['budget_to_mean_year'] / df['release_date_year']
    df['popularity_to_mean_year_to_log_budget'] = df['popularity_to_mean_year'] / df['log_budget']
    df['year_to_log_budget'] = df['release_date_year'] / df['log_budget']
    df['budget_to_runtime_to_year'] = df['budget_to_runtime'] / df['release_date_year']
    df['genders_1_cast_to_log_budget'] = df['genders_1_cast'] / df['log_budget']
    df['all_genres_to_popularity_to_mean_year'] = df['all_genres'] / df['popularity_to_mean_year']
    df['weightedRating'] = (df['rating'] * df['totalVotes'] + 6.367 * 1000) / (df['totalVotes'] + 1000)
    df['_popularity_totalVotes_ratio'] = df['totalVotes'] / df['popularity']
    df['_rating_popularity_ratio'] = df['rating'] / df['popularity']
    df['_rating_totalVotes_ratio'] = df['totalVotes'] / df['rating']
    df['_totalVotes_releaseYear_ratio'] = df['totalVotes'] / df['release_date_year']
    df['_budget_rating_ratio'] = df['budget'] / df['rating']
    df['_runtime_rating_ratio'] = df['runtime'] / df['rating']
    df['_budget_totalVotes_ratio'] = df['budget'] / df['totalVotes']
    df['meantotalVotesByYear'] = df.groupby("release_date_year")["totalVotes"].transform('mean')
    df['meanTotalVotesByRating'] = df.groupby("rating")["totalVotes"].transform('mean')
    df.fillna(value=0.0, inplace=True)
    return df


X = add_extra_features(X)
X_test = add_extra_features(X_test)

X = X.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

random_seed = 2000
import xgboost as xgb

def xgb_model(trn_x, trn_y, val_x, val_y, test):
    model = xgb.XGBRegressor(max_depth=5,
                            learning_rate=0.01,
                            n_estimators=10000,
                            objective='reg:linear',
                            seed=random_seed,
                            silent=True,
                            subsample=0.8,
                            colsample_bytree=0.7,
                            colsample_bylevel=0.5)

    model.fit(trn_x, trn_y)
    val_pred = model.predict(val_x)
    test_pred = model.predict(test)
    rmse = np.sqrt(mean_squared_error(np.log1p(val_pred), np.log1p(val_y))) * 10

    return {'val':val_pred, 'test':test_pred, 'error':rmse}


import lightgbm as lgb


def lgb_model(trn_x, trn_y, val_x, val_y, test):
    model = lgb.LGBMRegressor(n_estimators=10000,
                              objective='regression',
                              metric='rmse',
                              max_depth=5,
                              num_leaves=30,
                              min_child_samples=50,
                              learning_rate=0.01,
                              boosting='gbdt',
                              min_data_in_leaf=10,
                              feature_fraction=0.9,
                              bagging_freq=1,
                              bagging_fraction=0.9,
                              importance_type='gain',
                              lambda_l1=0.2,
                              bagging_seed=random_seed,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              use_best_model=True)

    model.fit(trn_x, trn_y, verbose=False)
    val_pred = model.predict(val_x)
    test_pred = model.predict(test)
    rmse = np.sqrt(mean_squared_error(np.log1p(val_pred), np.log1p(val_y))) * 10

    return {'val': val_pred, 'test': test_pred, 'error': rmse}


import catboost as cat


def cat_model(trn_x, trn_y, val_x, val_y, test):
    model = cat.CatBoostRegressor(iterations=10000,
                                  learning_rate=0.01,
                                  depth=5,
                                  eval_metric='RMSE',
                                  colsample_bylevel=0.8,
                                  bagging_temperature=0.2,
                                  metric_period=None,
                                  early_stopping_rounds=200,
                                  random_seed=random_seed)

    model.fit(trn_x, trn_y, verbose=False)
    val_pred = model.predict(val_x)
    test_pred = model.predict(test)
    rmse = np.sqrt(mean_squared_error(np.log1p(val_pred), np.log1p(val_y))) * 10

    return {'val': val_pred, 'test': test_pred, 'error': rmse}

random_seed = 2000
k = 5
fold = list(KFold(k, shuffle = True, random_state = random_seed).split(X))
np.random.seed(random_seed)

val_pred = np.zeros(X.shape[0])
test_pred = np.zeros(X_test.shape[0])
final_err = 0

for i, (trn, val) in enumerate(fold):
    print(i + 1, "fold.")

    trn_x = X.loc[trn, :]
    trn_y = y[trn]
    val_x = X.loc[val, :]
    val_y = y[val]
    fold_val_pred = []
    fold_test_pred = []
    fold_err = []

    result = xgb_model(trn_x, trn_y, val_x, val_y, X_test)
    fold_val_pred.append(result['val'])
    fold_test_pred.append(result['test'])
    fold_err.append(result['error'])
    print("xgb model.", "{0:.5f}".format(result['error']))
    val_pred[val] += np.mean(np.array(fold_val_pred), axis=0)
    test_pred += np.mean(np.array(fold_test_pred), axis=0) / k
    final_err += (sum(fold_err) / len(fold_err)) / k
    print('')

print("final avg   err.", final_err)

pred_xgb = np.expm1(test_pred)


val_pred = np.zeros(X.shape[0])
test_pred = np.zeros(X_test.shape[0])
final_err = 0

for i, (trn, val) in enumerate(fold):
    print(i + 1, "fold.")

    trn_x = X.loc[trn, :]
    trn_y = y[trn]
    val_x = X.loc[val, :]
    val_y = y[val]

    fold_val_pred = []
    fold_test_pred = []
    fold_err = []

    result = lgb_model(trn_x, trn_y, val_x, val_y, X_test)
    fold_val_pred.append(result['val'])
    fold_test_pred.append(result['test'])
    fold_err.append(result['error'])
    print("lgb model.", "{0:.5f}".format(result['error']))

    val_pred[val] += np.mean(np.array(fold_val_pred), axis=0)
    test_pred += np.mean(np.array(fold_test_pred), axis=0) / k
    final_err += (sum(fold_err) / len(fold_err)) / k
    print('')

print("final avg   err.", final_err)

pred_lgb = np.expm1(test_pred)


val_pred = np.zeros(X.shape[0])
test_pred = np.zeros(X_test.shape[0])
final_err = 0

for i, (trn, val) in enumerate(fold):
    print(i + 1, "fold.")

    trn_x = X.loc[trn, :]
    trn_y = y[trn]
    val_x = X.loc[val, :]
    val_y = y[val]

    fold_val_pred = []
    fold_test_pred = []
    fold_err = []

    result = cat_model(trn_x, trn_y, val_x, val_y, X_test)
    fold_val_pred.append(result['val'])
    fold_test_pred.append(result['test'])
    fold_err.append(result['error'])
    print("cat model.", "{0:.5f}".format(result['error']))

    val_pred[val] += np.mean(np.array(fold_val_pred), axis=0)
    test_pred += np.mean(np.array(fold_test_pred), axis=0) / k
    final_err += (sum(fold_err) / len(fold_err)) / k
    print('')

print("final avg   err.", final_err)

pred_cat = np.expm1(test_pred)


submission = pd.DataFrame()
submission['imdb_scores'] = pred_xgb*0.45 + pred_cat*0.3 + pred_lgb*0.25
submission.to_csv("submission_xgb_lgb_cat_imdb.csv", index=False)