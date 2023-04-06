import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def get_X_y(data):
    X = data.drop(['Transported'], axis=1)
    y = data['Transported']
    #data.dropna(axis=0, subset=['Cabin'], inplace=True)
    return X, y

def drop_pointless_columns(df):
    df = df.drop(['PassengerId'], axis=1)
    return df

def snake_case_columns(df):
    df.rename(columns={
    'CryoSleep' : 'cryo_sleep',
    'HomePlanet' : 'home_planet',
    'Destination' : 'destination',
    'Cabin' : 'cabin',
    'Age' : 'age',
    'VIP' : 'vip',
    'RoomService' : 'room_service',
    'FoodCourt' : 'food_court',
    'ShoppingMall' : 'shopping_mall',
    'Spa' : 'spa',
    'VRDeck' : 'vr_deck',
    'Name' : 'name'
    }, inplace=True)

    return df

def impute_name(df):
    df.name = df.name.fillna('0')
    return df

def make_letter_cols(df):
    letters = []
    for name in df.name:
        for letter in name.lower():
            if letter not in letters:
                letters.append(letter)
    for letter in letters:
        df[letter] = df.name.str.contains(letter)
    return df

# def vectorize_name(df): # this is super dumb
#     df.name = df.name.str.lower()
#     df.name = df.name.apply(lambda x: unidecode(x))

#     stemmer = PorterStemmer()
#     df.name = df.name.apply(lambda x: stemmer.stem(x))

#     lemmatizer = WordNetLemmatizer()
#     df.name = df.name.apply(lambda x: lemmatizer.lemmatize(x))

#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform(df['name'])

#     df.drop(columns='name', inplace=True)
#     return df, vectors

# def add_len_name(df):
#     df['len_name'] = df.name.str.len()
#     return df

def add_len_name_surname_ratio(df):
    df['len_name'] = df.name.str.len()
    df['len_surname'] = df.name.str.split().str[0].str.len()
    df['len_name_surname_ratio'] = df.len_name / df.len_surname
    return df

def impute_cabin(df):
    df.cabin = df.cabin.fillna('F/0/P')
    return df


def engineer_cabin_cols(df):
    df['cabin_deck'] = df.cabin.str[0]
    df['cabin_num'] = df.cabin.str[2:-2]
    df['cabin_side'] = df.cabin.str[-1]
    df.drop('cabin', axis=1, inplace=True)
    return df

def impute_services(df):
    df['room_service'] = df.room_service.fillna(0)
    df['food_court'] = df.food_court.fillna(0)
    df['shopping_mall'] = df.shopping_mall.fillna(0)
    df['spa'] = df.spa.fillna(0)
    df['vr_deck'] = df.vr_deck.fillna(0)
    return df

def calculate_service_total(df):
    df['service_total'] = df.room_service + df.food_court + df.shopping_mall + df.spa + df.vr_deck
    return df

def impute_vip(df):
    df.vip = df.vip.astype(bool)
    vip_imputer = KNNImputer(n_neighbors=10)
    df['vip'] = vip_imputer.fit_transform(df[['vip']])
    return df

def impute_age(df):
    age_imputer = SimpleImputer(strategy='median')
    df['age'] = age_imputer.fit_transform(df[['age']])
    return df

def impute_cryo_sleep(df):
    df.cryo_sleep = df.cryo_sleep.astype(bool)
    if df.cryo_sleep is None and df.service_total > 0:
        cryo_sleep = 1
    else:
        cryo_sleep = 0
    return df

def scale_and_ohe(df):
    scaler = StandardScaler()
    df[['age', 'cabin_num', 'len_name_surname_ratio']] = scaler.fit_transform(df[['age', 'cabin_num', 'len_name_surname_ratio']])

    df = pd.get_dummies(df, columns=['cabin_deck', 'cabin_side', 'home_planet', 'destination'])

    df = df.drop(['name'], axis=1)
    return df

def process_df(df):
    df = drop_pointless_columns(df)
    df = snake_case_columns(df)
    df = impute_name(df)
    df = make_letter_cols(df)
#    df, vectors = vectorize_name(df)
#    df = add_len_name(df)
    df = add_len_name_surname_ratio(df)
    df = impute_cabin(df)
    df = engineer_cabin_cols(df)
    df = impute_services(df)
    df = calculate_service_total(df)
    df = impute_vip(df)
    df = impute_age(df)
    df = impute_cryo_sleep(df)
    df = scale_and_ohe(df)

#    df = pd.concat([df, pd.DataFrame(vectors.toarray())], axis=1)
    return df

def load_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data

def get_data():
    train_data, test_data = load_data()
    train_data = process_df(train_data)
    proc_test_data = process_df(test_data)
    X, y = get_X_y(train_data)
    return X, y, train_data, proc_test_data, test_data
