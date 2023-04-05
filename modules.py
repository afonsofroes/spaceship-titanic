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

def get_X_y(data):
    X = data.drop(['Transported'], axis=1)
    y = data['Transported']
    data.dropna(axis=0, subset=['Cabin'], inplace=True)
    return X, y

def drop_pointless_columns(X):
    X = X.drop(['PassengerId', 'Name', 'HomePlanet', 'Destination'], axis=1)
    return X

def snake_case_columns(df):
    df.rename(columns={
    'CryoSleep' : 'cryo_sleep',
    'Cabin' : 'cabin',
    'Age' : 'age',
    'VIP' : 'vip',
    'RoomService' : 'room_service',
    'FoodCourt' : 'food_court',
    'ShoppingMall' : 'shopping_mall',
    'Spa' : 'spa',
    'VRDeck' : 'vr_deck'
    }, inplace=True)

    return df

def engineer_cabin_cols(df):
    cabin_imputer = SimpleImputer(strategy='constant', fill_value='F0P')
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
    df[['age', 'cabin_num']] = scaler.fit_transform(df[['age', 'cabin_num']])

    df = pd.get_dummies(df, columns=['cabin_deck', 'cabin_side'])
    return df

def process_df(df):
    df = drop_pointless_columns(df)
    df = snake_case_columns(df)
    df = engineer_cabin_cols(df)
    df = impute_services(df)
    df = calculate_service_total(df)
    df = impute_vip(df)
    df = impute_age(df)
    df = impute_cryo_sleep(df)
    df = scale_and_ohe(df)
    return df
