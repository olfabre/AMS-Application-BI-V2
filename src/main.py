import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

# ====================================
# Importation des donneés
# ====================================

df = pd.read_csv(os.path.join('resources/csv', 'ks-projects.csv'), encoding='latin1')

#print(df.head())

# ====================================
# Définition des fonctions d'analyse
# ====================================

# Comptage des valeurs manquantes dans le dataset
def count_missing_values(df):
    counter = 0
    columns_missing_values = []
    nb_values_missing = {}
    for col in df.columns:
        nb_values_missing[col] = 0
        for value in df[col]:
            if pd.isnull(value) or value == '?':
                counter += 1
                nb_values_missing[col] += 1
                if col not in columns_missing_values:
                    columns_missing_values.append(col)
    return counter, columns_missing_values, nb_values_missing
#print(f"Nombre total de valeurs manquantes : {count_missing_values(df)}")

# Calcul du nombre de valeurs manquantes dans les colonnes country et sex
def count_missing_country_sex(df):
    nb_missing_values_country_sex = 0
    for i in range(len(df)):
        if (pd.isnull(df.loc[i, 'country']) or df.loc[i, 'country'] == '?') and (pd.isnull(df.loc[i, 'sex']) or df.loc[i, 'sex'] == '?'):
            nb_missing_values_country_sex += 1
    return nb_missing_values_country_sex
#print(f"Nombre de valeurs manquantes dans les colonnes country et sex : {count_missing_country_sex(df)}")

# Affichage des types de données et des valeurs manquantes
def display_data_types_and_missing_values(df):
    _, columns_missing_values, nb_values_missing = count_missing_values(df)
    for col in df.columns:
        str = f"{col}: {df[col].dtype}"
        if col in columns_missing_values:
            str += f" (missing values : {nb_values_missing[col]})"
        print(str)

# Récupération des différentes devises
def get_currencies(df):
    currencies = {}
    for row in df['currency']:
        if row not in currencies:
            currencies[row] = 1
        else:
            currencies[row] += 1
    return currencies

#print(f"Liste des devises utilisées : {get_currencies(df)}")

# Charger le fichier de conversion des devises en tableau
def load_currencies_conversion_table(filepath = "resources/conversion_usd.txt"):
    conversion_table = pd.read_csv(filepath, sep=" ", header=None, names=["currency", "rate"], engine='python')
    print(f"Table de conversion des devises chargée :\n{conversion_table}")
    return conversion_table

def convert_to_usd(amount, currency, conversion_table: pd.DataFrame):
    if currency == 'USD':
        return amount
    for index, row in conversion_table.iterrows():
        if row['currency'] == currency:
            return round(amount * row['rate'], 2)
    print(f"Taux de conversion non trouvé pour la devise : {currency}")
    return np.nan

def convert_dataframe_to_usd(dataframe, currencies, conversion_table: pd.DataFrame):
    converted_dataframe = dataframe.copy()
    
    for index, row in converted_dataframe.iterrows():
        for col in currencies:
            if col in converted_dataframe.columns and np.isnan(row[col]):
                print(f"Ligne {index}, colonne {col} contient la valeur NaN")
                return False
            else:
                converted_value = convert_to_usd(row[col], row['currency'], conversion_table)
                if converted_value == np.nan:
                    print(f"La valeur convertie pour {row[col]} : {col}, à la ligne {index} est np.nan")
                    return False
                converted_dataframe.at[index, col] = converted_value
    
    # Renommer les colonnes en ajoutant "_usd"
    rename_dict = {col: col + "_usd" for col in currencies}
    converted_dataframe = converted_dataframe.rename(columns=rename_dict)
    
    return converted_dataframe

# Nettoyage des données
def cleaning(dataframe):
    # Suppression des lignes avec une valeur manquante dans la colonne 'name' (4 lignes / +300'000 lignes)
    cleaned_dataframe = dataframe.copy()
    for row in cleaned_dataframe.itertuples():
        if (pd.isnull(row.name) or row.name == '?'):
            cleaned_dataframe = cleaned_dataframe.drop(row.Index)
            #print(f"Ligne {row.Index} supprimée (valeur manquante dans la colonne 'name')")
    
    # Remplissage des valeurs manquantes des colonnes 'country' et 'sex' par la valeur la plus fréquente de chaque colonne
    columns_to_impute = ['country', 'sex']
    imputer = SimpleImputer(strategy='most_frequent')
    cleaned_dataframe[columns_to_impute] = imputer.fit_transform(cleaned_dataframe[columns_to_impute])
    
    # Convertion des valeurs 'goal' et 'pledged' en USD
    convesion_table = load_currencies_conversion_table()
    cleaned_dataframe = convert_dataframe_to_usd(cleaned_dataframe, ['goal', 'pledged'], convesion_table)
    
    return cleaned_dataframe

#count = df.isnull().sum()
#print(f"Nombre de valeurs manquantes par colonne avant nettoyage :\n{count}")
#df_cleaned = cleaning(df)
#count_cleaned = df_cleaned.isnull().sum()
#print(f"Nombre de valeurs manquantes par colonne après nettoyage :\n{count_cleaned}")

df_cleaned = cleaning(df)
print(df_cleaned)

# Enregistrement des données nettoyées dans un fichier CSV
df_cleaned.to_csv(os.path.join('resources/csv', 'ks-projects-cleaned.csv'), index=False, sep=',', encoding='latin1')
print("Données nettoyées enregistrées dans 'resources/csv/ks-projects-cleaned.csv'")
