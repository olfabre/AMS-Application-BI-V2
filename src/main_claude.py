import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ====================================
# Importation des données
# ====================================

df = pd.read_csv(os.path.join('resources/csv', 'ks-projects.csv'), encoding='latin1')

# ====================================
# Définition des fonctions d'analyse OPTIMISÉES
# ====================================

# Comptage vectorisé des valeurs manquantes
def count_missing_values(df):
    """Version vectorisée - beaucoup plus rapide"""
    missing_mask = df.isnull() | (df == '?')
    nb_values_missing = missing_mask.sum().to_dict()
    columns_missing_values = [col for col, count in nb_values_missing.items() if count > 0]
    counter = sum(nb_values_missing.values())
    return counter, columns_missing_values, nb_values_missing

# Comptage vectorisé pour country et sex
def count_missing_country_sex(df):
    """Version vectorisée"""
    country_missing = df['country'].isnull() | (df['country'] == '?')
    sex_missing = df['sex'].isnull() | (df['sex'] == '?')
    return (country_missing & sex_missing).sum()

# Charger le fichier de conversion des devises
def load_currencies_conversion_table(filepath="resources/conversion_usd.txt"):
    conversion_table = pd.read_csv(filepath, sep=" ", header=None, 
                                   names=["currency", "rate"], engine='python')
    print(f"Table de conversion des devises chargée :\n{conversion_table}")
    return conversion_table

# Conversion vectorisée en USD
def convert_dataframe_to_usd(dataframe, currencies, conversion_table: pd.DataFrame):
    """Version VECTORISÉE - beaucoup plus rapide"""
    missing_columns = [col for col in currencies if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Colonnes manquantes : {missing_columns}")
        return dataframe
    
    converted_dataframe = dataframe.copy()
    
    # Créer un dictionnaire de conversion pour un accès rapide
    conversion_dict = dict(zip(conversion_table['currency'], conversion_table['rate']))
    conversion_dict['USD'] = 1.0  # USD vers USD
    
    # Convertir en utilisant map (vectorisé)
    currency_rates = converted_dataframe['currency'].map(conversion_dict)
    
    # Vérifier les devises manquantes
    missing_currencies = converted_dataframe.loc[currency_rates.isnull(), 'currency'].unique()
    if len(missing_currencies) > 0:
        print(f"Devises sans taux de conversion : {missing_currencies}")
        return dataframe
    
    # Conversion vectorisée pour chaque colonne
    for col in currencies:
        converted_dataframe[col + '_usd'] = (converted_dataframe[col] * currency_rates).round(2)
    
    # Supprimer les colonnes originales
    converted_dataframe = converted_dataframe.drop(columns=currencies)
    
    return converted_dataframe

# Nettoyage VECTORISÉ
def cleaning(dataframe):
    """Version OPTIMISÉE avec opérations vectorisées NumPy/Pandas"""
    print("Début du nettoyage...")
    print(f"Nombre de lignes initial : {len(dataframe)}")
    
    cleaned_dataframe = dataframe.copy()
    
    # 1. SUPPRESSION VECTORISÉE des lignes avec name manquant
    name_valid = ~(cleaned_dataframe['name'].isnull() | (cleaned_dataframe['name'] == '?'))
    rows_removed_name = (~name_valid).sum()
    cleaned_dataframe = cleaned_dataframe[name_valid].copy()
    print(f"Lignes supprimées (name manquant) : {rows_removed_name}")
    
    # 2. SUPPRESSION VECTORISÉE des états non désirés
    valid_states = ~cleaned_dataframe['state'].isin(['live', 'undefined', 'suspended', 'canceled'])
    rows_removed_state = (~valid_states).sum()
    cleaned_dataframe = cleaned_dataframe[valid_states].copy()
    print(f"Lignes supprimées (state invalide) : {rows_removed_state}")
    
    # 3. REMPLISSAGE VECTORISÉ des valeurs manquantes
    columns_to_impute = ['country', 'sex']
    
    # Remplacer '?' par NaN pour l'imputation
    for col in columns_to_impute:
        cleaned_dataframe[col] = cleaned_dataframe[col].replace('?', np.nan)
    
    imputer = SimpleImputer(strategy='most_frequent')
    cleaned_dataframe[columns_to_impute] = imputer.fit_transform(cleaned_dataframe[columns_to_impute])
    print(f"Valeurs manquantes imputées dans {columns_to_impute}")
    
    # 4. CONVERSION VECTORISÉE en USD
    conversion_table = load_currencies_conversion_table()
    cleaned_dataframe = convert_dataframe_to_usd(cleaned_dataframe, ['goal', 'pledged'], conversion_table)
    
    # Supprimer la colonne currency
    if 'currency' in cleaned_dataframe.columns:
        cleaned_dataframe = cleaned_dataframe.drop('currency', axis=1)
    
    # 5. SUPPRESSION VECTORISÉE des objectifs < 1$
    valid_goal = cleaned_dataframe['goal_usd'] >= 1
    rows_removed_goal = (~valid_goal).sum()
    cleaned_dataframe = cleaned_dataframe[valid_goal].copy()
    print(f"Lignes supprimées (goal < 1$) : {rows_removed_goal}")
    
    # Réinitialiser l'index
    cleaned_dataframe = cleaned_dataframe.reset_index(drop=True)
    
    print(f"Nombre de lignes final : {len(cleaned_dataframe)}")
    print("Nettoyage terminé !")
    
    return cleaned_dataframe

# Fonction de préparation OPTIMISÉE
def prepar_dataframe(dataframe):
    """Version optimisée de la préparation"""
    prepared_dataframe = dataframe.copy()
    
    # Suppression de colonnes
    col_to_drop = ['id', 'name', 'subcategory', 'pledged_usd', 'backers']
    prepared_dataframe = prepared_dataframe.drop(
        columns=[col for col in col_to_drop if col in prepared_dataframe.columns]
    )
    
    # Conversion vectorisée des dates
    prepared_dataframe['start_date'] = pd.to_datetime(prepared_dataframe['start_date'])
    prepared_dataframe['end_date'] = pd.to_datetime(prepared_dataframe['end_date'])
    
    # Calcul vectorisé de la durée
    prepared_dataframe['duration'] = (
        prepared_dataframe['end_date'] - prepared_dataframe['start_date']
    ).dt.total_seconds().astype(int)
    
    # Extraction vectorisée année et mois
    prepared_dataframe['start_year'] = prepared_dataframe['start_date'].dt.year
    prepared_dataframe['start_month'] = prepared_dataframe['start_date'].dt.month
    
    # Suppression des colonnes de dates
    prepared_dataframe = prepared_dataframe.drop(columns=['start_date', 'end_date'])
    
    # Encodage vectorisé
    if 'state' in prepared_dataframe.columns:
        prepared_dataframe['state'] = (prepared_dataframe['state'] == 'successful')
    
    if 'sex' in prepared_dataframe.columns:
        prepared_dataframe['sex'] = (prepared_dataframe['sex'] == 'female')
    
    return prepared_dataframe

# Application du OneHotEncoding sur 'category', 'country'
def apply_one_hot_encoding(df, columns_to_encode=['category', 'country']):
    """
    Applique le OneHotEncoding sur les colonnes catégorielles spécifiées
    """
    df_encoded = df.copy()
    
    print(f"Colonnes avant encodage : {list(df_encoded.columns)}")
    print(f"Shape avant encodage : {df_encoded.shape}")
    
    # Vérifier que les colonnes existent
    columns_to_encode = [col for col in columns_to_encode if col in df_encoded.columns]
    
    if not columns_to_encode:
        print("Aucune colonne à encoder trouvée.")
        return df_encoded
    
    # Appliquer OneHotEncoding avec pandas (plus simple)
    df_encoded = pd.get_dummies(df_encoded, columns=columns_to_encode, prefix=columns_to_encode, drop_first=False)
    
    print(f"\nColonnes après encodage : {list(df_encoded.columns)}")
    print(f"Shape après encodage : {df_encoded.shape}")
    print(f"Nombre de nouvelles colonnes créées : {df_encoded.shape[1] - df.shape[1] + len(columns_to_encode)}")
    
    return df_encoded

# Rééquilibrer la classe 'state'
def balance_classes(df, target_col='state', random_state=42):
    """
    Équilibre les classes en sous-échantillonnant la classe majoritaire
    """
    # Séparer les classes
    successful = df[df[target_col] == 1]
    failed = df[df[target_col] == 0]
    
    print(f"Avant équilibrage :")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    # Sous-échantillonner la classe majoritaire
    n_samples = min(len(successful), len(failed))
    
    successful_sampled = successful.sample(n=n_samples, random_state=random_state)
    failed_sampled = failed.sample(n=n_samples, random_state=random_state)
    
    # Combiner et mélanger
    balanced_df = pd.concat([successful_sampled, failed_sampled])
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nAprès équilibrage :")
    print(f"  Successful: {(balanced_df[target_col] == 1).sum()}")
    print(f"  Failed: {(balanced_df[target_col] == 0).sum()}")
    print(f"  Total: {len(balanced_df)}")
    
    return balanced_df

# Fonction de répartition équitable des données
def balanced_split(df, target_col='state', test_size=0.2, val_size=0.2, random_state=42):
    df_copy = df.copy()
    
    # Colonnes numériques à équilibrer
    numeric_cols = ['goal_usd']
    
    # Créer des bins (quartiles) pour chaque colonne numérique
    stratify_col = df_copy[target_col].astype(str)
    
    for col in numeric_cols:
        if col in df_copy.columns:
            # Découper en 4 quartiles
            df_copy[f'{col}_bin'] = pd.qcut(df_copy[col], q=4, labels=False, duplicates='drop')
            # Combiner avec la stratification
            stratify_col = stratify_col + '_' + df_copy[f'{col}_bin'].astype(str)
    
    # Division Train vs Temp
    train_idx, temp_idx = train_test_split(
        df_copy.index,
        test_size=(test_size + val_size),
        stratify=stratify_col,
        random_state=random_state
    )
    
    # Division Validation vs Test
    temp_df = df_copy.loc[temp_idx]
    stratify_temp = temp_df[target_col].astype(str)
    
    for col in numeric_cols:
        if f'{col}_bin' in temp_df.columns:
            stratify_temp = stratify_temp + '_' + temp_df[f'{col}_bin'].astype(str)
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_size/(test_size + val_size),
        stratify=stratify_temp,
        random_state=random_state
    )
    
    # Récupérer les ensembles
    train = df.loc[train_idx]
    val = df.loc[val_idx]
    test = df.loc[test_idx]
    
    return train, val, test

# Fonction pour enregistrer les sections faites (train, validate, test)
def save_splits(train_df, val_df, test_df, output_dir='resources/csv'):
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False, encoding='latin1')
    val_df.to_csv(val_path, index=False, encoding='latin1')
    test_df.to_csv(test_path, index=False, encoding='latin1')
    
    print(f"\n{'='*80}")
    print("SAUVEGARDE DES DIVISIONS")
    print(f"{'='*80}")
    print(f"✓ Train sauvegardé : {train_path} ({len(train_df)} lignes)")
    print(f"✓ Validation sauvegardé : {val_path} ({len(val_df)} lignes)")
    print(f"✓ Test sauvegardé : {test_path} ({len(test_df)} lignes)")
    print(f"{'='*80}\n")

# ====================================
# EXEMPLE D'UTILISATION
# ====================================

if __name__ == "__main__":
    # Nettoyage
    df_cleaned = cleaning(df)
    
    # Sauvegarde
    df_cleaned.to_csv(
        os.path.join('resources/csv', 'ks-projects-cleaned-state.csv'), 
        index=False, sep=',', encoding='latin1'
    )
    print("Données nettoyées enregistrées !")
    
    # Préparation
    prepared_dataframe = prepar_dataframe(df_cleaned)
    prepared_dataframe.to_csv(
        os.path.join('resources/csv', 'prepared_dataframe.csv'), 
        index=False, sep=',', encoding='latin1'
    )
    print("Données préparées enregistrées !")
    
    # Répartition des données
    prepared_dataframe = pd.read_csv(os.path.join('resources/csv', 'prepared_dataframe.csv'), encoding='latin1')
    
    print("=" * 80)
    print("DONNÉES AVANT ENCODAGE")
    print("=" * 80)
    print(f"\nShape : {prepared_dataframe.shape}")
    print(f"\nColonnes : {list(prepared_dataframe.columns)}")
    print(f"\nPremières lignes :")
    print(prepared_dataframe.head(10))
    print(f"\nTypes de données :")
    print(prepared_dataframe.dtypes)
    print(f"\nValeurs uniques dans 'category' : {prepared_dataframe['category'].nunique()}")
    print(f"Valeurs uniques dans 'country' : {prepared_dataframe['country'].nunique()}")
    
    # ÉTAPE 1 : OneHotEncoding
    encoded_dataframe = apply_one_hot_encoding(prepared_dataframe, columns_to_encode=['category', 'country'])
    
    print("\n" + "=" * 80)
    print("DONNÉES APRÈS ENCODAGE")
    print("=" * 80)
    print(f"\nShape : {encoded_dataframe.shape}")
    print(f"\nColonnes : {list(encoded_dataframe.columns)}")
    print(f"\nPremières lignes :")
    print(encoded_dataframe.head(10))
    print(f"\nTypes de données :")
    print(encoded_dataframe.dtypes)
    
    # ÉTAPE 2 : Rééquilibrage
    balanced_dataframe = balance_classes(encoded_dataframe, target_col='state', random_state=42)
    
    print("\n" + "=" * 80)
    print("DONNÉES APRÈS RÉÉQUILIBRAGE")
    print("=" * 80)
    print(f"\nShape : {balanced_dataframe.shape}")
    print(f"\nPremières lignes :")
    print(balanced_dataframe.head(10))
    
    # ÉTAPE 3 : Split
    train_df, val_df, test_df = balanced_split(balanced_dataframe, target_col='state', test_size=0.15, val_size=0.15)
    save_splits(train_df, val_df, test_df)
    
    print("\n" + "=" * 80)
    print("DONNÉES APRÈS SPLIT")
    print("=" * 80)
    print(f"\nTrain shape : {train_df.shape}")
    print(f"Val shape : {val_df.shape}")
    print(f"Test shape : {test_df.shape}")
    print(f"\nPremières lignes du TRAIN :")
    print(train_df.head(5))
    
    '''
    print(f"State mean - Train: {train_df['state'].mean():.3f}, Val: {val_df['state'].mean():.3f}, Test: {test_df['state'].mean():.3f}")
    print("=== Comptages par division ===")
    print(f"\nTrain:")
    print(f"  Total: {len(train_df)}")
    print(f"  Successful: {(train_df['state'] == 1).sum()}")
    print(f"  Failed: {(train_df['state'] == 0).sum()}")

    print(f"\nValidation:")
    print(f"  Total: {len(val_df)}")
    print(f"  Successful: {(val_df['state'] == 1).sum()}")
    print(f"  Failed: {(val_df['state'] == 0).sum()}")

    print(f"\nTest:")
    print(f"  Total: {len(test_df)}")
    print(f"  Successful: {(test_df['state'] == 1).sum()}")
    print(f"  Failed: {(test_df['state'] == 0).sum()}")

    # Bonus : proportions
    print(f"\n=== Proportions de succès ===")
    print(f"Train: {(train_df['state'] == 1).sum() / len(train_df) * 100:.1f}%")
    print(f"Val:   {(val_df['state'] == 1).sum() / len(val_df) * 100:.1f}%")
    print(f"Test:  {(test_df['state'] == 1).sum() / len(test_df) * 100:.1f}%")
    '''
    
    