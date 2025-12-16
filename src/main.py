import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
    missing_columns = [col for col in currencies if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Colonnes manquantes : {missing_columns}")
        return dataframe
    
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
    counter = 0
    for row in cleaned_dataframe.itertuples():
        if (pd.isnull(row.name) or row.name == '?'):
            cleaned_dataframe = cleaned_dataframe.drop(row.Index)
            print(f"Ligne {row.Index} supprimée (valeur manquante dans la colonne 'name')")
        elif row.state in ['live', 'undefined', 'suspended', 'canceled']:
            cleaned_dataframe = cleaned_dataframe.drop(row.Index)
            counter += 1
            print(f"Ligne {row.Index} supprimée (state dans ['live', 'undefined', 'suspended', 'canceled']) | counter : {counter}")
    
    # Remplissage des valeurs manquantes des colonnes 'country' et 'sex' par la valeur la plus fréquente de chaque colonne
    columns_to_impute = ['country', 'sex']
    imputer = SimpleImputer(strategy='most_frequent')
    cleaned_dataframe[columns_to_impute] = imputer.fit_transform(cleaned_dataframe[columns_to_impute])
    
    # Convertion des valeurs 'goal' et 'pledged' en USD
    convesion_table = load_currencies_conversion_table()
    cleaned_dataframe = convert_dataframe_to_usd(cleaned_dataframe, ['goal', 'pledged'], convesion_table)
    
    if 'currency' in dataframe.columns:
        cleaned_dataframe = cleaned_dataframe.drop('currency', axis=1)
    
    for row in cleaned_dataframe.itertuples():
        if row.goal_usd < 1:
            cleaned_dataframe = cleaned_dataframe.drop(row.Index)
            print(f"Ligne {row.Index} supprimé (objectif < 1$)")
    
    return cleaned_dataframe


def analyze_categorical_numerical(df, cat_var, num_var, use_log=False, figsize=(14, 6)):
    df_clean = df[[cat_var, num_var]].dropna()
    plot_var = num_var
    y_label = num_var
    #if use_log:
    #    df_clean[f'{num_var}_plot'] = np.log1p(df_clean[num_var])
    #    y_label = f'{num_var} (log)'
    #    plot_var = f'{num_var}_plot'
    #else:
    #    df_clean[f'{num_var}_plot'] = df_clean[num_var]
    #    y_label = num_var
    
    # ==========================================
    # 1. TABLEAU DES STATISTIQUES DESCRIPTIVES
    # ==========================================
    print("="*80)
    print(f"ANALYSE : {cat_var} <-> {num_var}")
    print("="*80)
    
    stats_summary = df_clean.groupby(cat_var)[num_var].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('25%', lambda x: x.quantile(0.25)),
        ('75%', lambda x: x.quantile(0.75)),
        ('max', 'max')
    ]).round(2)
    
    print("\nSTATISTIQUES DESCRIPTIVES PAR GROUPE:")
    print(stats_summary)
    
    # ==========================================
    # 2. VISUALISATIONS
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # --- VIOLIN PLOT --- #
    sns.violinplot(data=df_clean, x=cat_var, y=plot_var, ax=axes[0], palette='Set2')
    axes[0].set_title(f'Violin Plot: {cat_var} vs {num_var}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(cat_var, fontsize=12)
    axes[0].set_ylabel(y_label, fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    if use_log:
        axes[0].set_yscale('log')
        axes[0].set_ylabel(f'{y_label} (échelle log)', fontsize=12)
        y_min = df_clean[num_var].min()
        y_max = df_clean[num_var].max()
        axes[0].set_ylim(y_min * 0.5, y_max * 2)
    
    if df_clean[cat_var].nunique() > 5:
        axes[0].tick_params(axis='x', rotation=45)
    
    # --- BOXPLOT --- #
    sns.boxplot(data=df_clean, x=cat_var, y=plot_var, ax=axes[1], palette='Set2')
    axes[1].set_title(f'Boxplot: {cat_var} vs {num_var}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(cat_var, fontsize=12)
    axes[1].set_ylabel(y_label, fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    if use_log:
        axes[1].set_yscale('log')
        axes[1].set_ylabel(f'{y_label} (échelle log)', fontsize=12)
        axes[1].set_ylim(y_min * 0.5, y_max * 2)
    
    if df_clean[cat_var].nunique() > 5:
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'resources/categoriel_x_numeric/{cat_var}_x_{num_var}.png', bbox_inches='tight', dpi=300)
    
    # ==========================================
    # 3. TABLEAU DANS UN FICHIER SÉPARÉ
    # ==========================================
    n_rows = len(stats_summary)
    fig_height = max(4, n_rows * 0.4 + 2)
    
    fig_table = plt.figure(figsize=(12, fig_height))
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table_data = []
    headers = [cat_var] + list(stats_summary.columns)
    
    for idx in stats_summary.index:
        row = [str(idx)] + [f"{val:.2f}" if isinstance(val, (int, float)) else str(val) 
                            for val in stats_summary.loc[idx]]
        table_data.append(row)
    
    table = ax_table.table(cellText=table_data, 
                           colLabels=headers,
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.15] + [0.1] * len(stats_summary.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    fig_table.suptitle(f'Statistiques descriptives : {cat_var} vs {num_var}', 
                       fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(f'resources/categoriel_x_numeric/{cat_var}_x_{num_var}_tab.png', bbox_inches='tight', dpi=300)
    
    print("\n" + "="*80 + "\n")
    
    return stats_summary

#print(f"Liste des devises utilisées : {get_currencies(df)}")

#count = df.isnull().sum()
#print(f"Nombre de valeurs manquantes par colonne avant nettoyage :\n{count}")
#df_cleaned = cleaning(df)
#count_cleaned = df_cleaned.isnull().sum()
#print(f"Nombre de valeurs manquantes par colonne après nettoyage :\n{count_cleaned}")

df = pd.read_csv(os.path.join('resources/csv', 'ks-projects-cleaned-state.csv'), encoding='latin1')
#df_cleaned = df

counter = {}
for row in df.itertuples():
    if row.state in ['live', 'undefined', 'suspended', 'canceled']:
        if row.state not in counter:
            counter[row.state] = 1
        else:
            counter[row.state] += 1
print(counter)

df_cleaned = cleaning(df)
print(df_cleaned)

# Enregistrement des données nettoyées dans un fichier CSV
df_cleaned.to_csv(os.path.join('resources/csv', 'ks-projects-cleaned-state.csv'), index=False, sep=',', encoding='latin1')
print("Données nettoyées enregistrées dans 'resources/csv/ks-projects-cleaned-state.csv'")

counter = {}
for row in df_cleaned.itertuples():
    if row.state in ['live', 'undefined', 'suspended', 'canceled']:
        if row.state not in counter:
            counter[row.state] = 1
        else:
            counter[row.state] += 1
print(counter)

#analyze_categorical_numerical(df_cleaned, 'state', 'goal_usd', use_log=True)
col_cat = ['state', 'category', 'sex', 'country']
col_num = ['goal_usd', 'pledged_usd', 'backers', 'age']
col_log = ['goal_usd', 'pledged_usd', 'backers']

for cat in col_cat:
    for num in col_num:
        log = False
        if num in col_log:
            log = True
        analyze_categorical_numerical(df_cleaned, cat, num, use_log=log)

#for i in range(len(df_cleaned)):
#    for num in col_num:
#        if df_cleaned.loc[i, num] > 0 and df_cleaned.loc[i, num] < 1:
#            print(f'{i} : {df_cleaned.loc[i, num]} | {num}')
