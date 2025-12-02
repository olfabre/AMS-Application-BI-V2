# Nettoyage des données

---

- Importation des données
- Définitions des différentes fonctions (utiles à la prise d'information sur le dataframe):
  - `count_missing_values(df)` : Retourne le nombre de valeurs manquantes, la listes dans colonnes qui contiennent des valeurs manquantes et un dictionnaire des colonnes précédentes avec le nombre de valeurs manquantes.
  - `count_missing_country_sex(df)` : Retourne le nombre de lignes présentants des valeurs manquantes dans les colonnes `country` et `sex`.
  - `display_data_types_and_missing_values(df)` : Imprime le type de chaque colonnes ainsi que le nombre de valeurs manquantes dans celle-ci.
  - `get_currencies(df)` : Retourne un dictionnaire comportant les différentes devises avec le nombre de leur occurences.
  - `cleaning(dataframe)` : Nettoie le dataframe en entrée :
    - Retire les lignes avec une valeur manquante dans la colonne `name` (4 lignes / +300'000 lignes)
    - Rempli les valeurs manquantes dans les colonnes `country` et `sex` par la valeur la plus fréquente de chaque colonne.
    - Converti les valeurs des colonnes `goal` et `pledged`, pour chaque ligne ayant une devise différentes de `USD`, en `USD`.
    - Renomme les colonnes `goal` et `pledged` en : `goal_usd` et `pledged_usd`.