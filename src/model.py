import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings("ignore")

# ====================================
# Chargement des données
# ====================================

def load_data():
    """Charge les données train, val et test"""
    train_df = pd.read_csv(os.path.join('resources/csv', 'train.csv'), encoding='latin1')
    val_df = pd.read_csv(os.path.join('resources/csv', 'val.csv'), encoding='latin1')
    test_df = pd.read_csv(os.path.join('resources/csv', 'test.csv'), encoding='latin1')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df, target_col='state'):
    """Sépare features et target"""
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ====================================
# Normalisation des données
# ====================================

def scale_features(X_train, X_val, X_test):
    """Normalise les features avec StandardScaler"""
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir en DataFrame pour garder les noms de colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_val_scaled, X_test_scaled

# ====================================
# Grilles d'hyperparamètres
# ====================================

def get_param_grids():
    """Retourne les grilles de paramètres pour chaque modèle"""
    param_grids = {
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'Naive Bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'Decision Tree': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [1000]
        }
    }
    return param_grids

# ====================================
# Entraînement avec GridSearch
# ====================================

def train_with_gridsearch(model, model_name, param_grid, X_train, y_train, X_val, y_val, X_test, y_test):
    """Entraîne avec GridSearch sur ensemble de validation"""
    print(f"\n{'='*80}")
    print(f"GridSearch : {model_name}")
    print(f"{'='*80}")
    print(f"Paramètres testés : {len(list(param_grid.values())[0]) if param_grid else 'Aucun'}")
    
    start_time = time.time()
    
    if param_grid and len(param_grid) > 0:
        # Créer un dataset combiné train+val pour le GridSearch
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        
        # Créer des indices pour le split manuel train/val
        train_indices = list(range(len(X_train)))
        val_indices = list(range(len(X_train), len(X_train_val)))
        custom_cv = [(train_indices, val_indices)]
        
        # GridSearch avec validation manuelle
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=custom_cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_val, y_train_val)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"\n✓ Meilleurs paramètres : {best_params}")
        print(f"✓ Meilleur score F1 (val) : {grid_search.best_score_:.4f}")
    else:
        # Pas d'hyperparamètres à optimiser
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = "Aucun"
    
    training_time = time.time() - start_time
    print(f"Temps total : {training_time:.2f}s")
    
    # Évaluation
    results = evaluate_model(best_model, model_name, X_train, y_train, X_val, y_val, X_test, y_test)
    results['training_time'] = training_time
    results['best_params'] = str(best_params)
    
    return results, best_model

def evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """Évalue un modèle sur tous les ensembles"""
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Métriques de base
    results = {
        'model': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'val_precision': precision_score(y_val, y_val_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'val_recall': recall_score(y_val, y_val_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'val_f1': f1_score(y_val, y_val_pred),
        'test_f1': f1_score(y_test, y_test_pred),
    }
    
    # AUC-ROC
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
        results['test_auc_roc'] = roc_auc_score(y_test, y_test_proba)
    
    # Matthews Correlation Coefficient (bon pour classes équilibrées)
    results['test_mcc'] = matthews_corrcoef(y_test, y_test_pred)
    
    # Affichage
    print(f"\n--- Résultats TEST ---")
    print(f"Accuracy:  {results['test_accuracy']:.4f}")
    print(f"Precision: {results['test_precision']:.4f}")
    print(f"Recall:    {results['test_recall']:.4f}")
    print(f"F1-Score:  {results['test_f1']:.4f}")
    print(f"MCC:       {results['test_mcc']:.4f}")
    if 'test_auc_roc' in results:
        print(f"AUC-ROC:   {results['test_auc_roc']:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n--- Matrice de confusion (TEST) ---")
    print(cm)
    print(f"VP: {cm[1,1]}, VN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}")
    
    return results

# ====================================
# Test de tous les modèles (avec GridSearch)
# ====================================

def test_all_models_with_gridsearch(X_train, y_train, X_val, y_val, X_test, y_test):
    """Teste tous les algorithmes avec GridSearch"""
    
    base_models = {
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    param_grids = get_param_grids()
    
    all_results = []
    trained_models = {}
    
    for model_name, model in base_models.items():
        param_grid = param_grids.get(model_name, {})
        
        results, trained_model = train_with_gridsearch(
            model, model_name, param_grid,
            X_train, y_train, 
            X_val, y_val, 
            X_test, y_test
        )
        all_results.append(results)
        trained_models[model_name] = trained_model
    
    return pd.DataFrame(all_results), trained_models

# ====================================
# Analyse de l'importance des features
# ====================================

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Affiche l'importance des features pour les modèles compatibles"""
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name} ne supporte pas l'importance des features")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Features - {model_name}', fontsize=14, fontweight='bold')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    filename = f'feature_importance_{model_name.replace(" ", "_")}.png'
    plt.savefig("resources/train/" + filename, dpi=300, bbox_inches='tight')
    #plt.show()
    
    print(f"\n✓ Graphique sauvegardé : {filename}")
    
    # Afficher le top 10
    print(f"\nTop 10 features importantes ({model_name}) :")
    for i in range(min(10, top_n)):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# ====================================
# Visualisations améliorées
# ====================================

def plot_comparison_enhanced(results_df):
    """Affiche des graphiques de comparaison améliorés"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparaison des algorithmes (après optimisation)', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'mcc']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'MCC']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        
        if metric == 'auc_roc':
            # AUC-ROC n'a pas train/val
            if 'test_auc_roc' in results_df.columns:
                ax.bar(results_df['model'], results_df['test_auc_roc'], alpha=0.8, color='purple')
                ax.set_ylabel('AUC-ROC')
            else:
                ax.text(0.5, 0.5, 'Non disponible', ha='center', va='center')
        elif metric == 'mcc':
            # MCC uniquement sur test
            ax.bar(results_df['model'], results_df['test_mcc'], alpha=0.8, color='orange')
            ax.set_ylabel('MCC')
        else:
            # Métriques train/val/test
            train_col = f'train_{metric}'
            val_col = f'val_{metric}'
            test_col = f'test_{metric}'
            
            x = np.arange(len(results_df))
            width = 0.25
            
            ax.bar(x - width, results_df[train_col], width, label='Train', alpha=0.8)
            ax.bar(x, results_df[val_col], width, label='Val', alpha=0.8)
            ax.bar(x + width, results_df[test_col], width, label='Test', alpha=0.8)
            ax.legend()
            ax.set_xticks(x)
            ax.set_ylabel(title)
        
        ax.set_title(title)
        ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        if metric not in ['mcc']:
            ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('resources/train/comparison_models_optimized.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    print("\n✓ Graphique sauvegardé : comparison_models_optimized.png")

# ====================================
# Entraînement et évaluation (sans GridSearch)
# ====================================

def train_and_evaluate(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """Entraîne un modèle et l'évalue"""
    print(f"\n{'='*80}")
    print(f"Entraînement : {model_name}")
    print(f"{'='*80}")
    
    # Entraînement
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Temps d'entraînement : {training_time:.2f}s")
    
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Métriques
    results = {
        'model': model_name,
        'training_time': training_time,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'val_precision': precision_score(y_val, y_val_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'val_recall': recall_score(y_val, y_val_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'val_f1': f1_score(y_val, y_val_pred),
        'test_f1': f1_score(y_test, y_test_pred),
    }
    
    # AUC-ROC (si le modèle supporte predict_proba)
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
        results['test_auc_roc'] = roc_auc_score(y_test, y_test_proba)
    
    # Affichage des résultats
    print(f"\n--- Résultats sur TRAIN ---")
    print(f"Accuracy:  {results['train_accuracy']:.4f}")
    print(f"Precision: {results['train_precision']:.4f}")
    print(f"Recall:    {results['train_recall']:.4f}")
    print(f"F1-Score:  {results['train_f1']:.4f}")
    
    print(f"\n--- Résultats sur VALIDATION ---")
    print(f"Accuracy:  {results['val_accuracy']:.4f}")
    print(f"Precision: {results['val_precision']:.4f}")
    print(f"Recall:    {results['val_recall']:.4f}")
    print(f"F1-Score:  {results['val_f1']:.4f}")
    
    print(f"\n--- Résultats sur TEST ---")
    print(f"Accuracy:  {results['test_accuracy']:.4f}")
    print(f"Precision: {results['test_precision']:.4f}")
    print(f"Recall:    {results['test_recall']:.4f}")
    print(f"F1-Score:  {results['test_f1']:.4f}")
    if 'test_auc_roc' in results:
        print(f"AUC-ROC:   {results['test_auc_roc']:.4f}")
    
    # Matrice de confusion
    print(f"\n--- Matrice de confusion (TEST) ---")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    return results, model

# ====================================
# Test de tous les modèles
# ====================================

def test_all_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Teste tous les algorithmes"""
    
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    all_results = []
    trained_models = {}
    
    for model_name, model in models.items():
        results, trained_model = train_and_evaluate(
            model, model_name, 
            X_train, y_train, 
            X_val, y_val, 
            X_test, y_test
        )
        all_results.append(results)
        trained_models[model_name] = trained_model
    
    return pd.DataFrame(all_results), trained_models

# ====================================
# Visualisation des résultats
# ====================================

def plot_comparison(results_df):
    """Affiche des graphiques de comparaison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparaison des algorithmes', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Données pour train, val, test
        train_col = f'train_{metric}'
        val_col = f'val_{metric}'
        test_col = f'test_{metric}'
        
        x = np.arange(len(results_df))
        width = 0.25
        
        ax.bar(x - width, results_df[train_col], width, label='Train', alpha=0.8)
        ax.bar(x, results_df[val_col], width, label='Validation', alpha=0.8)
        ax.bar(x + width, results_df[test_col], width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Modèle')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('resources/train/comparison_models.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    print("\nGraphique sauvegardé : comparison_models.png")

def plot_training_time(results_df):
    """Affiche le temps d'entraînement"""
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(results_df))
    bars = plt.bar(results_df['model'], results_df['training_time'], color=colors)
    
    plt.xlabel('Modèle', fontsize=12)
    plt.ylabel('Temps d\'entraînement (s)', fontsize=12)
    plt.title('Temps d\'entraînement par modèle', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('resources/train/training_time.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    print("\nGraphique sauvegardé : training_time.png")

# ====================================
# MAIN
# ====================================

if __name__ == "__main__":
    greadsearch = True

    if greadsearch:
        # ======================= #
        # GridSearch
        # ======================= #
        print("="*80)
        print("ENTRAÎNEMENT DES MODÈLES AVEC OPTIMISATION DES HYPERPARAMÈTRES")
        print("="*80)
        # ======================= #

    print("Chargement des données...")
    train_df, val_df, test_df = load_data()
    
    print("\nPréparation des données...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_df, val_df, test_df)
    
    print("\nNormalisation des features...")
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)
    
    if not greadsearch:
        print("\n" + "="*80)
        print("DÉBUT DES TESTS DES ALGORITHMES")
        print("="*80)
        
        # Test de tous les modèles
        results_df, trained_models = test_all_models(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test
        )
        
        # Affichage du tableau récapitulatif
        print("\n" + "="*80)
        print("TABLEAU RÉCAPITULATIF")
        print("="*80)
        print("\n--- Performances sur TEST ---")
        print(results_df[['model', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']].to_string(index=False))
        
        # Visualisations
        print("\nGénération des graphiques...")
        plot_comparison(results_df)
        plot_training_time(results_df)
        
        # Sauvegarde des résultats
        results_df.to_csv('model_comparison_results.csv', index=False)
        print("\nRésultats sauvegardés : model_comparison_results.csv")
        
        print("\n" + "="*80)
        print("TESTS TERMINÉS !")
        print("="*80)
        
        exit(0)

    print("\n" + "="*80)
    print("DÉBUT DES TESTS AVEC GRIDSEARCH")
    print("="*80)
    
    # Test de tous les modèles avec GridSearch
    results_df, trained_models = test_all_models_with_gridsearch(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test
    )
    
    # Tableau récapitulatif
    print("\n" + "="*80)
    print("TABLEAU RÉCAPITULATIF FINAL")
    print("="*80)
    print("\n--- Performances sur TEST ---")
    display_cols = ['model', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_mcc']
    if 'test_auc_roc' in results_df.columns:
        display_cols.append('test_auc_roc')
    print(results_df[display_cols].to_string(index=False))
    
    print("\n--- Temps d'entraînement ---")
    print(results_df[['model', 'training_time']].to_string(index=False))
    
    # Visualisations
    print("\n" + "="*80)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*80)
    
    plot_comparison_enhanced(results_df)
    
    # Importance des features pour Random Forest
    if 'Random Forest' in trained_models:
        print("\nAnalyse de l'importance des features (Random Forest)...")
        plot_feature_importance(
            trained_models['Random Forest'], 
            X_train.columns.tolist(), 
            'Random Forest'
        )
    
    # Sauvegarde des résultats
    results_df.to_csv('resources/train/model_comparison_optimized.csv', index=False)
    print("\n✓ Résultats sauvegardés : model_comparison_optimized.csv")
    
    print("\n" + "="*80)
    print("OPTIMISATION TERMINÉE !")
    print("="*80)