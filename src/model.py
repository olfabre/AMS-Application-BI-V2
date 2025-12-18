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
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
import time

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
# Entraînement et évaluation
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
    plt.savefig('comparison_models.png', dpi=300, bbox_inches='tight')
    plt.show()
    
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
    plt.savefig('training_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nGraphique sauvegardé : training_time.png")

# ====================================
# MAIN
# ====================================

if __name__ == "__main__":
    print("Chargement des données...")
    train_df, val_df, test_df = load_data()
    
    print("\nPréparation des données...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_df, val_df, test_df)
    
    print("\nNormalisation des features...")
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)
    
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