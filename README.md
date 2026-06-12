# Modélisation de Portefeuilles avec Python & Machine Learning (Projet Étudiant P26)

## Présentation du Projet
Ce dépôt contient les travaux réalisés dans le cadre du **Projet Etudiant (PE) - Modélisation de Portefeuilles avec Python & Machine Learning** à l'Université de Technologie de Troyes (UTT). L'objectif est de concevoir un moteur quantitatif d'allocation d'actifs sur le **CAC 40** en faisant converger la théorie financière classique et l'apprentissage supervisé / non supervisé.

L'écosystème intègre :
* **Data Engineering :** Ingestion automatisée en temps réel depuis l'API Yahoo Finance.
* **Machine Learning :** Modèles de prédiction et de classification directionnelle (KNN, SVM, Random Forest) et clustering comportemental (K-Means + PCA).
* **Moteur Quantitatif :** Implémentation de la frontière efficiente de Markowitz, du MEDAF (CAPM) et du modèle bayésien de **Black-Litterman** interconnecté aux prédictions de l'IA.
* **Interface Décisionnelle :** Un tableau de bord interactif développé avec **Streamlit**.

---

## Dépendances et Packages Python

Le projet est développé localement en utilisant **Python 3.14.5** (via Anaconda et Visual Studio Code). Les bibliothèques requises se divisent en plusieurs catégories :

### 1. Collecte et Manipulation de Données
* `yfinance` : Connexion aux API de Yahoo Finance pour la récupération des flux boursiers.
* `pandas` : Traitement des structures de données temporelles (*DataFrames*), calcul des rendements logarithmiques et statistiques glissantes.
* `numpy` : Opérations matricielles et calculs algébriques lourds.

### 2. Machine Learning et Optimisation Financière
* `scikit-learn` : Implémentation des algorithmes de ML (KNN, SVM, Random Forest, K-Means, PCA) et régularisation de covariance via Ledoit-Wolf.
* `scipy` : Optimisation non linéaire sous contraintes pour la maximisation du ratio de Sharpe (`scipy.optimize`).
* `PyPortfolioOpt` : Bibliothèque d'ingénierie financière utilisée pour stabiliser l'implémentation de la Frontière Efficiente de Markowitz et le modèle de Black-Litterman.

### 3. Visualisation et Interface Graphique
* `matplotlib` & `seaborn` : Génération des histogrammes de rendement (KDE), tracés temporels et cartes de chaleur (*heatmaps*).
* `streamlit` : Framework de déploiement de l'application web interactive.

---

## Tutoriel d'Installation et de Lancement

Suivez ces étapes pour cloner le projet, installer l'environnement et lancer l'application Streamlit sur votre machine locale.

### Étape 1 : Cloner le dépôt distant
Ouvrez votre terminal et exécutez la commande suivante pour copier le projet en local :
```bash
git clone [https://github.com/KamgueAnge/projet_etudiant_P26.git](https://github.com/KamgueAnge/projet_etudiant_P26.git)
cd projet_etudiant_P26
```

### Étape 2 : Créer et activer un environnement virtuel (Recommandé)
- Sur Windows :
```bash
python -m venv venv
.\venv\Scripts\activate
```
- Sur macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Étape 3 : Installer les dépendances
```bash
pip install yfinance pandas numpy scikit-learn scipy PyPortfolioOpt matplotlib seaborn streamlit
```

### Pour lancer l'application streamlit
Placez vous dans le dossier script puis effectuez cette commande:
```bash
streamlit run streamlitApp.py
```


---

## Arborescence du dépôt

projet_etudiant_P26/
├── .gitignore                             
├── README.md                              
│
├── data/                                 
│   └── cac40_stocks_2010_2021.csv                  
│
├── documentations/                        
│   ├── rapport_PE_P26.pdf                 
│   ├── fiche_PE.pdf                      
│   └── projet_etudiant.pdf               
│
└── scripts/                             
    ├── streamlitApp.py                    
    └── analyse_des_donnes_financieres.ipynb 
