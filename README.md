# 🏠 Prédiction Prix Maison — Régression Linéaire
> 50 000 maisons • 18 features • R² = 99.8% • 100% CPU

## 📁 Structure
```
house_price_v2/
├── data/
│   └── house_price_50k.csv    ← Dataset (50 000 maisons)
├── models/                    ← Pipeline sauvegardé
├── resultats/                 ← Graphiques d'analyse
├── train.py                   ← Entraînement + évaluation
├── app.py                     ← Interface web Gradio
└── requirements.txt
```

## 🚀 Lancement (3 commandes)
```bash
pip install -r requirements.txt
python train.py      # ~30 secondes sur CPU
python app.py        # Ouvre http://localhost:7860
```

## 📊 Performances
| Métrique          | Valeur     |
|-------------------|------------|
| R²                | **99.8%**  |
| MAE               | ~16 000    |
| RMSE              | ~20 000    |
| Cross-val (5-fold)| 0.998 ± 0.0002 |

## 🧠 Pipeline complet
```
Données brutes (18 features)
    ↓ StandardScaler        → colonnes numériques
    ↓ OneHotEncoder         → location (premium/medium/low)
    ↓ OrdinalEncoder        → income_level (low < mid < high)
    ↓ Ridge Regression (α=1)
    → Prix estimé
```

## 📌 Features utilisées
| Feature | Type | Description |
|---|---|---|
| area | Numérique | Surface en m² (dominant !) |
| bedrooms / bathrooms / floors | Numérique | Nombre de pièces |
| age | Numérique | Âge de la maison |
| distance | Numérique | Distance du centre (km) |
| crime_rate | Numérique | Taux de criminalité |
| population_density | Numérique | Densité habitants/km² |
| garage / parking / garden ... | Binaire (0/1) | Équipements |
| location | Catégoriel | premium / medium / low |
| income_level | Ordinal | low < mid < high |
