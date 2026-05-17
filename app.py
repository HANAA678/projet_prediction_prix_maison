"""
=============================================================
  INTERFACE WEB — Prédiction Prix Maison
  python app.py  →  http://localhost:7860
=============================================================
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib, os

# ── Charger le pipeline ───────────────────────────────────────────────────────
MODEL_PATH = "models/pipeline_prix_maison.pkl"
if not os.path.exists(MODEL_PATH):
    print("❌ Lance d'abord : python train.py")
    exit(1)

pipeline = joblib.load(MODEL_PATH)
print("✅ Modèle chargé !")

# ── Fonction de prédiction ────────────────────────────────────────────────────
def predire(area, bedrooms, bathrooms, floors, age, distance,
            garage, parking, garden, security,
            school_nearby, hospital_nearby, shopping_mall_nearby,
            public_transport, crime_rate, population_density,
            location, income_level):

    input_df = pd.DataFrame([{
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "floors": floors, "age": age, "distance": distance,
        "garage": int(garage), "parking": int(parking),
        "garden": int(garden), "security": int(security),
        "school_nearby": int(school_nearby),
        "hospital_nearby": int(hospital_nearby),
        "shopping_mall_nearby": int(shopping_mall_nearby),
        "public_transport": int(public_transport),
        "crime_rate": crime_rate,
        "population_density": population_density,
        "location": location,
        "income_level": income_level,
    }])

    prix = np.expm1(pipeline.predict(input_df)[0])
    # Fourchette ±5%
    bas  = prix * 0.95
    haut = prix * 1.05

    # Score qualité
    score = ""
    if prix < 300_000:    score = "🟢 Abordable"
    elif prix < 800_000:  score = "🟡 Milieu de gamme"
    elif prix < 1_500_000: score = "🟠 Premium"
    else:                  score = "🔴 Luxe"

    amenities = []
    if garage:            amenities.append("🚗 Garage")
    if parking:           amenities.append("🅿️ Parking")
    if garden:            amenities.append("🌳 Jardin")
    if security:          amenities.append("🔒 Sécurité")
    if school_nearby:     amenities.append("🏫 École proche")
    if hospital_nearby:   amenities.append("🏥 Hôpital proche")
    if shopping_mall_nearby: amenities.append("🛍️ Centre commercial")
    if public_transport:  amenities.append("🚌 Transport public")
    amenities_str = "  ".join(amenities) if amenities else "Aucune"

    result = f"""
## 🏠 Prix Estimé : **{prix:,.0f}**   {score}

### 📊 Fourchette de prix
| Bas (−5%) | Estimation centrale | Haut (+5%) |
|:---:|:---:|:---:|
| {bas:,.0f} | **{prix:,.0f}** | {haut:,.0f} |

---
### 📋 Résumé de la maison
| Caractéristique | Valeur |
|---|---|
| Surface | **{area} m²** |
| Chambres / SDB / Étages | **{bedrooms} / {bathrooms} / {floors}** |
| Âge | **{age} ans** |
| Distance centre | **{distance} km** |
| Localisation | **{location.capitalize()}** |
| Niveau de revenus quartier | **{income_level.upper()}** |
| Taux de criminalité | **{crime_rate:.2f}** |
| Densité population | **{population_density:,}** hab/km² |

### 🏡 Équipements
{amenities_str}
"""
    return result

# ── Interface Gradio ──────────────────────────────────────────────────────────
with gr.Blocks(title="🏠 Prédiction Prix Maison") as demo:

    gr.Markdown("""
    # 🏠 Prédiction du Prix d'une Maison
    **Modèle : Régression Linéaire (Ridge) — R² = 77.7%** (réaliste ✅)
    Remplis les caractéristiques et obtiens une estimation instantanée.
    ---
    """)

    with gr.Row():
        # ── Col 1 : Physique ──────────────────────────────────────────────────
        with gr.Column():
            gr.Markdown("### 📐 Caractéristiques physiques")
            area      = gr.Slider(300, 10000, value=2000, step=50,  label="Surface (m²)")
            bedrooms  = gr.Slider(1, 10,  value=3, step=1,          label="Chambres")
            bathrooms = gr.Slider(1, 6,   value=2, step=1,          label="Salles de bain")
            floors    = gr.Slider(1, 5,   value=2, step=1,          label="Étages")
            age       = gr.Slider(0, 100, value=10, step=1,         label="Âge de la maison (ans)")
            distance  = gr.Slider(1, 50,  value=10, step=1,         label="Distance du centre (km)")

        # ── Col 2 : Localisation & Quartier ──────────────────────────────────
        with gr.Column():
            gr.Markdown("### 📍 Localisation & Quartier")
            location     = gr.Radio(["premium","medium","low"],
                                    value="medium", label="Type de zone")
            income_level = gr.Radio(["high","mid","low"],
                                    value="mid",    label="Niveau de revenus du quartier")
            crime_rate   = gr.Slider(0, 10, value=3.0, step=0.1,   label="Taux de criminalité (0=sûr, 10=dangereux)")
            population_density = gr.Slider(100, 15000, value=5000, step=100, label="Densité population (hab/km²)")

        # ── Col 3 : Équipements ───────────────────────────────────────────────
        with gr.Column():
            gr.Markdown("### 🏡 Équipements & Services")
            garage               = gr.Checkbox(label="🚗 Garage",               value=True)
            parking              = gr.Checkbox(label="🅿️ Parking",              value=False)
            garden               = gr.Checkbox(label="🌳 Jardin",               value=True)
            security             = gr.Checkbox(label="🔒 Système de sécurité",  value=False)
            school_nearby        = gr.Checkbox(label="🏫 École proche",         value=True)
            hospital_nearby      = gr.Checkbox(label="🏥 Hôpital proche",       value=False)
            shopping_mall_nearby = gr.Checkbox(label="🛍️ Centre commercial",    value=False)
            public_transport     = gr.Checkbox(label="🚌 Transport public",     value=True)

    btn = gr.Button("💰 Estimer le Prix", variant="primary", size="lg")
    result = gr.Markdown("*Remplis le formulaire et clique sur Estimer...*")

    btn.click(
        fn=predire,
        inputs=[area, bedrooms, bathrooms, floors, age, distance,
                garage, parking, garden, security,
                school_nearby, hospital_nearby, shopping_mall_nearby,
                public_transport, crime_rate, population_density,
                location, income_level],
        outputs=result
    )

    gr.Markdown("""
    ---
    ### 📌 Ce qui influence le plus le prix
    | Feature | Impact |
    |---|---|
    | Surface (area) | ⭐⭐⭐⭐⭐ Dominant (corrélation 0.99) |
    | Localisation (premium) | ⭐⭐⭐⭐ Fort |
    | Âge (neuf = +cher) | ⭐⭐⭐ Moyen |
    | Distance du centre | ⭐⭐⭐ Moyen |
    | Criminalité (faible = +cher) | ⭐⭐ Faible |
    | Équipements | ⭐ Léger |
    """)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
