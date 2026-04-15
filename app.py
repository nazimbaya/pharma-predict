import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# 1. CONFIGURATION DE L'INTERFACE
# ==========================================
st.set_page_config(page_title="Pharma-Predict | Pharmakode", page_icon="💊")

st.title("💊 Pharma-Predict (Version Chronologique)")
st.subheader("Analyse de tendance par Intelligence Artificielle")
st.write("---")

# ==========================================
# 2. LE CERVEAU IA (Scikit-Learn)
# ==========================================
def predire_prochaine_vente(historique):
    # Si on a trop peu de données, on fait une moyenne simple
    if len(historique) < 3:
        return np.mean(historique) if len(historique) > 0 else 0
        
    # Transformation de l'historique pour l'algorithme
    X = np.array(range(len(historique))).reshape(-1, 1)
    y = np.array(historique)
    
    # Régression Polynomiale pour détecter les accélérations/ralentissements
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Prédiction du point suivant (demain)
    X_next = poly.transform([[len(historique)]])
    prediction = model.predict(X_next)[0]
    
    return max(0.1, prediction) # On évite le zéro pour la division

# ==========================================
# 3. L'APPLICATION UTILISATEUR
# ==========================================
uploaded_file = st.file_uploader("Déposez votre fichier (Date, Vente, Stock)", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Fichier chargé !")
    
    st.write("### ⚙️ Étape 1 : Mappage des colonnes")
    cols = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        col_nom = st.selectbox("Nom du Produit", cols)
        col_date = st.selectbox("Colonne Date", cols)
    with col2:
        col_vente = st.selectbox("Colonne Ventes (Unités)", cols)
        col_stock = st.selectbox("Colonne Stock actuel", cols)

    st.write("---")
    
    if st.button("🚀 Lancer le Diagnostic IA"):
        # Conversion de la date
        df[col_date] = pd.to_datetime(df[col_date])
        
        st.write("### 🚨 Analyse des Ruptures Prévues")
        
        # On groupe par produit pour analyser chaque historique séparément
        produits = df[col_nom].unique()
        
        for p in produits:
            # On récupère les données du produit triées par date
            data_p = df[df[col_nom] == p].sort_values(by=col_date)
            
            # On extrait les séries
            historique_ventes = data_p[col_vente].values
            dernier_stock = data_p[col_stock].iloc[-1]
            
            # Prédiction IA
            prediction_demain = predire_prochaine_vente(historique_ventes)
            
            # Calcul de couverture
            jours_restants = dernier_stock / prediction_demain
            
            # Affichage des alertes
            if jours_restants < 5:
                st.error(f"**{p}** | Stock: {int(dernier_stock)} | **Rupture sous {int(jours_restants)} jours**")
                st.caption(f"Tendance IA : {prediction_demain:.2f} ventes prévues demain.")
            elif jours_restants < 15:
                st.warning(f"**{p}** | Stock: {int(dernier_stock)} | Couverture : {int(jours_restants)} jours")
