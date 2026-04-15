import streamlit as st
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Pharma-Predict | Pharmakode", page_icon="💊")

# Header avec ton identité
st.title("💊 Pharma-Predict")
st.subheader("Anticipez vos ruptures de stock en un clic")
st.write("---")

# 1. Zone d'Upload
uploaded_file = st.file_uploader("Déposez votre export CSV (PharmX, Chifa, etc.)", type=["csv"])

if uploaded_file is not None:
    # Lecture du fichier
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()
    
    st.success("Fichier chargé avec succès !")
    
    # 2. Mapping intelligent (L'utilisateur aide l'IA)
    st.sidebar.header("Configuration")
    col_prod = st.sidebar.selectbox("Colonne Produit", cols)
    col_stock = st.sidebar.selectbox("Colonne Stock Actuel", cols)
    col_ventes = st.sidebar.selectbox("Colonne Ventes (7 derniers jours)", cols)
    
    if st.sidebar.button("Lancer l'Analyse"):
        # Nettoyage des données
        df_analyse = df[[col_prod, col_stock, col_ventes]].copy()
        df_analyse.columns = ['Produit', 'Stock', 'Ventes']
        
        # Calcul de la tendance (Simple mais efficace)
        # On calcule combien de jours il reste avant la rupture
        df_analyse['Jours_Restants'] = df_analyse['Stock'] / (df_analyse['Ventes'] / 7)
        
        # 3. Affichage des Alertes
        st.write("### 🚨 Alertes Prioritaires")
        
        # On filtre ceux qui risquent de rompre sous 5 jours
        alertes = df_analyse[df_analyse['Jours_Restants'] < 5].sort_values(by='Jours_Restants')
        
        if not alertes.empty:
            for index, row in alertes.iterrows():
                st.error(f"**{row['Produit']}** : Rupture prévue dans **{int(row['Jours_Restants'])} jours**")
        else:
            st.balloons()
            st.success("Bravo ! Aucun risque de rupture détecté à court terme.")

        # 4. Tableau complet
        st.write("### 📊 État complet du stock")
        st.dataframe(df_analyse)

st.write("---")
st.caption("Propulsé par Pharmakode - L'innovation au service de l'officine algérienne.")
