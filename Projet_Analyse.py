import pandas as pd
import numpy as np
import streamlit as st
from sklearn import impute
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import compose
import category_encoders as ce
from sklearn.decomposition import PCA
import plotly.express as px
from streamlit_option_menu import option_menu
import time
import base64
import matplotlib.pyplot as plt
from scipy import stats
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
st.set_page_config(
    page_title="Analyse Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #008080;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #008080;
        margin-bottom: 10px;
    }
    .info-text {
        font-size: 16px;
        margin-bottom: 10px;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .stProgress .st-bo {
        background-color: #008080;
    }
    .stButton>button {
        background-color: #008080;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #36648B;
    }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'x_cleaned' not in st.session_state:
    st.session_state.x_cleaned = None
if 'x_encoded' not in st.session_state:
    st.session_state.x_encoded = None
if 'x_normalized' not in st.session_state:
    st.session_state.x_normalized = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'one_binary' not in st.session_state: 
    st.session_state.one_binary = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'reduction_method' not in st.session_state:
    st.session_state.reduction_method = None
if 'x_reduced' not in st.session_state:
    st.session_state.x_reduced = None
if 'pca' not in st.session_state:
    st.session_state.pca = None
if 'n_components' not in st.session_state:
    st.session_state.n_components = None
if 'imputer' not in st.session_state:
    st.session_state.imputer = None

with st.sidebar:
    st.markdown('<div class="main-header">Prediction Data</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    choice = option_menu(
        menu_title="Navigation",
        options=[
            'Charger les données',
            'Aperçu des données',
            'Analyse Statistique',
            'Nettoyage des données',
            'Encodage des données',
            'Normalisation des données',
            'Matrice de corrélation',
            'Réduction de dimension',
            'Entraînement du modèle',
            'Évaluation du modèle',
            'Visualisation des prédictions',
            'Prédiction sur nouvelles données'
        ],
        icons=[
            'upload', 'eye', 'brush', 'bar-chart', 'code', 'sliders', 
        'grid', 'layers', 'gear', 'graph-up', 'file-earmark-bar-graph', 'lightbulb'
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#E0F2F1"},
            "icon": {"color": "#008080", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#008080", "color": "white"},
        }
    )
    
    st.markdown("---")
    st.markdown(" État du processus")
    data_loaded = st.session_state.df is not None
    data_cleaned = st.session_state.x_cleaned is not None
    data_encoded = st.session_state.x_encoded is not None
    data_normalized = st.session_state.x_normalized is not None
    model_trained = st.session_state.model is not None
    
    st.markdown(f"📂 Données chargées: {'✅' if data_loaded else '❌'}")
    st.markdown(f"🧹 Données nettoyées: {'✅' if data_cleaned else '❌'}")
    st.markdown(f"🔄 Données encodées: {'✅' if data_encoded else '❌'}")
    st.markdown(f"📊 Données normalisées: {'✅' if data_normalized else '❌'}")
    st.markdown(f"⚙️ Modèle entraîné: {'✅' if model_trained else '❌'}")

    progress_items = [data_loaded, data_cleaned, data_encoded, data_normalized, model_trained]
    progress = sum(progress_items) / len(progress_items)
    st.progress(progress)

if choice == 'Charger les données':
    st.markdown('<div class="main-header">Chargement des données</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-text">Chargez votre fichier CSV pour commencer l\'analyse.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Choisir un fichier CSV', type='csv')

    if uploaded_file is not None:
        with st.spinner('Chargement des données...'):
            file_data = uploaded_file.read()
            uploaded_file.seek(0)  #
            st.session_state.df = pd.read_csv(uploaded_file)
            st.markdown('<div class="success-box">✅ Fichier chargé avec succès !</div>', unsafe_allow_html=True)
            file_name = uploaded_file.name
            b64 = base64.b64encode(file_data).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Télécharger {file_name}</a>'
            st.markdown(f'<div class="link-box">Lien de votre fichier: {href}</div>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
            st.markdown('<div class="sub-hexader">Aperçu des données</div>', unsafe_allow_html=True)
            st.dataframe(st.session_state.df, use_container_width=True)

elif choice == 'Aperçu des données':
    st.markdown('<div class="main-header">Aperçu des données</div>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        option = st.selectbox('Choisissez ce que vous voulez voir:',
                                ['Résumé statistique',
                                 'Variables indépendantes',
                                 'Variable cible',
                                 'Nombre d\'instances',
                                 'Nombre d\'attributs',
                                 'Nombre de Valeurs manquantes',
                                 'detection des doublons',
                                 'Premières lignes du jeu de données'])
        
        st.markdown('<div class="sub-header">Types de données</div>', unsafe_allow_html=True)
        st.dataframe(df.dtypes.reset_index().rename(columns={0:'Type', 'index':'Colonne'}), use_container_width=True)
        
        if option == 'Résumé statistique':
            st.markdown('<div class="sub-header">Résumé statistique</div>', unsafe_allow_html=True)
            st.dataframe(df.describe(), use_container_width=True) 

        elif option == 'Variables indépendantes':
            st.markdown('<div class="sub-header">Variables indépendantes</div>', unsafe_allow_html=True)
            x = df.iloc[:, :-1]
            st.dataframe(x, use_container_width=True) 
            
        elif option == 'Variable cible':
            st.markdown('<div class="sub-header">Variable cible</div>', unsafe_allow_html=True)
            y = df.iloc[:, -1]
            st.dataframe(pd.DataFrame(y), use_container_width=True)
            fig = px.histogram(y, title="Distribution de la variable cible", nbins=20)   
            fig.update_traces(marker=dict(color='skyblue', line=dict(color='black', width=2)),opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

        elif option == 'Nombre d\'instances':
            st.markdown('<div class="sub-header">Nombre d\'instances</div>', unsafe_allow_html=True)
            st.info(f"Le jeu de données contient {df.shape[0]} instances.")   

        elif option == 'Nombre d\'attributs':
            st.markdown('<div class="sub-header">Nombre d\'attributs</div>', unsafe_allow_html=True)
            st.info(f"Le jeu de données contient {df.shape[1]} attributs.")         

        elif option == 'Nombre de Valeurs manquantes':
            st.markdown('<div class="sub-header">Nombre de Valeurs manquantes</div>', unsafe_allow_html=True)
            st.info(f"Le jeu de données contient {st.session_state.df.isnull().sum().sum()}")
        
        elif option == 'detection des doublons':
            st.markdown('<div class="sub-header">Nombre de doublons</div>', unsafe_allow_html=True)
            st.info(f"Le jeu de données contient {st.session_state.df.duplicated().sum()}")   

        elif option == 'Premières lignes du jeu de données':
                st.markdown('<div class="sub-header">Premières lignes du jeu de données</div>', unsafe_allow_html=True)
                st.dataframe(df.head(), use_container_width=True)
    else:
        st.warning("Veuillez d'abord charger un jeu de données.")

elif choice == 'Analyse Statistique':
    st.markdown('<div class="main-header">Analyse Statistique Exploratoire</div>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        tab1, tab2, tab3 = st.tabs([
            "Statistiques Descriptives",
            "Visualisation des données",
            "Détection des Valeurs Aberrantes"])

        with tab1:
            st.markdown('<div class="sub-header">Statistiques Descriptives </div>', unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected_var = st.selectbox("Choisissez une variable numérique:", numeric_cols)
            if selected_var:
                    st.metric("Moyenne", f"{df[selected_var].mean():.2f}")
                    st.metric("Médiane", f"{df[selected_var].median():.2f}")
                    st.metric("Écart-type", f"{df[selected_var].std():.2f}")
                    st.metric("Variance", f"{df[selected_var].var():.2f}")
            skewness = df[selected_var].skew()
            kurtosis = df[selected_var].kurtosis()  

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Asymétrie (Skewness)", f"{skewness:.2f}")
                st.caption("\> 0: queue à droite, < 0: queue à gauche")
            with col2:
                st.metric("Aplatissement (Kurtosis)", f"{kurtosis:.2f}")
                st.caption("\> 3: distribution pointue, < 3: distribution plate")  

        with tab2:
            st.markdown('<div class="sub-header">Visualisation des données</div>', unsafe_allow_html=True)  
            plot_type = st.radio("Type de visualisation:", 
                                ["Histogramme", "Box Plot", "Nuage de points"])
            
            selected_var_dist = st.selectbox("Variable à analyser:", numeric_cols)
            
            if plot_type == "Histogramme":
                fig = px.histogram(df, x=selected_var_dist, title=f"Distribution de {selected_var_dist}",nbins=30)
                fig.update_traces(marker=dict(color='skyblue', line=dict(color='black', width=2)), opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Box Plot":
                fig = px.box(df, y=selected_var_dist, title=f"Box Plot de {selected_var_dist}")
                fig.update_traces(marker=dict(color='skyblue', line=dict(color='black', width=2)), opacity=0.9)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Nuage de points":
                st.markdown('<div class="sub-header">Nuage de Points (Scatterplot)</div>', unsafe_allow_html=True)
                if len(numeric_cols) > 1:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_var = st.selectbox(
                            "Sélectionner la variable en X",
                            numeric_cols,
                            index=0,
                            key='scatter_x'
                        )      

                    with col2:
                        y_var = st.selectbox(
                            "Sélectionner la variable en X",
                            numeric_cols,
                            index=0,
                            key='scatter_y'
                        )

                    fig, ax = plt.subplots(figsize=(8, 6))

                    scatter = ax.scatter( df[x_var], df[y_var], alpha=0.6, edgecolors='w', linewidth=0.5,s=50) 
                    ax.set_xlabel(x_var, fontsize=12)
                    ax.set_ylabel(y_var, fontsize=12)
                    ax.set_title(f"Relation entre {x_var} et {y_var}", fontsize=14, pad=20)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                        st.warning("⚠️ Pas assez de variables numériques pour afficher un nuage de points. "
                        f"({len(numeric_cols)} variable(s) disponible(s), 2 minimum requis)")           
           
        with tab3:
            st.markdown('<div class="sub-header">Détection des Valeurs Aberrantes</div>', unsafe_allow_html=True)
            selected_var_outlier = st.selectbox("Variable à analyser pour les outliers:", numeric_cols)
            if selected_var_outlier:
                Q1 = df[selected_var_outlier].quantile(0.25)
                Q3 = df[selected_var_outlier].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[selected_var_outlier] < lower_bound) | (df[selected_var_outlier] > upper_bound)]

                st.write(f"**Méthode IQR:**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Q1 ", f"{Q1:.2f}")
                with col2:
                    st.metric("Q3 ", f"{Q3:.2f}")
                with col3:
                    st.metric("IQR", f"{IQR:.2f}")
                
                st.write(f"**Bornes IQR:** [{lower_bound:.2f}, {upper_bound:.2f}]")

                st.markdown('<div class="sub-header">Boxplot (Boîte à moustaches</div>', unsafe_allow_html=True)
                show_outliers = st.checkbox("Afficher les outliers sur le graphique", value=True)
                fig = px.box(df,y=selected_var_outlier,title=f"Boxplot - Détection des outliers pour {selected_var_outlier}",
                    points="outliers" if show_outliers else False)
                
                if st.checkbox("Afficher les lignes des bornes IQR"):
                    fig.add_hline(
                        y=lower_bound,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Borne inférieure: {lower_bound:.2f}",
                        annotation_position="bottom right"
                    )
                    fig.add_hline(
                        y=upper_bound,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Borne supérieure: {upper_bound:.2f}",
                        annotation_position="top right"
                    )
                
                fig.update_layout(
                    yaxis_title="",
                    xaxis_title=selected_var_outlier,
                    showlegend=False,
                    height=500
                )
                
                fig.update_traces(marker=dict(color='skyblue', line=dict(color='black', width=2)), opacity=0.9)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("ℹ️ Comment lire ce boxplot ?"):
                    st.markdown("""
                    **Éléments du boxplot :**
                    
                     **Boîte (rectangle) :**
                    - **Bord inférieur** = Q1 premier quartile
                    - **Bord supérieur** = Q3 troisième quartile
                    - **Largeur** = IQR (Intervalle InterQuartile)
                    - **Ligne au milieu** = Médiane (Q2)
                    
                     **Moustaches :**
                    - S'étendent jusqu'à la dernière valeur dans les limites
                    - Limites = [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
                    
                     **Points (outliers) :**
                    - Les valeurs en dehors de ces bornes sont considérées comme des outliers.
                    """)

                if len(outliers) > 0:
                    st.markdown(f"🔴 {len(outliers)} Outliers détectés")                
                else:
                    st.success(f"✅ Aucun outlier détecté pour la variable '{selected_var_outlier}'")

    else:
        st.warning("Veuillez d'abord charger un jeu de données.")

elif choice == 'Nettoyage des données':
    st.markdown('<div class="main-header">Nettoyage des données</div>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        tab1, tab2, tab3= st.tabs(["Information sur les valeurs manquantes", "Gestion des valeurs manquantes", 
                                   "Gestion des doublons"])
        
        with tab1:
            option = st.selectbox('Choisissez ce que vous voulez voir:',
                                ['Colonnes avec des Valeurs Manquantes',
                                 'Nombre de Lignes avec des Valeurs Manquantes'])
            if option == 'Colonnes avec des Valeurs Manquantes':
                st.markdown('<div class="sub-header">Colonnes avec des Valeurs Manquantes</div>', unsafe_allow_html=True)
                missing_data = df.isnull().sum().reset_index()
                missing_data.columns = ['Colonne', 'Nombre de valeurs manquantes']
                missing_data['Pourcentage'] = round(missing_data['Nombre de valeurs manquantes'] / len(df) * 100, 2)
                st.dataframe(missing_data, use_container_width=True)
                fig = px.bar(missing_data, x='Colonne', y='Nombre de valeurs manquantes', title="Valeurs manquantes par colonne")
                fig.update_traces(marker=dict(color='skyblue', line=dict(color='black', width=2)), opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)

            elif option == 'Nombre de Lignes avec des Valeurs Manquantes':
                st.markdown('<div class="sub-header">Nombre de Lignes avec des Valeurs Manquantes</div>', unsafe_allow_html=True)
                rows_with_missing = df.isnull().any(axis=1).sum()
                st.info(f"{rows_with_missing} lignes (sur {len(df)}) contiennent au moins une valeur manquante.")
                labels = ['Avec valeurs manquantes', 'Sans valeurs manquantes']
                values = [rows_with_missing, len(df) - rows_with_missing]
                fig = px.pie(values=values, names=labels, title="Proportion de lignes avec des valeurs manquantes")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown('<div class="sub-header">Gestion des Valeurs Manquantes</div>', unsafe_allow_html=True)
            imputation_method = st.selectbox('Méthode d\'imputation:',
                                            ['Moyenne', 'Médiane', 'Most_frequent', 'Constant'])
            
            fill_value = None
            if imputation_method == 'Constant':
                fill_value = st.number_input('Valeur de remplacement:', value=0.0)

           
            if st.button('Nettoyer les Données', key='clean_data_button'):
                with st.spinner('Nettoyage des données en cours...'):
                    x = df.iloc[:, :-1].values  
                    y = df.iloc[:, -1].values  
                    numeric_columns = df.select_dtypes(include=np.number).columns
                    columns_to_impute = [i for i, col in enumerate(df.columns[:-1]) if col in numeric_columns and df[col].isnull().any()]
                    if imputation_method == 'Moyenne':
                        imputer = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
                    elif imputation_method == 'Médiane':
                        imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
                    elif imputation_method == 'Most_frequent':
                        imputer = impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                    elif imputation_method == 'Constant':
                        imputer = impute.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=fill_value)
                    
                    if columns_to_impute:
                        x[:, columns_to_impute] = imputer.fit_transform(x[:, columns_to_impute])
                    
                    time.sleep(2)
                    st.session_state.x_cleaned = x  
                    st.success('Données nettoyées avec succès')
            
            if st.session_state.x_cleaned is not None:
                st.markdown('<div class="sub-header">Données après nettoyage</div>', unsafe_allow_html=True)
                cleaned_df = pd.DataFrame(st.session_state.x_cleaned, columns=df.iloc[:, :-1].columns)
                st.dataframe(cleaned_df, use_container_width=True)

        with tab3:
            st.markdown('<div class="sub-header">Gestion des Doublons</div>', unsafe_allow_html=True)
        
            duplicate_count = df.duplicated().sum()
            
            if duplicate_count > 0:
                st.warning(f"⚠️ **{duplicate_count} doublons détectés** dans le dataset")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nombre de doublons", duplicate_count)
                with col2:
                    percentage = (duplicate_count / len(df)) * 100
                    st.metric("Pourcentage de doublons", f"{percentage:.1f}%")
                with col3:
                    st.metric("Lignes uniques", len(df) - duplicate_count)
                
                fig = px.pie(
                    values=[duplicate_count, len(df) - duplicate_count],
                    names=['Doublons', 'Lignes uniques'],
                    title=f"Distribution des doublons ({duplicate_count} doublons détectés)",
                    color_discrete_sequence=["#1F2BD6", "#78CCE4"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="sub-header"> Aperçu des doublons</div>', unsafe_allow_html=True)
                
                if st.checkbox("Afficher les doublons détectés", key="show_duplicates"):
                    duplicate_rows = df[df.duplicated()]
                    st.dataframe(duplicate_rows, use_container_width=True)
                    
                    if st.button("Télécharger la liste des doublons"):
                        csv = duplicate_rows.to_csv(index=False)
                        st.download_button(
                            label="Télécharger CSV",
                            data=csv,
                            file_name="doublons_detectes.csv",
                            mime="text/csv"
                        )
     
                st.markdown('<div class="sub-header">Options de gestion des doublons</div>', unsafe_allow_html=True)
                
                gestion_option = st.radio(
                    "Choisissez une action :",
                    ["Garder la première occurrence (recommandé)",
                    "Garder la dernière occurrence",
                    "Supprimer tous les doublons",
                    "Ne rien faire"],
                    horizontal=True
                )
                
                if gestion_option != "Ne rien faire" and st.button("Appliquer l'action"):
                        st.spinner("Traitement des doublons en cours...")
                        df_clean = df.copy()
                        
                        if gestion_option == "Garder la première occurrence (recommandé)":
                            df_clean = df.drop_duplicates(keep='first')
                            message = f"✅ {duplicate_count} doublons supprimés (première occurrence gardée)"
                            
                        elif gestion_option == "Garder la dernière occurrence":
                            df_clean = df.drop_duplicates(keep='last')
                            message = f"✅ {duplicate_count} doublons supprimés (dernière occurrence gardée)"
                            
                        elif gestion_option == "Supprimer tous les doublons":
                            df_clean = df.drop_duplicates(keep=False)
                            message = f"✅ {duplicate_count} doublons complètement supprimés"
                        
                        st.session_state.df = df_clean
                        
                        st.success(message)
                        st.info(f"**Nouvelle taille du dataset** : {len(df_clean)} lignes (précédemment {len(df)})")
                       
                        fig2 = px.bar(
                            x=['Avant nettoyage', 'Après nettoyage'],
                            y=[len(df), len(df_clean)],
                            title="Impact du nettoyage des doublons",
                            labels={'x': 'État', 'y': 'Nombre de lignes'},
                            color=['Avant nettoyage', 'Après nettoyage'],
                            color_discrete_sequence=["#1F2BD6", "#78CCE4"]
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        time.sleep(2)
                        
            else:
                st.success("✅ **Aucun doublon détecté** dans le dataset")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nombre total de lignes", len(df))
                with col2:
                    st.metric("Lignes uniques", len(df))

                st.info("🎉 Votre dataset est déjà propre ! Aucune action nécessaire.")
    else:
        st.warning("Veuillez d'abord charger un jeu de données.")

elif choice == 'Encodage des données':
        st.markdown('<div class="main-header">Encodage des données</div>', unsafe_allow_html=True)
        
        if st.session_state.df is not None and st.session_state.x_cleaned is not None:
            df = st.session_state.df
            x = st.session_state.x_cleaned

            categorical_cols = []
            for i, col in enumerate(df.iloc[:, :-1].columns):
                if df[col].dtype == 'object':
                    categorical_cols.append((i, col))
            
            if categorical_cols:
                st.markdown('<div class="sub-header">Colonnes catégorielles</div>', unsafe_allow_html=True)
                for i, col in categorical_cols:
                    st.write(f"- {col} (index {i})")

                col_to_encode = st.selectbox('Colonne à encoder:', 
                                            options=[col[0] for col in categorical_cols],
                                            format_func=lambda x: df.iloc[:, :-1].columns[x])
                
                encoding_method = st.selectbox('Méthode d\'encodage:',
                                            ['One-Hot Encoding',
                                            'Binary Encoding',
                                            'Label Encoding'])
            else:
                st.info("Aucune colonne catégorielle détectée dans le jeu de données.")
                col_to_encode = None

            if categorical_cols:
                if st.button('Encoder les Données', key='encode_data_button'):
                    with st.spinner('Encodage des données en cours...'):
                        if encoding_method == 'One-Hot Encoding':
                            encoder = compose.ColumnTransformer(
                                transformers=[('encoder', preprocessing.OneHotEncoder(), [col_to_encode])], 
                                remainder='passthrough')
                            x_encoded = encoder.fit_transform(x)
                            st.session_state.one_binary = 'One-Hot Encoding'
                        elif encoding_method == 'Binary Encoding':
                            encoder = compose.ColumnTransformer(
                                transformers=[('encoder', ce.BinaryEncoder(), [col_to_encode])], 
                                remainder='passthrough')
                            x_encoded = encoder.fit_transform(x)
                            st.session_state.one_binary = 'Binary Encoding'
                        elif encoding_method == 'Label Encoding':
                            encoder = preprocessing.LabelEncoder()
                            x_copy = x.copy()
                            x_copy[:, col_to_encode] = encoder.fit_transform(x[:, col_to_encode])
                            x_encoded = x_copy
                            st.session_state.one_binary = 'Label Encoding'
                        
                        time.sleep(2)
                        st.session_state.x_encoded = x_encoded
                        st.session_state.encoder = encoder
                        st.success('Données encodées avec succès')

                if st.session_state.x_encoded is not None:
                    st.markdown('<div class="sub-header">Aperçu des données encodées</div>', unsafe_allow_html=True)
                    try:
                        if isinstance(st.session_state.x_encoded, np.ndarray):
                            encoded_df = pd.DataFrame(st.session_state.x_encoded)
                        else:
                            encoded_df = pd.DataFrame(st.session_state.x_encoded.toarray())
                        st.dataframe(encoded_df, use_container_width=True)
                    except:
                        st.write(st.session_state.x_encoded)
            else:
                st.markdown('<div class="sub-header">Aucun encodage nécessaire</div>', unsafe_allow_html=True)
                st.info("Toutes les colonnes sont déjà numériques. Vous pouvez passer à l'étape de normalisation.")
                if st.button('Continuer sans encodage'):
                    st.session_state.x_encoded = st.session_state.x_cleaned
                    st.session_state.one_binary = 'None'
                    st.success('Données copiées pour la prochaine étape')

        else:
            st.warning("Veuillez d'abord nettoyer les données.")

elif choice == 'Normalisation des données':
    st.markdown('<div class="main-header">Normalisation des données</div>', unsafe_allow_html=True)
    
    if st.session_state.x_encoded is not None:
        x = st.session_state.x_encoded
        
        st.markdown('<div class="sub-header">Méthodes de normalisation</div>', unsafe_allow_html=True)
        normalization_method = st.selectbox('Choisissez une méthode:',
                                            ['StandardScaler',
                                             'MinMaxScaler',
                                             'RobustScaler',
                                             'MaxAbsScaler'])
        
        st.markdown("### Informations sur les méthodes")
        
        if normalization_method == 'StandardScaler':
            st.info("Normalise les données pour obtenir une moyenne de 0 et un écart-type de 1.")
        elif normalization_method == 'MinMaxScaler':
            st.info("Normalise les données dans une plage spécifique, généralement [0, 1].")
        elif normalization_method == 'RobustScaler':
            st.info("Utilise les statistiques robustes comme la médiane et les quartiles, moins sensible aux valeurs aberrantes.")
        elif normalization_method == 'MaxAbsScaler':
            st.info("Met à l'échelle chaque caractéristique par sa valeur absolue maximale.")
        
        if st.button('Normaliser les données', key='normalize_data_button'):
            with st.spinner('Normalisation des données en cours...'):
                if st.session_state.one_binary in ['One-Hot Encoding', 'Binary Encoding']:
                    x_to_normalize = x
                else:
                    x_to_normalize = x
                
                if normalization_method == 'StandardScaler':
                    scaler = preprocessing.StandardScaler()
                elif normalization_method == 'MinMaxScaler':
                    scaler = preprocessing.MinMaxScaler()
                elif normalization_method == 'RobustScaler':
                    scaler = preprocessing.RobustScaler()
                elif normalization_method == 'MaxAbsScaler':
                    scaler = preprocessing.MaxAbsScaler()

                if isinstance(x_to_normalize, np.ndarray):
                    x_normalized = scaler.fit_transform(x_to_normalize)
                else:
                    try:
                        x_normalized = scaler.fit_transform(x_to_normalize.toarray())
                    except:
                        x_normalized = scaler.fit_transform(x_to_normalize)
                
                time.sleep(2)
                st.session_state.x_normalized = x_normalized
                st.session_state.scaler = scaler
                st.success('Données normalisées avec succès')

        if st.session_state.x_normalized is not None:
            st.markdown('<div class="sub-header">Aperçu des données normalisées</div>', unsafe_allow_html=True)
            normalized_df = pd.DataFrame(st.session_state.x_normalized)
            st.dataframe(normalized_df, use_container_width=True)
            
            st.markdown('<div class="sub-header">Statistiques des données normalisées</div>', unsafe_allow_html=True)
            st.dataframe(normalized_df.describe(), use_container_width=True)
    else:
        st.warning("Veuillez d'abord encoder les données.")

elif choice == 'Matrice de corrélation':
    st.markdown('<div class="main-header">Matrice de corrélation</div>', unsafe_allow_html=True)
   
    if st.session_state.df is not None:
        df = st.session_state.df
        numeric_df = df.select_dtypes(include=np.number)
        
        if 'Profit' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['Profit'])
       
        if numeric_df.shape[1] > 1:
            correlation_matrix = numeric_df.corr()
            fig = px.imshow(correlation_matrix,
                          text_auto=True,
                          color_continuous_scale='RdBu_r',
                          title="Matrice de corrélation")
            st.plotly_chart(fig, use_container_width=True)
           
            st.markdown('<div class="sub-header">Valeurs de corrélation</div>', unsafe_allow_html=True)
            st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
           
            st.markdown('<div class="sub-header">Corrélations les plus fortes</div>', unsafe_allow_html=True)
           
            correlations = correlation_matrix.unstack().sort_values(ascending=False)
            # 
            high_corr = correlations[correlations < 1].head(5)
           
            for idx, corr in high_corr.items():
                st.write(f"**{idx[0]}** et **{idx[1]}**: {corr:.4f}")
        else:
            st.warning("Il n'y a pas assez de colonnes numériques pour calculer une matrice de corrélation.")
    else:
        st.warning("Veuillez d'abord charger un jeu de données.")

elif choice == 'Réduction de dimension':
    st.markdown('<div class="main-header">Réduction de dimension</div>', unsafe_allow_html=True)
    
    if st.session_state.x_normalized is not None:
        x = st.session_state.x_normalized
        st.markdown('<div class="sub-header">Choisissez une méthode de réduction</div>', unsafe_allow_html=True)
        reduction_method = st.selectbox('Choisissez une méthode:',
                                        ['Sans Réduction', 
                                         'PCA'])
        
        n_components = min(3, x.shape[1])
        
        if reduction_method == 'PCA':
            n_components = st.slider('Nombre de composantes', 
                                     min_value=2, 
                                     max_value=min(5, x.shape[1]), 
                                     value=min(3, x.shape[1]))

        if reduction_method == 'PCA' and st.button('Appliquer PCA', key='apply_pca_button'):
            with st.spinner('Réduction de dimension en cours...'):
                pca = PCA(n_components=n_components)
                x_pca = pca.fit_transform(x)
                
                st.session_state.x_reduced = x_pca
                st.session_state.reduction_method = 'PCA'
                st.session_state.pca = pca
                st.session_state.n_components = n_components 
                time.sleep(2) 
                st.success('Réduction PCA appliquée avec succès')

        if 'x_reduced' in st.session_state and st.session_state.reduction_method == 'PCA':
            x_pca = st.session_state.x_reduced
            pca = st.session_state.pca
            n_components = st.session_state.n_components 
            
            st.markdown('<div class="sub-header">Variance expliquée</div>', unsafe_allow_html=True)
            explained_variance = pca.explained_variance_ratio_
            total_variance = sum(explained_variance)
            
            fig = px.bar(
                x=[f'PC{i+1}' for i in range(len(explained_variance))],
                y=explained_variance,
                labels={'x': 'Composantes principales', 'y': 'Ratio de variance expliquée'},
                title=f"Variance expliquée par composante (Total: {total_variance:.2%})"
            )
            st.plotly_chart(fig, use_container_width=True)
 
            st.markdown('<div class="sub-header">Données après PCA</div>', unsafe_allow_html=True)
            pca_df = pd.DataFrame(
                x_pca,
                # 
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            st.dataframe(pca_df, use_container_width=True)
          
        elif reduction_method == 'Sans Réduction':
            st.markdown('<div class="sub-header">Aucune réduction appliquée</div>', unsafe_allow_html=True)
            st.info("Les données normalisées seront utilisées telles quelles pour l'entraînement du modèle.")
            if st.session_state.x_normalized is not None:
                normalized_df = pd.DataFrame(st.session_state.x_normalized)
                st.dataframe(normalized_df, use_container_width=True)
    else:
        st.warning("Veuillez d'abord normaliser les données dans l'onglet 'Normalisation'.")

elif choice == 'Entraînement du modèle':
    st.markdown('<div class="main-header">Entraînement du modèle</div>', unsafe_allow_html=True)
    
    if st.session_state.x_normalized is not None:
        if 'x_reduced' in st.session_state and st.session_state.reduction_method == 'PCA':
            x = st.session_state.x_reduced
            data_source = "PCA"
        else:
            x = st.session_state.x_normalized
            data_source = "normalisées"
            
        y = st.session_state.df.iloc[:, -1].values
        st.markdown('<div class="sub-header">Paramètres d\'entraînement</div>', unsafe_allow_html=True)
        
        if st.session_state.X_train is None:
            train_size = st.slider('Pourcentage de données d\'entraînement', 
                                   min_value=0, 
                                   max_value=100, 
                                   value=80, 
                                   step=5,
                                   help="Proportion des données utilisées pour l'entraînement du modèle")
            
            random_state = st.number_input('Valeur de random_state', 
                                          min_value=0, 
                                          max_value=100, 
                                          value=42,
                                          help="Initialisation du générateur de nombres aléatoires pour assurer la reproductibilité")
        else:
            st.info(f"Données déjà divisées en ensembles d'entraînement et de test.")
            
            if st.button('Réinitialiser la division des données'):
                st.session_state.X_train = None
                st.session_state.X_test = None
                st.session_state.y_train = None
                st.session_state.y_test = None
                st.rerun()

        if st.session_state.X_train is None:
            if st.button('Diviser les données et entraîner le modèle', key='train_model_button'):
                with st.spinner('Division des données et entraînement du modèle en cours...'):
                    test_size = 1 - (train_size / 100)
                    X_train, X_test, y_train, y_test = train_test_split(
                        x, y, test_size=test_size, random_state=random_state, shuffle=True
                    )
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    st.session_state.y_pred = model.predict(X_test)
                    
                    time.sleep(1)  
                    st.success('Modèle entraîné avec succès!')
        else:
            if st.session_state.model is None:
                if st.button('Entraîner le modèle', key='just_train_model_button'):
                    with st.spinner('Entraînement du modèle en cours...'):
                        model = LinearRegression()
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        
                        st.session_state.model = model
                        st.session_state.y_pred = model.predict(st.session_state.X_test)
                        
                        time.sleep(0.5) 
                        st.success('Modèle entraîné avec succès!')
            else:
                st.markdown('<div class="success-box">Le modèle a déjà été entraîné</div>', unsafe_allow_html=True)
        
        if st.session_state.X_train is not None:
            st.markdown(f'<div class="sub-header">Résumé de l\'entraînement</div>', unsafe_allow_html=True)
            
            st.markdown(f"**Source des données** : Données {data_source}")
            st.markdown(f"**Taille de l'ensemble d'entraînement** : {st.session_state.X_train.shape[0]} instances")
            st.markdown(f"**Taille de l'ensemble de test** : {st.session_state.X_test.shape[0]} instances")
            st.markdown(f"**Nombre de caractéristiques** : {st.session_state.X_train.shape[1]}")
            
            if st.session_state.model is not None:
                st.markdown("**Modèle** : Régression linéaire")
            
            tab1, tab2, tab3 = st.tabs(["Données d'entraînement", "Données de test", "Prédictions vs Réalité"])
            
            with tab1:
                train_df = pd.DataFrame(
                    st.session_state.X_train, 
                    columns=[f"Feature {i+1}" for i in range(st.session_state.X_train.shape[1])]
                )
                train_df["Target"] = st.session_state.y_train
                st.dataframe(train_df, use_container_width=True)
            
            with tab2:
                test_df = pd.DataFrame(
                    st.session_state.X_test, 
                    columns=[f"Feature {i+1}" for i in range(st.session_state.X_test.shape[1])]
                )
                test_df["Target"] = st.session_state.y_test
                if st.session_state.model is not None:
                    test_df["Prediction"] = st.session_state.y_pred
                st.dataframe(test_df, use_container_width=True)
            
            with tab3:
                if st.session_state.model is not None:
                    comparison_df = pd.DataFrame({
                        "Instance": range(1, len(st.session_state.y_test) + 1),
                        "Valeur Réelle": st.session_state.y_test,
                        "Prédiction": st.session_state.y_pred,
                        "Erreur": st.session_state.y_test - st.session_state.y_pred
                    })
                    
                    st.dataframe(comparison_df, use_container_width=True)
    else:
        st.warning("Veuillez d'abord normaliser les données.")

elif choice == 'Évaluation du modèle':
    st.markdown('<div class="main-header">Évaluation du modèle</div>', unsafe_allow_html=True)
    
    if (st.session_state.model is not None and st.session_state.y_pred is not None and
        st.session_state.y_test is not None):
        
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        model = st.session_state.model
        
        if y_test is not None and y_pred is not None:
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if np.all(y_test != 0) else np.nan
            
            st.markdown('<div class="sub-header">Métriques d\'évaluation</div>', unsafe_allow_html=True)
            
            metrics_data = {
                "Métrique": ["MAE", "MSE", "RMSE", "R²", "MAPE (%)"],
                "Valeur": [mae, mse, rmse, r2, f"{mape:.2f}" if not np.isnan(mape) else "N/A"],
                "Description": [
                    "Erreur absolue moyenne",
                    "Erreur quadratique moyenne",
                    "Racine de l'erreur quadratique moyenne",
                    "Coefficient de détermination",
                    "Erreur absolue moyenne en pourcentage"
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            if r2 < 0.3:
                interpretation = "Faible pouvoir prédictif"
                color = "red"
            elif r2 < 0.6:
                interpretation = "Pouvoir prédictif moyen"
                color = "orange"
            elif r2 < 0.8:
                interpretation = "Bon pouvoir prédictif"
                color = "blue"
            else:
                interpretation = "Excellent pouvoir prédictif"
                color = "green"
            
            st.info(f"**Interprétation du R² ({r2:.3f}) :** {interpretation}")
            
            st.markdown('<div class="sub-header">Paramètres du modèle de régression</div>', unsafe_allow_html=True)
            
            try:
                if 'x_reduced' in st.session_state and st.session_state.reduction_method == 'PCA':
                    n_components = st.session_state.n_components
                    feature_names = [f'PC{i+1}' for i in range(n_components)]
                    source_info = "composantes principales (PCA)"
                elif st.session_state.x_encoded is not None and st.session_state.one_binary in ['One-Hot Encoding', 'Binary Encoding']:
                    n_features = model.coef_.shape[0] if len(model.coef_.shape) > 0 else 1
                    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
                    source_info = "features après encodage"
                else:
                    df = st.session_state.df
                    feature_names = df.iloc[:, :-1].columns.tolist()
                    source_info = "variables originales"
                
                if hasattr(model, 'coef_'):
                    coefficients = model.coef_
                    intercept = model.intercept_
                    
                    if len(coefficients.shape) == 0: 
                        coeff_list = [coefficients]
                    elif len(coefficients.shape) == 1: 
                        coeff_list = coefficients.tolist()
                    else: 
                        coeff_list = coefficients[0] if coefficients.shape[0] == 1 else coefficients.tolist()
                    
                    if len(coeff_list) == len(feature_names):
                        coeff_df = pd.DataFrame({
                            'Variable': feature_names,
                            'Coefficient (pente)': [f"{c:.6f}" for c in coeff_list],
                            'Impact': ['Positif' if c > 0 else 'Négatif' for c in coeff_list]
                        })
                        
                        coeff_df['Abs_Coefficient'] = [abs(c) for c in coeff_list]
                        coeff_df = coeff_df.sort_values('Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1)
                        
                        st.write(f"**Variables utilisées :** {source_info}")
                        st.dataframe(coeff_df, use_container_width=True)
                        
                        st.markdown("##### Interprétation des coefficients :")
                        st.markdown("""
                        - **Coefficient positif** : Quand la variable augmente, la valeur prédite augmente
                        - **Coefficient négatif** : Quand la variable augmente, la valeur prédite diminue
                        - **Valeur absolue élevée** : Impact plus important sur la prédiction
                        """)
                        
                        fig = px.bar(
                            coeff_df,
                            x='Variable',
                            y=[float(c) for c in coeff_df['Coefficient (pente)']],
                            color=[float(c) for c in coeff_df['Coefficient (pente)']],
                            color_continuous_scale='RdYlGn',
                            title="Importance des variables (coefficients)",
                            labels={'y': 'Coefficient', 'x': 'Variable'}
                        )
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            coloraxis_showscale=True,
                            coloraxis_colorbar_title="Valeur"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.warning(f"Nombre de coefficients ({len(coeff_list)}) ne correspond pas au nombre de features ({len(feature_names)})")
                
                if hasattr(model, 'intercept_'):
                    st.markdown(f"##### **Ordonnée à l'origine (intercept) :** `{intercept:.6f}`")
                    st.caption("Valeur prédite lorsque toutes les variables indépendantes sont à zéro")
            except Exception as e:
                st.warning(f"Impossible d'afficher les coefficients : {str(e)}")
                st.write("Coefficients bruts :", model.coef_ if hasattr(model, 'coef_') else "Non disponible")   
        else:
            st.warning("Les données de test ou les prédictions ne sont pas disponibles.")
    else:
        st.warning("Veuillez d'abord entraîner un modèle.")

elif choice == 'Visualisation des prédictions':
    st.markdown('<div class="main-header">Visualisation des prédictions</div>', unsafe_allow_html=True)
    required_vars = ['model', 'y_pred', 'y_test']
    missing_vars = [var for var in required_vars if not hasattr(st.session_state, var) or st.session_state[var] is None]
    
    if not missing_vars and st.session_state.y_test is not None and st.session_state.y_pred is not None:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        if y_test.shape[0] == 0 or y_pred.shape[0] == 0:
            st.warning("Les données de test ou les prédictions sont vides.")
        elif y_test.shape[0] != y_pred.shape[0]:
            st.warning(f"Les dimensions ne correspondent pas: y_test={y_test.shape[0]}, y_pred={y_pred.shape[0]}")
        else:
            st.markdown('<div class="sub-header">Options de visualisation</div>', unsafe_allow_html=True)
            plot_type = st.selectbox(
                "Type de graphique",
                options=["Scatter Plot", "Courbes", "Barres"]
            )
            results_df = pd.DataFrame({
                "Réel": y_test,
                "Prédit": y_pred,
                "Résidus": y_test - y_pred,
                "Erreur Relative (%)": np.where(
                    y_test != 0, 
                    abs((y_test - y_pred) / y_test) * 100, 0)
            })
            
            results_df["Échantillon"] = range(1, len(y_test) + 1)
            
            if len(y_test) > 20:
                num_samples = st.slider(
                    "Nombre d'échantillons à afficher",
                    min_value=5,
                    max_value=min(len(y_test), 100),
                    value=20
                )
                results_sample = results_df.sample(num_samples).sort_index()
            else:
                results_sample = results_df
            st.markdown('<div class="sub-header">Graphique</div>', unsafe_allow_html=True)
            
            if plot_type == "Scatter Plot":
                fig = px.scatter(
                    results_df,
                    x="Réel",
                    y="Prédit",
                    hover_data=["Échantillon", "Résidus", "Erreur Relative (%)"],
                    title="Valeurs Réelles vs Prédites",
                    color="Erreur Relative (%)",
                    color_continuous_scale="RdYlGn_r"
                )
                
                fig.add_shape(
                    type="line",
                    line=dict(dash="dash", color="gray"),
                    x0=results_df["Réel"].min(),
                    y0=results_df["Réel"].min(),
                    x1=results_df["Réel"].max(),
                    y1=results_df["Réel"].max()
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "Courbes":
                fig = px.line(
                    results_sample,
                    x="Échantillon",
                    y=["Réel", "Prédit"],
                    title="Comparaison des Valeurs Réelles et Prédites",
                    markers=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "Barres":
                fig = px.bar(
                    results_sample,
                    x="Échantillon",
                    y=["Réel", "Prédit"],
                    barmode="group",
                    title="Comparaison des Valeurs Réelles et Prédites"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            st.markdown('<div class="sub-header">Tableau des résultats</div>', unsafe_allow_html=True)
            st.dataframe(results_sample.style.format({
                "Réel": "{:.2f}",
                "Prédit": "{:.2f}",
                "Résidus": "{:.2f}",
                "Erreur Relative (%)": "{:.2f}%"
            }), use_container_width=True)

            st.markdown('<div class="sub-header">Histogramme des residus pour verifier la normalite des residus</div>', unsafe_allow_html=True)
            fig = px.histogram(
                results_df,
                x="Résidus",
                nbins=10,  
                title="Histogramme des résidus pour vérifier la normalité des résidus",
                labels={"Résidus": "Résidus", "count": "Fréquence"},
                opacity=0.5, 
                color_discrete_sequence=['skyblue'],  
                marginal=None  
            )
            fig.update_traces(
                marker=dict(
                    line=dict(color='black', width=1) 
                )
            )
            residus = results_df["Résidus"].values
            x_kde = np.linspace(residus.min(), residus.max(), 100)
            kde = stats.gaussian_kde(residus)
            y_kde = kde(x_kde)
            bin_width = (residus.max() - residus.min()) / 10
            y_kde_scaled = y_kde * len(residus) * bin_width
            fig.add_scatter(
                x=x_kde,
                y=y_kde_scaled,
                mode='lines',
                name='Densité (KDE)',
                line=dict(color='red', width=2)
            )

            fig.update_layout(
                xaxis_title="Résidus",
                yaxis_title="Fréquence",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="sub-header">Graphique des residus pour verifier l\'independance et l\'homoscedasiticite des redidus</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 6))  
            ax.scatter(y_pred, results_df["Résidus"], alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel("Valeurs prédites")
            ax.set_ylabel("Résidus")
            ax.set_title("Graphique des résidus pour vérifier l'indépendance des résidus")
            ax.grid(True, alpha=0.3, linestyle='--')
                
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('<div class="sub-header">QQ-plot des residus</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(results_df["Résidus"], dist="norm", plot=ax)

            ax.set_title('QQ-plot des résidus pour vérifier la normalité')
            ax.set_xlabel('Quantiles théoriques (distribution normale)')
            ax.set_ylabel('Quantiles observés (résidus)')
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        if missing_vars:
            st.warning(f"Veuillez d'abord entraîner un modèle. Éléments manquants: {', '.join(missing_vars)}")
        elif st.session_state.y_test is None or st.session_state.y_pred is None:
            st.warning("Les données de test ou les prédictions ne sont pas disponibles.")

elif choice == 'Prédiction sur nouvelles données':
    st.markdown('<div class="main-header">Prédiction sur nouvelles données</div>', unsafe_allow_html=True)
    if st.session_state.model is not None:
        df = st.session_state.df
        feature_columns = df.iloc[:, :-1].columns.tolist()
        target_column = df.iloc[:, -1].name
        
        st.markdown('Entrez de nouvelles valeurs', unsafe_allow_html=True)
        new_data = {}
        
        for col in feature_columns:
            col_type = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(col_type):
                default_val = float(df[col].mean())
                new_data[col] = st.number_input(
                    f"{col}:",
                    value=default_val,
                    format="%.2f"
                )
            elif pd.api.types.is_categorical_dtype(col_type) or df[col].dtype == 'object':
                unique_values = df[col].unique().tolist()
                new_data[col] = st.selectbox(
                    f"{col}:",
                    options=unique_values
                )
            else:
                new_data[col] = st.text_input(f"{col}:")
        predict_button = st.button('Effectuer une prédiction')
        
        if predict_button:
            try:
                with st.spinner('Préparation des données et prédiction en cours...'):
                    input_df = pd.DataFrame([new_data])
                    input_df = input_df.reindex(columns=feature_columns, fill_value=None)
                    transformed_data = input_df.copy()
                    
                    # imputation des donnees si necessaire
                    if hasattr(st.session_state, 'imputer') and st.session_state.imputer is not None:
                        numeric_cols = transformed_data.select_dtypes(include=np.number).columns
                        if len(numeric_cols) > 0:
                            transformed_data[numeric_cols] = st.session_state.imputer.transform(transformed_data[numeric_cols])
                    
                    # encodage des donnees si necessaire
                    if hasattr(st.session_state, 'encoder') and st.session_state.encoder is not None:
                        try:
                            if isinstance(st.session_state.encoder, compose.ColumnTransformer):
                                transformed_data = st.session_state.encoder.transform(transformed_data)
                            else:
                                for col in transformed_data.columns:
                                    if transformed_data[col].dtype == 'object':
                                        transformed_data[col] = st.session_state.encoder.transform(transformed_data[col])
                            
                            if hasattr(transformed_data, 'toarray'):
                                transformed_data = transformed_data.toarray()
                        except Exception as e:
                            st.error(f"Erreur lors de l'encodage des données : {e}")
                            raise
                    
                    # normalisation des donnees si necessaire
                    if hasattr(st.session_state, 'scaler') and st.session_state.scaler is not None:
                        try:
                            transformed_data = st.session_state.scaler.transform(transformed_data)
                        except Exception as e:
                            st.error(f"Erreur lors de la normalisation des données : {e}")
                            raise
                    
                    # application de la reduction de dimension  si necessaire
                    if hasattr(st.session_state, 'pca') and st.session_state.pca is not None and st.session_state.reduction_method == 'PCA':
                        try:
                            transformed_data = st.session_state.pca.transform(transformed_data)
                        except Exception as e:
                            st.error(f"Erreur lors de l'application de la PCA : {e}")
                            raise
              
                    # conversion garantie en array numpy
                    if isinstance(transformed_data, pd.DataFrame):
                        transformed_data = transformed_data.values
                    elif isinstance(transformed_data, list):
                        transformed_data = np.array(transformed_data)
                    elif hasattr(transformed_data, 'toarray'):
                        transformed_data = transformed_data.toarray()    
                    # si les données ont une forme de dimension 1, on les ajuste
                    if len(transformed_data.shape) == 1:
                        transformed_data = transformed_data.reshape(1, -1)
                    
                    # prediction
                    prediction = st.session_state.model.predict(transformed_data)
                    st.markdown(' ', unsafe_allow_html=True)
                    st.markdown(f"### Prédiction pour {target_column}")
                    st.markdown(f"## {prediction[0]:.2f}")
                    st.markdown(' ', unsafe_allow_html=True)
                    
                    # intervalle de confiance a 95%
                    rmse = np.sqrt(metrics.mean_squared_error(st.session_state.y_test, st.session_state.y_pred))
                    lower_bound = prediction[0] - 1.96 * rmse
                    upper_bound = prediction[0] + 1.96 * rmse
                    
                    st.info(f"Intervalle de confiance à 95% : [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de la prédiction : {e}")
        else:
            st.markdown("""
             👈 Entrez les valeurs pour les variables indépendantes, puis cliquez sur 'Effectuer une prédiction'.
            
            Le modèle appliquera les mêmes transformations que pendant l'entraînement et retournera une prédiction.
            """)
    else:
        st.warning("Veuillez d'abord entraîner un modèle.")

