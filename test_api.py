import streamlit as st
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# Constants
CLIENT_ID = "xxxxx"
CLIENT_SECRET = "yyyyy"
ENDPOINT = "https://api.igdb.com/v4/games"

class GameRecommender:
    def __init__(self):
        self.df = None
        self.token = None
        self.features = ['rating', 'rating_count', 'total_rating', 'total_rating_count']
        self.genre_features = None

    def get_access_token(self):
        """Obtenir le token d'accès pour l'API IGDB"""
        auth_url = "https://id.twitch.tv/oauth2/token"
        auth_data = {
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'grant_type': 'client_credentials'
        }
        response = requests.post(auth_url, data=auth_data)
        return response.json()['access_token']

    def fetch_games_data(self):
        """Récupérer les données des jeux via l'API IGDB avec pagination"""
        if not self.token:
            self.token = self.get_access_token()

        headers = {
            'Client-ID': CLIENT_ID,
            'Authorization': f'Bearer {self.token}',
            'Accept': 'application/json'
        }

        all_games = []
        offset = 0
        batch_size = 500
        total_games = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        while True:
            body = f"""
fields name, first_release_date, genres.name, platforms.name, summary, cover.url, rating, rating_count, total_rating, total_rating_count;
where platforms = (167, 130, 169, 6, 14) & rating != null & total_rating >=50 ;
limit {batch_size};
offset {offset};
"""

            response = requests.post(ENDPOINT, headers=headers, data=body)

            if response.status_code != 200:
                st.error(f"Erreur API: {response.status_code} - {response.text}")
                break

            batch_games = response.json()

            if not batch_games:
                break

            all_games.extend(batch_games)
            total_games += len(batch_games)

            status_text.text(f"Chargement des jeux... {total_games} jeux récupérés")
            progress_bar.progress(min(total_games / 3000, 1.0))

            if len(batch_games) < batch_size:
                break

            offset += batch_size
            time.sleep(0.25)

        progress_bar.empty()
        status_text.empty()
        return all_games

    def process_games_data(self, data):
        """Transformer les données brutes en DataFrame"""
        if not data:
            return pd.DataFrame()

        processed_data = []
        for game in data:
            processed_game = {
                'name': game.get('name', ''),
                'release_date': datetime.fromtimestamp(game['first_release_date']).strftime('%Y-%m-%d') if 'first_release_date' in game else None,
                'genres': ', '.join([g['name'] for g in game.get('genres', [])]),
                'platforms': ', '.join([p['name'] for p in game.get('platforms', [])]),
                'summary': game.get('summary', ''),
                'cover_url': game.get('cover', {}).get('url', '') if game.get('cover') else '',
                'rating': game.get('rating', 0),
                'rating_count': game.get('rating_count', 0),
                'total_rating': game.get('total_rating', 0),
                'total_rating_count': game.get('total_rating_count', 0)
            }
            processed_data.append(processed_game)

        df = pd.DataFrame(processed_data)
        df = df.drop_duplicates(subset='name')

        # Prepare genre features after processing
        df = self._prepare_genre_features(df)
        return df

    def _prepare_genre_features(self, df):
        """Prépare les features basées sur les genres"""
        genres = df['genres'].str.get_dummies(sep=', ')
        self.genre_features = genres.columns.tolist()
        df = pd.concat([df, genres], axis=1)
        return df

    def get_recommendations(self, game_name, platform, n_recommendations=5):
        """Obtenir des recommandations de jeux similaires"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        platform_mapping = {
            'Xbox': 'Xbox Series X|S',
            'PS5': 'PlayStation 5',
            'Nintendo Switch': 'Nintendo Switch',
            'PC': 'PC (Microsoft Windows)',
            'MAC': 'Mac'
        }

        platform_name = platform_mapping.get(platform, platform)
        platform_games = self.df[self.df['platforms'].str.contains(platform_name, case=False, na=False)]

        if platform_games.empty:
            return pd.DataFrame()

        X = platform_games[self.features].fillna(0)
        y = platform_games['rating']

        if len(X) < n_recommendations + 1:
            return platform_games

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Utiliser XGBoost au lieu de KNN
        model = xgb.XGBRegressor(objective='reg:squarederror', use_label_encoder=False, eval_metric='rmse')
        model.fit(X_scaled, y)

        game_idx = platform_games[platform_games['name'].str.lower() == game_name.lower()].index
        if len(game_idx) == 0:
            return pd.DataFrame()

        game_features = X_scaled[game_idx[0]].reshape(1, -1)
        predictions = model.predict(X_scaled)

        platform_games['predicted_rating'] = predictions
        recommendations = platform_games.nlargest(n_recommendations + 1, 'predicted_rating')
        recommendations = recommendations[recommendations['name'] != game_name]

        return recommendations

class RecommenderEvaluator:
    def __init__(self, recommender):
        self.recommender = recommender

    def evaluate_xgboost(self, param_grid, test_size=0.2):
        """Évalue la performance de XGBoost avec différentes valeurs d'hyperparamètres"""
        X = self.recommender.df[self.recommender.features].fillna(0)
        y = self.recommender.df['rating']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = xgb.XGBRegressor(objective='reg:squarederror', use_label_encoder=False, eval_metric='rmse')
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }

    def compare_similarity_metrics(self, sample_size=1000):
        """Compare différentes métriques de similarité"""
        sample_df = self.recommender.df.sample(n=min(sample_size, len(self.recommender.df)), random_state=42)

        # Ensure genre features are present in the sample DataFrame
        if self.recommender.genre_features is None:
            st.error("Les colonnes de genre n'ont pas été correctement préparées.")
            return {}

        if not all(feature in sample_df.columns for feature in self.recommender.genre_features):
            st.error("Les colonnes de genre ne sont pas présentes dans le DataFrame échantillonné.")
            return {}

        numeric_features = sample_df[self.recommender.features].fillna(0)
        genre_features = sample_df[self.recommender.genre_features]

        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_features)

        similarity_metrics = {
            'Euclidean (numeric)': 1 / (1 + np.linalg.norm(numeric_scaled[:, None] - numeric_scaled, axis=2)),
            'Cosine (numeric)': cosine_similarity(numeric_scaled),
            'Genre-based': cosine_similarity(genre_features),
            'Hybrid': (cosine_similarity(numeric_scaled) + cosine_similarity(genre_features)) / 2
        }

        return similarity_metrics

    def plot_similarity_distributions(self, similarity_metrics):
        """Visualise la distribution des similarités pour chaque métrique"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Distribution des scores de similarité par métrique')

        for (title, matrix), ax in zip(similarity_metrics.items(), axes.ravel()):
            similarities = matrix[np.triu_indices_from(matrix, k=1)]
            sns.histplot(similarities, bins=50, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Score de similarité')
            ax.set_ylabel('Fréquence')

        plt.tight_layout()
        return fig

    def evaluate_recommendations(self, n_samples=100, k=5):
        """Évalue la qualité des recommandations"""
        results = []
        sample_games = self.recommender.df.sample(n=n_samples, random_state=42)

        for platform in ['PlayStation 5', 'Xbox Series X|S', 'Nintendo Switch', 'PC (Microsoft Windows)', 'Mac']:
            platform_games = self.recommender.df[self.recommender.df['platforms'].str.contains(platform, case=False, na=False)]

            if len(platform_games) < k + 1:
                continue

            X = platform_games[self.recommender.features].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = xgb.XGBRegressor(objective='reg:squarederror', use_label_encoder=False, eval_metric='rmse')
            model.fit(X_scaled, platform_games['rating'])

            for _, game in sample_games.iterrows():
                if platform not in game['platforms']:
                    continue

                game_features = scaler.transform(game[self.recommender.features].fillna(0).values.reshape(1, -1))
                predictions = model.predict(X_scaled)

                platform_games['predicted_rating'] = predictions
                recommendations = platform_games.nlargest(k + 1, 'predicted_rating')
                recommendations = recommendations[recommendations['name'] != game['name']]

                genre_overlap = []
                rating_diff = []

                game_genres = set(game['genres'].split(', '))

                for _, rec_game in recommendations.iterrows():
                    rec_genres = set(rec_game['genres'].split(', '))
                    genre_overlap.append(len(game_genres.intersection(rec_genres)) / len(game_genres))
                    rating_diff.append(abs(game['rating'] - rec_game['rating']))

                results.append({
                    'platform': platform,
                    'game': game['name'],
                    'avg_genre_overlap': np.mean(genre_overlap),
                    'avg_rating_diff': np.mean(rating_diff)
                })

        return pd.DataFrame(results)

def main():
    st.title("Système de Recommandation de Jeux")

    # Initialiser le recommender
    recommender = GameRecommender()

    # Créer les onglets
    tab1, tab2 = st.tabs(["Recommandations", "Évaluation"])

    with tab1:
        # Interface de recommandation
        if st.button("Récupérer la base de données des jeux"):
            with st.spinner("Chargement des données..."):
                data = recommender.fetch_games_data()
                if data:
                    recommender.df = recommender.process_games_data(data)
                    st.success(f"Données chargées avec succès! {len(recommender.df)} jeux trouvés.")
                    st.session_state['game_data'] = recommender.df
                else:
                    st.error("Erreur lors du chargement des données.")

        if 'game_data' in st.session_state:
            recommender.df = st.session_state['game_data']

        platforms = ["Xbox", "PS5", "Nintendo Switch", "PC", "MAC"]
        selected_platform = st.selectbox("Choisissez votre plateforme", platforms)

        if recommender.df is not None and not recommender.df.empty:
            genres = recommender.df['genres'].str.split(', ').explode().unique()
            genres = [g for g in genres if pd.notna(g)]
            genres = sorted(list(set(genres)))
            selected_genre = st.selectbox("Filtrer par genre (facultatif)", ['Tous les genres'] + genres)

            filtered_df = recommender.df
            if selected_genre != 'Tous les genres':
                filtered_df = filtered_df[filtered_df['genres'].str.contains(selected_genre, na=False)]

            game_names = filtered_df['name'].tolist()
            search_term = st.text_input("Rechercher un jeu")

            if search_term:
                suggestions = [name for name in game_names if search_term.lower() in name.lower()]
                if suggestions:
                    selected_game = st.selectbox("Suggestions de jeux", suggestions)

                    if st.button("Obtenir des recommandations"):
                        recommendations = recommender.get_recommendations(selected_game, selected_platform)

                        if not recommendations.empty:
                            st.subheader("Jeux recommandés :")
                            for _, game in recommendations.iterrows():
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    if game['cover_url']:
                                        st.image(f"https:{game['cover_url']}", width=100)
                                with col2:
                                    st.write(f"**{game['name']}**")
                                    st.write(f"Note: {game['rating']:.1f}/100 ({game['rating_count']} votes)")
                                    if game['summary']:
                                        st.write(game['summary'])
                                st.write("---")
                        else:
                            st.warning("Aucune recommandation trouvée pour ce jeu sur cette plateforme.")
                else:
                    st.warning("Aucun jeu trouvé correspondant à votre recherche.")

    with tab2:
        # Interface d'évaluation
        if recommender.df is not None and not recommender.df.empty:
            evaluator = RecommenderEvaluator(recommender)

            st.header("1. Performance de XGBoost")
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
            xgboost_results = evaluator.evaluate_xgboost(param_grid)

            st.write("Métriques d'évaluation pour XGBoost:")
            st.dataframe(pd.DataFrame([xgboost_results]).round(4))

            st.header("2. Comparaison des métriques de similarité")
            similarity_metrics = evaluator.compare_similarity_metrics()
            if similarity_metrics:
                fig = evaluator.plot_similarity_distributions(similarity_metrics)
                st.pyplot(fig)

            st.header("3. Qualité des recommandations")
            recommendation_results = evaluator.evaluate_recommendations()

            st.write("Statistiques moyennes par plateforme:")
            platform_stats = recommendation_results.groupby('platform').agg({
                'avg_genre_overlap': ['mean', 'std'],
                'avg_rating_diff': ['mean', 'std']
            }).round(3)
            st.dataframe(platform_stats)

            st.header("Conclusions")
            st.write("""
            1. La performance de XGBoost varie selon les paramètres :
               - Les résultats montrent l'impact des hyperparamètres
               - L'erreur (RMSE) et le R² indiquent la qualité des prédictions

            2. Comparaison des métriques de similarité :
               - La similarité basée sur les notes (Euclidean, Cosine)
               - La similarité basée sur les genres
               - L'approche hybride combinant les deux

            3. Qualité des recommandations par plateforme :
               - Chevauchement des genres entre jeux recommandés
               - Différence moyenne des notes
               - Performance relative selon les plateformes
            """)
        else:
            st.warning("Veuillez d'abord charger les données dans l'onglet Recommandations.")

if __name__ == "__main__":
    main()
