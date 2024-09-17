# 🎮 Qu'est ce que G-Search ? 🎮

Vous êtes vous déjà retrouvés à jouer sur votre console/PC/autre et sentir un peu de lassitude ? 

Vous dire "tiens, ce jeu est pas mal, mais j'aimerais bien jouer à autre chose" ?

**_G-Search est là pour ça !_**



# 🎯 Objectifs 🎯 

- À partir d'un nom de jeu, avoir des recommandations de jeux qui pourraient nous interesser disponibles sur la plateforme sur laquelle nous sommes
- Pouvoir accéder à une base de données des jeux actualisée régulièrement pour pouvoir choisir un jeu qui vient de sortir, en fonction de nos goûts, de leur popularité,etc...
- Avoir la visibilité du jeu (jaquette, développeurs, durée de vie du jeu, etc...) mais aussi de voir comment il fonctionne "en vrai" par une vidéo teasing ou un test dans les médias
- Avoir la possibilité de l'acheter en ligne, via une plateforme de vente de jeu ou un distributeur
- Pouvoir faire la recherche depuis son ordinateur MAIS AUSSI depuis son téléphone

  ...Et d'autres possibilités à venir !

# 📆  Evolution du projet 📆  

#### Update au 8/9/24 :
- Créer un notebook permettant un df clean - ✔️ (format Parquet) [Cleaning](http://github.com/WildAlex37/g_search_lite/blob/main/G_search_lite_cleaning.ipynb)
- Créer un notebook unique pour la data exploration ✔️ (continuer pour trouver des insights) [Exploration](http://github.com/WildAlex37/g_search_lite/blob/main/G_search_lite_exploration.ipynb)
- Créer un notebook pour le ML ✔️ (G_search_lite_ML.ipynb) [ML](http://github.com/WildAlex37/g_search_lite/blob/main/G_search_lite_ML.ipynb)
- Mettre en place le .py avec code Streamlit ⏳

#### Update 9/9/24 :
- Travailler sur l'API ⏳
- tester différents modèles de ML ⏳

#### Update au 11/09/24 : 
- Difficultés sur le meilleur modèle ML ⏳
- Système de reco mis en place, mais pas assez pertinent : travailler le poids des features ⏳

#### Update au 13/09/24 : 
- Modéliser une recherche avec recommandation pour tout nom de jeu incomplet ou approximatif ✔️

<a href="https://ibb.co/GdtHJ0N"><img src="https://i.ibb.co/grP473h/test-g-search.png" alt="test-g-search" border="0"></a>
  
- Utiliser l'API avec une request redondante (hebdomadaire dans l'idéal) ⏳
- filtrage des données selon la plateforme en AMONT du ML ✔️
- Reflexion autour du meilleur train pour le modèle ⏳
