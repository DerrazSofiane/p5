import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import sklearn.cluster as cluster
from dotenv import load_dotenv
from plotly.offline import iplot
from pywaffle import Waffle
from scipy.cluster.hierarchy import dendrogram
from sklearn import decomposition
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}

load_dotenv()


def assign_frequency(frequency):
    """
    Fonction permettant d'attribuer un score de fréquence

    Entrée :
    - fréquence - commande passée, int

    Sortie :
    - F - score de fréquence, int

    """

    if frequency >= 7:
        return 4
    elif frequency >= 4:
        return 3
    elif frequency >= 2:
        return 2
    else:
        return 1


def convert_to_dt(dataframe, columns, dt_format=None):
    """
    Fonction prenant en compte le nom du dataframe et les colonnes de date pour la conversion au format date

    Entrée :
    - Dataframe

    Sortie :
    - None (Convertit le format de la colonne en datetime)

    """
    for column in columns:
        dataframe[column] = pd.to_datetime(dataframe[column],
                                           format=dt_format).dt.date


def k_means_func(dataframe, n_clusters):
    """
    Fonction permettant de calculer la somme des erreurs quadratiques pour un nombre donné de clusters.

    Entrées :
    - dataframe - dataframe avec des données normalisées
    - n_clusters - nombre de clusters

    Sortie :
    - sse - somme des erreurs quadratiques

    """
    k_means = cluster.KMeans(n_clusters=n_clusters, random_state=1)
    k_means.fit(dataframe)

    return k_means.inertia_


def kmean_time_stability(dataframe, recency_col, n_clusters, ari=True):
    """
    Affiche la stabilité dans le temps d'un modèle non supervisé
    TODO : Améliorer la flexibilité de l'algorithme

    Entrée :
    - dataframe - unscaled dataframe
    - recency_col - str, la colonne récence
    - n_clusters - int, nombre de clusters
    - ari - bool, choix entre ARI ou AMI

    Sortie :
    - Liste de score
    """
    np.seterr(all='ignore')

    date_init = 365
    date_slipping = 0
    date_lim = date_init
    date_max = dataframe[recency_col].max()

    score = []

    first_year = dataframe[dataframe[recency_col] <= date_init]
    first_year_scaled = StandardScaler().fit_transform(first_year)
    first_year_clustered = cluster.KMeans(n_clusters=n_clusters)
    first_year_clustered.fit(first_year_scaled)

    while date_lim < date_max:

        add_next_month = dataframe[(dataframe[recency_col] >= date_slipping)
                                   & (dataframe[recency_col] <= date_lim)]
        add_next_month_scaled = StandardScaler().fit_transform(add_next_month)
        y_pred = first_year_clustered.predict(add_next_month_scaled)
        y_label = first_year_clustered.fit(add_next_month_scaled).labels_

        if ari == True:
            score.append(adjusted_rand_score(y_pred, y_label))
        else:
            score.append(adjusted_mutual_info_score(y_pred, y_label))

        date_lim += 30
        date_slipping += 30

    return score[::-1]


def pca_clusters(dataframe, algorithm, args, kwds):
    """
    Fonction permettant de générer une projection PCA de clusters

    Entrée :
    - dataframe - unscaled dataframe
    - algorithm - algorithme de clustering
    - args - arguments de l'algorithme, liste
    - kwds - paramètres de l'algorithme, dictionnaire

    Sortie :
    - Aucun (nuage de points 3D interactif)
    """
    start_time = time.time()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    labels = algorithm(*args, **kwds).fit_predict(scaled_data)
    pca = decomposition.PCA(n_components=2).fit(scaled_data)
    X_projected = pca.transform(scaled_data)
    pca_df = pd.DataFrame(X_projected)
    pca_df['cluster'] = labels
    pca_df.columns = ['x1', 'x2', 'cluster']
    end_time = time.time()
    sns.scatterplot(data=pca_df,
                    x='x1',
                    y='x2',
                    hue='cluster',
                    legend="full",
                    alpha=0.7).set_title('Clusters trouvés par {}'.format(
        str(algorithm.__name__)))
    print(f"Le clustering a pris {round(end_time - start_time)}s")


def plot_dendrogram(model, **kwargs):
    """
    Fonction établissant une matrice de liaison et trace ensuite le dendrogramme.
    Source : https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

    Entrée :
    - model - fct, algorithme de clustering
    - **kwargs - paramètres de l'algorithme

    Sortie :
    - Aucun (dendrogramme)
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_map(dataframe,
             title,
             lower_bound,
             upper_bound,
             metric,
             maker_size=3,
             is_sub_segment=False):
    """
    Fonction de visualisation de données géographiques de métriques démographiques.

    Entrée :
    - dataframe - dataframe avec la feature target
        (métrique ; champ de code couleur nécessaire si sous_segment à visualiser), df
    - title - texte à afficher comme titre du graphique, str
    - lower_bound - seuil inférieur de l'échelle de couleurs, int
    - upper_bound - seuil supérieur de l'échelle de couleurs, int
    - metric - feature / métrique kpi à visualiser, str
    - is_sub_segment - booléen,
        si "True" : le sous-segment sera visualisé avec un code couleur,
        si "False", valeur conforme à la couleur
    - marker_size - taille du marqueur, int

    Sortie :
    - Visualisation des données géographiques, plot

    """

    if is_sub_segment is True:
        dict_marker = dict(
            size=maker_size,
            color=dataframe.color,
        )
    else:
        dict_marker = dict(size=maker_size,
                           color=dataframe[metric],
                           showscale=True,
                           colorscale=[[0, 'blue'], [1, 'red']],
                           cmin=lower_bound,
                           cmax=upper_bound)

    data_geo = [
        go.Scattermapbox(lon=dataframe['geolocation_lng'],
                         lat=dataframe['geolocation_lat'],
                         marker=dict_marker)
    ]

    layout = dict(title=title,
                  showlegend=False,
                  mapbox=dict(
                      accesstoken=os.getenv("MAPBOX_TOKEN"),
                      center=dict(lat=-23.5, lon=-46.6),
                      bearing=10,
                      pitch=0,
                      zoom=2,
                  ))
    fig = dict(data=data_geo, layout=layout)
    iplot(fig, validate=False)


def plot_waffle_chart(dataframe, metric, agg, title_txt, group='sub_segment'):
    """
    Fonction permettant de créer un graphique en forme de gaufre.
    La visualisation montre comment les sous-segments de clients sont répartis selon des métriques définies.

    Entrée :
    - dataframe
    - métrique - feature/ métrique kpi à visualiser
    - agg - méthode d'agrégation
    - title_txt - texte à afficher comme titre du graphique

    Sortie :
    - Un délicieux graphique gaufré

    """
    data_revenue = dict(
        round(dataframe.groupby(group).agg({metric: agg}))[metric])

    plt.figure(FigureClass=Waffle,
               rows=5,
               columns=10,
               values=data_revenue,
               labels=[f"{k, v}" for k, v in data_revenue.items()],
               legend={
                   'loc': 'lower left',
                   'bbox_to_anchor': (1, 0)
               },
               figsize=(15, 7))

    plt.title(title_txt)


def rfm_assiner(dataframe):
    """
    TODO : Rendre la fonction flexible

    Fonction permettant d'attribuer des classes RFM selon les conditions.

    Entrée :
    - dataframe - dataframe contenant le score RFM et la clé du segment RFM.

    Sortie :
    - renvoie une classe de segment RFM (str)

    """
    if (int(dataframe['segment_RFM']) >= 434) or (dataframe['score_rfm'] >= 9):
        return 'Meilleur client'
    elif (dataframe['score_rfm'] >= 8) and (dataframe['M'] == 4):
        return 'Dépensier'
    elif (dataframe['score_rfm'] >= 6) and (dataframe['F'] >= 2):
        return 'Fidèle'
    elif (int(dataframe['segment_RFM']) >= 231) or (dataframe['score_rfm'] >=
                                                    6):
        return 'Fidélité potentielle'
    elif ((int(dataframe['segment_RFM']) >= 121) and
          (dataframe['R'] == 1)) or dataframe['score_rfm'] == 5:
        return 'Presque perdu'
    elif (dataframe['score_rfm'] >= 4) and (dataframe['R'] == 1):
        return 'En hibernation'
    else:
        return 'Client perdu'


def rfm_iso_scatter(dataframe, x, y, z):
    """
    Fonction permettant de générer un nuage de points en 3D

    Entrée :
    - dataframe
    - x, y, z - 3 features pour les trois coordonnées

    Sortie :
    - Aucun (nuage de points 3D interactif)
    """
    x = dataframe[x]
    y = dataframe[dataframe[y] < 5][y]
    z = dataframe[dataframe[z] < 4000][z]

    fig = go.Figure(data=[
        go.Scatter3d(x=x,
                     y=y,
                     z=z,
                     mode='markers',
                     marker=dict(
                         size=1, color=y, colorscale='thermal', opacity=0.8))
    ])

    fig.update_layout(scene=dict(xaxis_title='Récence',
                                 yaxis_title='Fréquence',
                                 zaxis_title='Montant'),
                      width=700,
                      margin=dict(r=20, b=10, l=10, t=10))

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def subst_mean(dataframe, columns):
    """
    La fonction prend le nom du dataframe et la liste des colonnes à substituer,
    les NaN seront remplies avec la moyenne.

    Entrée :
    - Dataframe

    Sortie :
    - None (Convertit le format de la colonne en datetime)

    """
    for column in columns:
        dataframe[column] = dataframe[column].fillna(dataframe[column].mean())
