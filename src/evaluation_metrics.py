import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


ground_truth_data_df = pd.DataFrame([ 
('agde', 'Monsieur Agde',13,'M',36930), 
('altamira','Comte Altamira',26,'M',97835),
('amanda', 'Mme Amanda Binet',17,'F',60381),
('appert', 'M Appert',8,'M',2363),
('castanède','Abbé Castanède',19,'M',64861),
('caylus','Comte Caylus',24,'M',94378),
('chélan','Abbé Chélan',6,'M',1929),
('croisenois','Monsieur Croisenois',23,'M',94374),
('danton','Monsieur Danton',1,'M', 15),
('derville','Madame Derville',12,'F',13130),
('falcoz','Monsieur Falcoz',14,'M',45151),
('fervaques','Madame Fervaques',25,'F',96924),
('fouqué','Monsieur Fouqué',10,'M',7451),
('frilair','Monsieur Frilair',15,'M',53833),
('geronimo','Monsieur Geronimo',16,'M',55797),
('korasoff','Monsieur Korasoff',27,'M',102772),
('julien','Monsieur Julien Sorel',3,'M',4751),
('louise','Madame Louise Rênal',7,'F',45391),
('maslon','Monsieur Maslon',5,'M',1900),
('mathilde','Mademoiselle Mathilde Sorel',21,'F',90709),
('norbert','Monsieur Norbert Mole',20,'M',87123),
('pirard','Monsieur Pirard',18,'M',62166),
('rênal','Monsieur de Rênal',2,'M', 605),
('rênal','Madame Louise Rênal',7,'F', 2214),
('sorel','Monsieur Julien Sorel',3,'M', 940),
('tanbeau','Monsieur Tanbeau',22,'M',92323),
('valenod','Monsieur Valenod',4,'M',1724),
('élisa','Mademoiselle Élisa',11,'F',12267),
('mole', 'Mademoiselle Mathilde Sorel', 21,'F',90768),
('mole', 'Monsieur de la Mole',9,'M',2610)],
columns=['name', 'entity','entity_ID', 'gender','first_appearance' ])

def get_clustering_metrics(embeddings, embeddings_type):
    '''Given embeddings, and their ground truth data type, computes several clustering performance
    metrics. The right `ground_truth_data_df`, `textually_close_ent_ground_truth_df` or 
    `lax_ent_ground_truth_df` should have been loaded into memory before calling this function.

    Parameters
    ----------
    embeddings : dictionary
        The dictionary containing each entity and their associated embedding vector
    embeddings_type : str
        The matching ground truth data type for the given embeddings (either 'first_version',
        'textually_close' or 'lax')

    Returns
    -------
    same_entityness : list
        A list containing the performance metrics with regards to the 'same_entityness' axis
    gender : list
        A list containing the performance metrics with regards to the 'gender' axis
    first_appearance : list
        A list containing the performance metrics with regards to the 'first_appearance' axis
    '''
    
    # SAME ENTITY-NESS
    same_entityness = []
    
    mask_embs_entity = [(k, 
                         embeddings[k], 
                         ground_truth_data_df[ground_truth_data_df['name'] == k]['entity_ID'].values[0]) 
                        for k in embeddings 
                        if k.lower() in ground_truth_data_df['name'].tolist()]
        
    tmp_df = pd.DataFrame(mask_embs_entity)
    same_entityness.append(sklearn.metrics.silhouette_score(np.array(tmp_df[1].tolist()), 
                                                            np.array(tmp_df[2]), 
                                                            metric='euclidean', 
                                                            random_state=0))
    
    same_entityness.append(sklearn.metrics.calinski_harabasz_score(np.array(tmp_df[1].tolist()), 
                                                                   np.array(tmp_df[2])))
    
    same_entityness.append(sklearn.metrics.davies_bouldin_score(np.array(tmp_df[1].tolist()), 
                                                                np.array(tmp_df[2])))
    
    tmp_df = pd.DataFrame(mask_embs_entity)
    entityness_matrix = np.array([np.array(emb) for emb in tmp_df[1]])
    k_choice = 21 # obtained by the elbow method
    kmean = KMeans(n_clusters=k_choice, random_state=0).fit(entityness_matrix, )
    predicted_clusters = kmean.predict(np.array([np.array(emb) for emb in tmp_df[1]]))
    
    
    same_entityness.append(sklearn.metrics.rand_score(np.array(tmp_df[2]), predicted_clusters))
    same_entityness.append(sklearn.metrics.adjusted_rand_score(np.array(tmp_df[2]), predicted_clusters))
    same_entityness.append(sklearn.metrics.mutual_info_score(np.array(tmp_df[2]), predicted_clusters))
    same_entityness.append(sklearn.metrics.adjusted_mutual_info_score(np.array(tmp_df[2]), 
                                                                      predicted_clusters, 
                                                                      average_method='arithmetic'))
    
    
    # GENDER
    gender = []
    
    mask_embs_gender = [(k, 
                         embeddings[k], 
                         ground_truth_data_df[ground_truth_data_df['name'] == k]['gender'].values[0]) 
                        for k in embeddings 
                        if k.lower() in ground_truth_data_df['name'].tolist()]

    tmp_df = pd.DataFrame(mask_embs_gender)
    gender.append(sklearn.metrics.silhouette_score(np.array(tmp_df[1].tolist()), 
                                                   np.array(tmp_df[2] == 'M').astype(int), 
                                                   metric='euclidean', 
                                                   random_state=0))
    gender.append(sklearn.metrics.calinski_harabasz_score(np.array(tmp_df[1].tolist()), np.array(tmp_df[2])))
    gender.append(sklearn.metrics.davies_bouldin_score(np.array(tmp_df[1].tolist()), np.array(tmp_df[2])))
    
    tmp_df = pd.DataFrame(mask_embs_gender)
    gender_matrix = np.array([np.array(emb) for emb in tmp_df[1]])
    k_choice = 2 # two genders in PG literature (men and women)
    kmean = KMeans(n_clusters=k_choice, random_state=0).fit(gender_matrix)
    predicted_clusters = kmean.predict(np.array([np.array(emb) for emb in tmp_df[1]]))
    
    gender.append(sklearn.metrics.rand_score(np.array(tmp_df[2]), predicted_clusters))
    gender.append(sklearn.metrics.adjusted_rand_score(np.array(tmp_df[2]), predicted_clusters))
    gender.append(sklearn.metrics.mutual_info_score(np.array(tmp_df[2]), predicted_clusters))
    gender.append(sklearn.metrics.adjusted_mutual_info_score(np.array(tmp_df[2]), predicted_clusters, 
                                                             average_method='arithmetic'))
    
    # FIRST APPEARANCE
    first_appearance = []
    
    # build distance matrix 
    mask_embs_appear = [(k, 
                         embeddings[k], 
                         ground_truth_data_df[ground_truth_data_df['name'] == k]['first_appearance'].values[0]) 
                        for k in embeddings 
                        if k.lower() in ground_truth_data_df['name'].tolist()]
        
    tmp_df = pd.DataFrame(mask_embs_appear)
    appear_matrix = np.array(tmp_df[2]).reshape(-1, 1)

    # k based both on "vector" being predict (first appearance in book) and overall clustering
    # using elbow shape
    k_choice = 20
    kmean = KMeans(n_clusters=k_choice, random_state=0).fit(appear_matrix)

    first_appearance.append(sklearn.metrics.silhouette_score(np.array(tmp_df[1].tolist()), 
                                         kmean.predict(np.array(tmp_df[2]).reshape(-1,1)), 
                                         metric='euclidean', 
                                         random_state=0))
    
    first_appearance.append(sklearn.metrics.calinski_harabasz_score(np.array(tmp_df[1].tolist()), 
                                 kmean.predict(np.array(tmp_df[2]).reshape(-1,1))))
    
    first_appearance.append(sklearn.metrics.davies_bouldin_score(np.array(tmp_df[1].tolist()), 
                                 kmean.predict(np.array(tmp_df[2]).reshape(-1,1))))
    
    tmp_df = pd.DataFrame(mask_embs_appear)
    ground_truth_based_clusters = kmean.predict(np.array(tmp_df[2]).reshape(-1,1))
    appear_matrix = np.array([np.array(emb) for emb in tmp_df[1]])
    kmean = KMeans(n_clusters=k_choice, random_state=0).fit(appear_matrix)
    predicted_clusters = kmean.predict(np.array([np.array(emb) for emb in tmp_df[1]]))
    
    first_appearance.append(sklearn.metrics.rand_score(ground_truth_based_clusters, predicted_clusters))
    first_appearance.append(sklearn.metrics.adjusted_rand_score(ground_truth_based_clusters, predicted_clusters))
    first_appearance.append(sklearn.metrics.mutual_info_score(ground_truth_based_clusters, predicted_clusters))
    first_appearance.append(sklearn.metrics.adjusted_mutual_info_score(ground_truth_based_clusters, predicted_clusters, 
                                                                       average_method='arithmetic'))
    
    return same_entityness, gender, first_appearance

def print_clustering_metrics(embeddings, embeddings_type):
    '''Given embeddings, and their ground truth data type, display in a table several
    clustering performance metrics. The right `ground_truth_data_df`, 
    `textually_close_ent_ground_truth_df` or `lax_ent_ground_truth_df` should have been 
    loaded into memory before calling this function.

    Parameters
    ----------
    embeddings : dictionary
        The dictionary containing each entity and their associated embedding vector
    embeddings_type : str
        The matching ground truth data type for the given embeddings (either 'first_version',
        'textually_close' or 'lax')
    '''
    
    same_entityness, gender, first_appearance = get_clustering_metrics(embeddings, embeddings_type)
    print('-------------------------------------------------------------------------------')
    print('|                            | Same Entity-ness |  Gender  | First Appearance |')
    print('-------------------------------------------------------------------------------')
    print(f'| Silhouette Score           |     {same_entityness[0]:8.5f}     | {gender[0]:8.5f} |    {first_appearance[0]:8.5f}      |')
    print(f'| Calinski Harabasz Score    |     {same_entityness[1]:8.5f}     | {gender[1]:8.5f} |    {first_appearance[1]:8.5f}      |')
    print(f'| Davies Bouldin Score       |     {same_entityness[2]:8.5f}     | {gender[2]:8.5f} |    {first_appearance[2]:8.5f}      |')
    print(f'| Rand Score                 |     {same_entityness[3]:8.5f}     | {gender[3]:8.5f} |    {first_appearance[3]:8.5f}      |')
    print(f'| Adjusted Rand Score        |     {same_entityness[4]:8.5f}     | {gender[4]:8.5f} |    {first_appearance[4]:8.5f}      |')
    print(f'| Mutual Info Score          |     {same_entityness[5]:8.5f}     | {gender[5]:8.5f} |    {first_appearance[5]:8.5f}      |')
    print(f'| Adjusted Mutual Info Score |     {same_entityness[6]:8.5f}     | {gender[6]:8.5f} |    {first_appearance[6]:8.5f}      |')
    print('-------------------------------------------------------------------------------')