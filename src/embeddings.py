from character_extraction import *
from book_entities import *
import re
import os
import os.path
import pandas as pd
import json
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
import plotly.express as px
import string
from transformers import FlaubertModel, FlaubertTokenizer


def get_name_window(total_word_index, gutenberg_id, window_size=3):
    '''Given the index of a single word in the book, and the ID of the book, returns the book window
    surrounding that word.

    Parameters
    ----------
    total_word_index : int
        The book-wise index of the word for which to extract the context
    gutenberg_id : int
        The book's Project Gutenberg ID
    window_size : int, optional
        The context window size, in number of words, both backwards and forward (i.e. a window_size 
        of 3 will return a context of 7 words (3 + 1 + 3)) (default is 3)

    Returns
    -------
    context : str
        The context surrounding the given word's index
    '''
    
    book_text = get_book_text(gutenberg_id)

    # prepare for iteration over the book
    book_words = [w if w != 'word_tokenize_splits_cannot_into_2_words' else 'cannot'
              for w in word_tokenize(re.sub(r'[^a-zA-Z0-9À-ÿ]', ' \g<0> ', 
                                            ' '.join(re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', book_text))
                                           ).replace('cannot', 'word_tokenize_splits_cannot_into_2_words'))]
    
    word_window = book_words[max(total_word_index - window_size, 0): total_word_index + window_size]
    return re.sub(r' ([^a-zA-Z0-9À-ÿ]) ', '[_]\g<0>[_] ', ' '.join(word_window)).replace('[_] ', '')

def get_all_name_windows_for_entities(total_word_indexes, gutenberg_id, window_size=2):
    '''Given several indexes for an entity in the book, and the ID of the book, returns the context windows surrounding 
    this entity. The goal is to extract in those contexts all the possible denomiation for this entity.

    Parameters
    ----------
    total_word_indexes : list
        The list of book-wise indexes of the words for which to extract the context
    gutenberg_id : int
        The book's Project Gutenberg ID
    window_size : int, optional
        The context window size, in number of words, before the entity (i.e. a window_size 
        of 3 will return a context of 4 words (3 + 1)) (default is 2)

    Returns
    -------
    contexts : list
        The list of contexts surrounding each given entity appearance's indexes
    '''
    
    book_text = get_book_text(gutenberg_id)

    # prepare for iteration over the book
    book_words = [w if w != 'word_tokenize_splits_cannot_into_2_words' else 'cannot'
              for w in word_tokenize(re.sub(r'[^a-zA-Z0-9À-ÿ]', ' \g<0> ', 
                                            ' '.join(re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', book_text))
                                           ).replace('cannot', 'word_tokenize_splits_cannot_into_2_words'))]
    
    result = []
    for total_word_index in total_word_indexes:
        # keep as context : window_size words before the entity and the entity
        word_window = book_words[max(total_word_index - window_size, 0): total_word_index+1]
        result.append(re.sub(r' ([^a-zA-Z0-9À-ÿ]) ', '[_]\g<0>[_] ', ' '.join(word_window)).replace('[_] ', ''))
    
    return result

def from_name_window_to_entities (name_window, nlp) :
    '''Given list of contexts and a nlp model, returns all the possible denominations of an entity.

    Parameters
    ----------
    name_window : list
        List of word contexts surrounding an entity
    nlp : pipeline
        huggingface's NER pipeline object

    Returns
    -------
    list
        The list of strings corresponding to the different names for a same entity.
    '''
    
    final = []
    for l in name_window:
        # initiate list that will contain the entities and honorifics of a context
        line_entities = []
        entities = nlp(l)
        # list of entities in given context l
        word_entities = [e['word'] for e in entities if e['entity_group'] == 'PER']
        for w in l.translate(str.maketrans('', '', string.punctuation)).split(' ') :
            if w in word_entities or w.lower() in french_honorific :
                # keep only words that are either an entity of a honorific
                line_entities.append(w)
        if len(line_entities)>1 or (len(line_entities)==1 and (line_entities[0].lower() not in french_honorific)) :
            # join into a string only the lists that either contain a single word that is not an honorific 
                # or lists that contain more than 1 word
            final.append(' '.join(line_entities))
    return list(set(final))

def get_all_name_windows(total_word_indexes, gutenberg_id, window_size=3):
    '''Given several word indexes in the book, and the ID of the book, returns the book windows surrounding 
    each of those words.

    Parameters
    ----------
    total_word_indexes : list
        The list of book-wise indexes of the words for which to extract the context
    gutenberg_id : int
        The book's Project Gutenberg ID
    window_size : int, optional
        The context window size, in number of words, both backwards and forward (i.e. a window_size 
        of 3 will return a context of 7 words (3 + 1 + 3)) (default is 3)

    Returns
    -------
    contexts : list
        The list of contexts surrounding each given word's indexes
    '''
    
    book_text = get_book_text(gutenberg_id)

    # prepare for iteration over the book
    book_words = [w if w != 'word_tokenize_splits_cannot_into_2_words' else 'cannot'
              for w in word_tokenize(re.sub(r'[^a-zA-Z0-9À-ÿ]', ' \g<0> ', 
                                            ' '.join(re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', book_text))
                                           ).replace('cannot', 'word_tokenize_splits_cannot_into_2_words'))]
    
    result = []
    for total_word_index in tqdm(total_word_indexes):
        word_window = book_words[max(total_word_index - window_size, 0): total_word_index + window_size]
        result.append(re.sub(r' ([^a-zA-Z0-9À-ÿ]) ', '[_]\g<0>[_] ', ' '.join(word_window)).replace('[_] ', ''))
    
    return result


def get_nearest_relations(gutenberg_id, window_size = 15, grouped_entities=False):
    '''Given a book ID, return a dictionary with the count, for each of the book's entities,
    of its nearest verbs and other entities, where proximity is constrained by the `window_size`
    parameter.

    Parameters
    ----------
    gutenberg_id : int
        The book's Project Gutenberg ID
    window_size : int, optional
        The context window size, in number of words, both backwards and forward (i.e. a window_size 
        of 15 will find nearest entities in a 31-word (15 + 1 + 15) window surrounding each entity, 
        and the nearest verbs in a 30-word (15*2) after it) (default is 15)
    grouped_entities : bool, optional
        Flag indicating whether the NER pipeline used to create the entities was configured to output 
        grouped_entities or not (default is False)

    Returns
    -------
    nearest_relations : dictionary
        A dictionary containing the nearest entities and verbs to each entity and their count
    '''
    
    # check if df already exists on the disk
    nearest_relations_path = f'../data/nearest_relations/{gutenberg_id}_{window_size}.json'
    if grouped_entities:
        nearest_relations_path = nearest_relations_path[:-5] + '_GE.json'
    if os.path.isfile(nearest_relations_path):
        with open(nearest_relations_path) as f:
            return json.load(f)
        
        
    book_text = get_book_text(gutenberg_id)
    book_df = get_book_df(gutenberg_id, grouped_entities).drop_duplicates('total_word_index')

    # prepare for iteration over the book
    book_words = [w if w != 'word_tokenize_splits_cannot_into_2_words' else 'cannot'
              for w in word_tokenize(re.sub(r'[^a-zA-Z0-9À-ÿ]', ' \g<0> ', 
                                            ' '.join(re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', book_text))
                                           ).replace('cannot', 'word_tokenize_splits_cannot_into_2_words'))]
    book_df['total_word_index'].to_list()
    
    nearest_relations = {}
    lemma_function = WordNetLemmatizer()
    french_stopwords = pd.read_csv('../data/stopwords-fr.txt', header = None)[0].values.tolist()
    for i, row in tqdm(book_df.iterrows(), total=book_df.shape[0]):
        # get main entity, its lenght, and its index in the book
        entity = row['full_word']
        main_entity_words = len(entity.split())
        total_word_index = row['total_word_index']
        
        # create all relevant access indexes
        word_idx = total_word_index
        after_word_idx = total_word_index + main_entity_words
        ent_start_idx = max(total_word_index - window_size, 0)
        ent_end_idx = total_word_index + main_entity_words + window_size
        verb_end_idx = total_word_index + main_entity_words + 2*window_size
        
        # get nearest entities
        near_entities_window = book_words[ent_start_idx:word_idx] + book_words[after_word_idx:ent_end_idx]
        near_entities_text = re.sub(r' ([^a-zA-Z0-9À-ÿ]) ', '[_]\g<0>[_] ', 
                                    ' '.join(near_entities_window)).replace('[_] ', '')
        near_entities = [w[0].lower() for w in nltk.pos_tag(word_tokenize(near_entities_text)) 
                         if (w[1].startswith('NNP') 
                             and w[0].lower() not in french_stopwords
                             and w[0].isalpha())]
        
        # get nearest verbs
        near_verbs_window = book_words[after_word_idx: verb_end_idx]
        near_verbs_text = re.sub(r' ([^a-zA-Z0-9À-ÿ]) ', '[_]\g<0>[_] ', 
                                 ' '.join(near_verbs_window)).replace('[_] ', '')
        near_verbs = [lemma_function.lemmatize(w[0], pos='v') for w in nltk.pos_tag(word_tokenize(near_verbs_text)) 
                      if w[1].startswith('V') and w[0].isalpha() and len(w[0]) > 1]

        # record (and count) each near entity and verb
        if entity not in nearest_relations:
            nearest_relations[entity] = {'near_entities': {}, 'near_verbs': {}}

        for e in near_entities:
            if e in nearest_relations[entity]['near_entities']:
                nearest_relations[entity]['near_entities'][e] = nearest_relations[entity]['near_entities'][e] + 1
            else:
                nearest_relations[entity]['near_entities'][e] = 1
        for v in near_verbs:
            if v in nearest_relations[entity]['near_verbs']:
                nearest_relations[entity]['near_verbs'][v] = nearest_relations[entity]['near_verbs'][v] + 1
            else:
                nearest_relations[entity]['near_verbs'][v] = 1


    # save embeddings to disk and then return them
    with open(nearest_relations_path, 'w+') as f:
        json.dump(nearest_relations, f)
    return nearest_relations      


def get_book_df(gutenberg_id, grouped_entities=False):
    '''Creates an Entities dataframe for the given book. If it already exists, just loads it from memory.

    Parameters
    ----------
    gutenberg_id : int
        The book's Project Gutenberg ID
    grouped_entities : bool, optional
        Flag indicating whether the NER pipeline used to create the entities was configured to output 
        grouped_entities or not (default is False)

    Returns
    -------
    book_df : DataFrame
        A DataFrame containing the book's entities, their sentence-wise and book-wise indexes and their
        PER-classification scores
    '''
    
    # check if df already exists on the disk
    book_csv_path = f'../data/book_dfs/rouge_noir_df.csv'
    if grouped_entities:
        book_csv_path = f'../data/book_dfs/rouge_noir_df_grouped.csv'
    if os.path.isfile(book_csv_path):
        return pd.read_csv(book_csv_path)
    
    # if df doesn't already exist, create it
    (book_text, book_ent_tokens, book_ent_words) = get_person_entities(gutenberg_id, 
                                                                       grouped_entities=grouped_entities)
    book_df = pd.DataFrame(book_ent_words)
    book_df['full_word'] =  book_df['full_word'].apply(lambda s: s.lower())
    
    # save df to disk and then return it
    book_df.to_csv(book_csv_path, index=False) 
    return book_df


def get_embeddings_model(gutenberg_id, whole_book_embedding=True, embedding_model='fasttext', 
                         context_window_size=6, min_count=5, grouped_entities=False):
    '''Given a book ID, creates an embedding model for it, according to the specifications.

    Parameters
    ----------
    gutenberg_id : int
        The book's Project Gutenberg ID
    whole_book_embedding : bool, optional
        Wether to use the whole book to generate the embeddings (when true) or just context 
        windows surrounding entity references (default is True)
    embedding_model : str, optional
        Which embedding model to use, between skipgram, cbow and fasttext (default is 'fasttext')
    context_window_size : int, optional
        The context window size, in number of words, both backwards and forward (i.e. a window_size 
        of 6 will return a context of 13 words (6 + 1 + 6)) (default is 6)
    min_count : int, optional
        The minimum amount of times any term needs to appear in order for an embedding for it to be
        generated (default is 5)
    grouped_entities : bool, optional
        Flag indicating whether the NER pipeline used to create the entities was configured to output 
        grouped_entities or not (default is False)

    Returns
    -------
    emb_model : Model
        The trained embeddings model
    '''
    
    book_df = get_book_df(gutenberg_id, grouped_entities).drop_duplicates('total_word_index')
    
    # use either the whole book for the embeddings or just windows near entity mentions
    context = ''
    if whole_book_embedding:
        # set context as the whole book, split into sentences
        context = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', get_book_text(gutenberg_id))
    else:
        context = get_all_name_windows(book_df['total_word_index'].to_list(), gutenberg_id, 
                                       window_size=context_window_size)
    
    # clean and prepare the text
    processed_context = [re.sub(r'\s+', ' ', re.sub('[^a-zA-ZÀ-ÿ]', ' ', c.lower())) for c in context]
    all_sentences = [t for c in processed_context for t in nltk.sent_tokenize(c)]
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

    french_stopwords = pd.read_csv('../data/stopwords-fr.txt', header = None)[0].values.tolist()
    # remove stopwords
    for i in range(len(all_words)):
        all_words[i] = [w for w in all_words[i] if w not in french_stopwords]
    
    # select and create the embeddings
    if any([embedding_model.lower() == sg for sg in ['skipgram', 'skip-gram', 'skip gram', 'sg']]):
        # create SG model and return it
        return Word2Vec(all_words, sg=1, size=100,  workers=4, min_count=min_count)
        
    elif any([embedding_model.lower() == cbow for cbow in ['continuousbagofwords', 'continuous-bag-of-words', 
                                                           'continuous bag of words', 'cbow']]):
        # create CBOW model and return it
        return Word2Vec(all_words, sg=0, size=100,  workers=4, min_count=min_count)
    
    # create FastText model and return it
    return FastText(all_words, min_count=min_count)

def french_word_embeddings(model_name, gutenberg_id) :
    # modelname = 'flaubert/flaubert_base_cased' 

    # Load pretrained model and tokenizer
    flaubert, log = FlaubertModel.from_pretrained(model_name, output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(model_name, do_lowercase=False)

    french_stopwords = pd.read_csv('../data/stopwords-fr.txt', header = None)[0].values.tolist()
    
    book = get_book_text('798-8')
    pattern1 = r"""[,.;@#?!&$\s:]+"""
    pattern2 = r"""[-']"""

    contexts = re.sub(r'|'.join((pattern1, pattern2)),
               " ",          # and replace it with a single space
               book, flags=re.VERBOSE)
    contexts = [i for i in contexts.split(' ') if i.lower() not in french_stopwords]

    embeddings_dict = {}
    
    for i in tqdm(range(0, len(contexts), 510)) :
        bins = contexts[i:i+510]
        if bins : 
            token_ids = torch.tensor([flaubert_tokenizer.encode(bins)])
            last_layer = flaubert(token_ids)[0]

            for j in range (1, len(bins)-1) :
                key = bins[j-1].lower()
                value = last_layer[:,j,:]
                if key in embeddings_dict.keys():
                    old = embeddings_dict[key]
                    embeddings_dict[key] = torch.mean(torch.stack([old, value]), dim = 0)
                else :
                    embeddings_dict[key] = value

    return embeddings_dict

def get_entities_embeddings(gutenberg_id, emb_model, grouped_entities=False):
    '''Given the book DF and the embeddings model, returns a dictionary for the embeddings
    corresponding to each entity (entity list obtained from the book's DF).

    Parameters
    ----------
    gutenberg_id : int
        The book's Project Gutenberg ID
    emb_model : Model
        The trained (gensim) embeddings model
    grouped_entities : bool, optional
        Flag indicating whether the NER pipeline used to create the entities was configured to output 
        grouped_entities or not (default is False)

    Returns
    -------
    ent_vectors : dictionary
        A dictionary containing each entity and their associated embedding vector
    '''
    
    book_df = get_book_df(gutenberg_id, grouped_entities).drop_duplicates('total_word_index')
    ent_vectors = {}
    for n in book_df['full_word'].unique():
        if n in emb_model.keys():
            ent_vectors[n] = emb_model[n]
            
    return ent_vectors

def plot_embeddings_2D(gutenberg_id, emb_vectors, title, max_vectors=None, min_count=5, grouped_entities=False):
    '''Given the book DF, the embedding vectors and the title, plots them in 2D, using PCA for
    the dimensionality reduction.

    Parameters
    ----------
    gutenberg_id : int
        The book's Project Gutenberg ID
    ent_vectors : dictionary
        A dictionary containing each entity and their associated embedding vector
    title : str
        The plot's title
    max_vectors : int, optional
        The maximum number of entities to display in the plot. If not None, the max_vector most common
        entities will be plotted (default is None)
    min_count : int, optional
        The minimum amount of times any entity needs to appear in the books in order for it to be
        plotted (default is 5)
    grouped_entities : bool, optional
        Flag indicating whether the NER pipeline used to create the entities was configured to output 
        grouped_entities or not (default is False)
    '''
    
    # get entities and apply min_count
    book_df = get_book_df(gutenberg_id, grouped_entities).drop_duplicates('total_word_index')
    tmp_df = book_df.groupby(['full_word']).count().reset_index()
    tmp_df = tmp_df[tmp_df['full_word'].str.isalpha()]
    tmp_df = tmp_df[tmp_df['score'] >= min_count][['full_word', 'score']]
    
    # get embeddings (apply max_vectors if not None)
    key_list = (tmp_df.sort_values(by='score', ascending=False).reset_index())
    
    if max_vectors:
        key_list = key_list[:max_vectors]
    key_list = key_list.sort_values(by='full_word')['full_word'].unique()
    vec_to_plot = { key: emb_vectors[key] for key in key_list if key in emb_vectors}
        
    # apply PCA
    sg_df = pd.DataFrame(vec_to_plot).T
    pca = PCA(n_components=2)
    components = pca.fit_transform(sg_df)
    
    # plot the vectors
    fig = px.scatter(components, x=0, y=1, color=sg_df.index, title=title)
    fig.show()