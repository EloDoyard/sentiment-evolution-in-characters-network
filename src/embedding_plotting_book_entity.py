from embeddings import *
from book_entities import *
from transformers import  FlaubertTokenizer, FlaubertModel
from itertools import groupby
from operator import itemgetter
from functools import reduce
import numpy as np


def get_all_entity_based_BERT_embeddings(gutenberg_id, entities, context_window_size=7, 
                                         flaubert_model='flaubert/flaubert_large_cased'):
    '''Given the book ID (and optionally several other settings), returns, for each BookEntity in
    the given list (using a context of the specified size), all the [CLS], [MASK] and mean context 
    embeddings.

    Parameters
    ----------
    gutenberg_id : int
        The book's Project Gutenberg ID
    entities: list
        A list of BookEntity instances
    context_window_size : int, optional
        The context window size, in number of words, both backwards and forward (i.e. a window_size 
        of 7 will return a context of 15 words (7 + 1 + 7)) (default is 7)
    bert_model : str, optional
        The underlying pre-trained BERT model to use (default is 'bert-large-cased')

    Returns
    -------
    embeddings_BERT : dictionary
        A dictionary containing, for each entity, the lists of all their [MASK], [CLS], and 
        'context average' embeddings
    '''
    
    # get book df, all its entities and all their contexts
    book_df = get_book_df(gutenberg_id)
    contexts = get_all_name_windows(book_df.drop_duplicates('total_word_index')['total_word_index'].to_list(), 
                         gutenberg_id, window_size=context_window_size)

    # load FlauBERT model and tokenizer
    tokenizer = FlaubertTokenizer.from_pretrained(flaubert_model)
    model = FlaubertModel.from_pretrained(flaubert_model)

    # get the embeddings
    embeddings_BERT = {}
    for i, context in enumerate(tqdm(contexts)):

        # prepare context for embedding
        matching_entity = None
        for ent in entities:
            for address in ent.all_addresses:
                if address in context:
                    matching_entity = (BookEntity.from_list_entity(ent).get_shortname(), address)
        if not matching_entity:
            continue
        
        entity_key_form, entity_address = matching_entity
        entity_idx = context.lower().index(entity_address.lower())
        text = context[:entity_idx] + '<special1>' + context[entity_idx+len(entity_address):]

        # encode context and extract relevant indexes
        encoded_input = tokenizer(text, return_tensors='pt') 
        cls_idx = 0 # it's always the first token
        mask_idx = encoded_input.input_ids[0].tolist().index(5)

        # get the relevant embeddings and add them to the dictionary
        output = model(**encoded_input)
        
        if entity_key_form in embeddings_BERT:
            embeddings_BERT[entity_key_form]['CLS'].append(output[0][0][cls_idx, :].tolist())
            embeddings_BERT[entity_key_form]['MASK'].append(output[0][0][mask_idx, :].tolist())
            
            # embeddings for all other tokens, expect final [SEP] token (hence the -1)
            embeddings_BERT[entity_key_form]['context'].append(torch.mean(torch.cat((output[0][0][1:mask_idx, :],
                                                                      output[0][0][mask_idx+1:-1, :])), 0).tolist())
        else:
            embeddings_BERT[entity_key_form] = {'CLS': [output[0][0][cls_idx, :].tolist()],
                                                'MASK': [output[0][0][mask_idx, :].tolist()],
                                                'context': [torch.mean(torch.cat((output[0][0][1:mask_idx, :],
                                                                   output[0][0][mask_idx+1:-1, :])), 0).tolist()]}
              
#     # save embeddings to disk and then return them
#     with open(embeddings_path, 'w+') as f:
#         json.dump(embeddings_BERT, f)
    return embeddings_BERT

def get_averaged_entity_based_BERT_embeddings(gutenberg_id, entities, 
                                              context_window_size=7, flaubert_model='flaubert/flaubert_large_cased'):
    '''Given the book ID (and optionally several other settings), returns, for each BookEntity in
    the given list, three embeddings: the mean [CLS] vector, the mean [MASK] vector and the mean 
    of the mean context embeddings.

    Parameters
    ----------
    gutenberg_id : int
        The book's Project Gutenberg ID
    entities: list
        A list of BookEntity instances
    context_window_size : int, optional
        The context window size, in number of words, both backwards and forward (i.e. a window_size 
        of 7 will return a context of 15 words (7 + 1 + 7)) (default is 7)
    bert_model : str, optional
        The underlying pre-trained BERT model to use (default is 'bert-large-cased')

    Returns
    -------
    embeddings_BERT : dictionary
        A dictionary containing, for each entity, three average embeddings, respetively the mean of all 
        their [MASK] embeddings, of all their [CLS] embeddings, and of all their 'context average' embeddings
    '''

    # get the embeddings
    embeddings_BERT = get_all_entity_based_BERT_embeddings(gutenberg_id, entities, context_window_size, flaubert_model)
    
    # average all the embeddings
    avg_BERT_embs = {}
    emb_entities = embeddings_BERT.keys()
    
    for e in emb_entities:
        avg_BERT_embs[e] = {'CLS': np.array(embeddings_BERT[e]['CLS']).mean(axis=0),
                            'MASK': np.array(embeddings_BERT[e]['MASK']).mean(axis=0),
                            'context': np.array(embeddings_BERT[e]['context']).mean(axis=0)}
    
    return avg_BERT_embs

def plot_entity_embeddings_2D(avg_BERT_embeddings, emb_type='MASK', title_addition=''):
    '''Given a dictionary of average BERT embeddings (and optionally several other settings, 
    including which of the three embedding type to plot ['CLS', 'MASK' or 'context']), uses PCA 
    to reduce the embeddings to 2D and then plots them.

    Parameters
    ----------
    avg_BERT_embeddings : dictionary
        A dictionary containing each entity and their associated embedding vectors, for the three
        possible types (MASK, CLS and context) 
    emb_type : str, optional
        The type of embedding to plot, from the three possible choices (MASK, CLS and context) 
        (default is 'MASK')
    title_addition : str, optional
        The text to add to the title (containing for example the book name) (default is '')
    '''
    
    # filter embeddings according to their chosen embedding type and min/max counts
    if emb_type != 'CLS' and emb_type != 'context':
        emb_type = 'MASK'
    vec_to_plot = {key: avg_BERT_embeddings[key][emb_type] for key in avg_BERT_embeddings}
        
    # apply PCA
    sg_df = pd.DataFrame(vec_to_plot).T
    pca = PCA(n_components=2)
    components = pca.fit_transform(sg_df)
    
    # plot the vectors
    fig = px.scatter(components, x=0, y=1, color=sg_df.index, 
                     title=f'{emb_type} BERT embeddings {title_addition}',
                     color_discrete_sequence=px.colors.qualitative.Alphabet)
    fig.show()
    
def get_book_entities(book_pg_id):
    '''Given a book ID, computes the list of textually-close BookEntities and more laxly
    match BookEntities, and returns them.

    Parameters
    ----------
    book_pg_id : int
        The book's Project Gutenberg ID
        
    Returns
    -------
    textually_close_merged_book_entities: list
        A list of the textually-close matched BookEntity instances
    lax_merged_book_entities: list
        A list of the more laxly matched BookEntity instances
    '''
    
    luke_df = pd.read_csv(f'../data/book_dfs/luke_{book_pg_id}_df.csv', skiprows=[0],
                          names=['full_word', 'sentence_word_index', 'total_word_index'])
    # luke_df = pd.read_csv(f'../data/book_dfs/798-8.csv', skiprows=[0],
      #                    names = ['full_word','sentence_word_index','total_word_index','score'])

    french_stopwords = []
    with open('../data/stopwords-fr.txt', 'r') as f:
        french_stopwords = [*map(str.strip, f.readlines())]
    
    entity_list = [' '.join([word.strip(',,,.""”’‘\'!?;:-')
                             for word in row['full_word'].split() if word.lower() not in french_stopwords]) 
                   for i, row in luke_df.iterrows()]
    entity_list = [(ent, ent.lower(), 1) for ent in entity_list if ent != '']
    entity_list = [reduce(count_reduce, group) for _, group in groupby(sorted(entity_list), key=itemgetter(1))]

    final_entity_list = [ent[0] for ent in entity_list if ent[2] > 4]

    all_book_entities = []
    for name in final_entity_list:
        # merge with already existing entity if possible
        merged = False
        for ent in all_book_entities:
            if ent.exactly_references_entity(name):
                ent.merge(name, strictness='exact')
                merged = True
                break

        # create new entity if not compatible with any of the previously existing ones
        if not merged:
            new_ent = BookListEntity(name)
            all_book_entities.append(new_ent)

    merged_book_entities = []
    for ent in all_book_entities:
        merged = False
        for final_ent in merged_book_entities:
            if any([final_ent.exactly_references_entity(address) for address in ent.all_addresses]):
                [final_ent.merge(address, strictness='none') for address in ent.all_addresses]
                merged = True
                break

        if not merged:
            merged_book_entities.append(ent)

    textually_close_merged_book_entities = []
    for ent in merged_book_entities:
        merged = False
        for final_ent in textually_close_merged_book_entities:
            if any([final_ent.textually_close_references_entity(address) for address in ent.all_addresses]):
                [final_ent.merge(address, strictness='none') for address in ent.all_addresses]
                merged = True
                break

        if not merged:
            textually_close_merged_book_entities.append(ent)

    fuzzy_merged_book_entities = []
    for ent in merged_book_entities:
        merged = False
        for final_ent in fuzzy_merged_book_entities:
            if any([final_ent.references_entity(address) for address in ent.all_addresses]):
                [final_ent.merge(address) for address in ent.all_addresses]
                merged = True
                break

        if not merged:
            fuzzy_merged_book_entities.append(ent)

    lax_merged_book_entities = []
    for ent in fuzzy_merged_book_entities:
        merged = False
        for final_ent in lax_merged_book_entities:
            if any([final_ent.fuzzy_references_entity(address) for address in ent.all_addresses]):
                [final_ent.merge(address, strictness='fuzzy') for address in ent.all_addresses]
                merged = True
                break

        if not merged:
            lax_merged_book_entities.append(ent)
            
    return textually_close_merged_book_entities, lax_merged_book_entities