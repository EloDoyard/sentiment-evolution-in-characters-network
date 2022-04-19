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
    
    # get list of french stopwords
    french_stopwords = []
    with open('../data/stopwords-fr.txt', 'r') as f:
        french_stopwords = [*map(str.strip, f.readlines())]
    
    # get book df, all its entities and all their contexts
    book_df = get_book_df(gutenberg_id, grouped_entities = True)
    contexts = get_all_name_windows(book_df.drop_duplicates('total_word_index')['total_word_index'].to_list(), 
                         gutenberg_id, window_size=context_window_size)

    # load FlauBERT model and tokenizer
    tokenizer = FlaubertTokenizer.from_pretrained(flaubert_model)
    model = FlaubertModel.from_pretrained(flaubert_model)

    # get the embeddings
    embeddings_BERT = {}
    for i, context in enumerate(tqdm(contexts)):
        # prepare context for equality check
        processed_context = [w for w in context.split() if w.lower() not in french_stopwords]
        processed_context = ' '.join(processed_context)
        # prepare context for embedding
        matching_entity = None
        for ent in entities:
            for address in ent.all_addresses:
                if address.lower() in processed_context.lower():
                    matching_entity = (BookEntity.from_list_entity(ent).get_shortname(), address)
        if not matching_entity:
            continue
        
        entity_key_form, entity_address = matching_entity
        entity_idx = context.lower().index(entity_address.split()[0].lower())
        text = context[:entity_idx] + '<special1>' + context[entity_idx+len(entity_address):]

        # encode context and extract relevant indexes
        encoded_input = tokenizer(text, return_tensors='pt') 
        cls_idx = 0 # it's always the first token
        mask_idx = encoded_input.input_ids[0].tolist().index(0) # 0,1,5 # WHY THIS VALUE ?

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
    
    for key, value in vec_to_plot.items() :
        temp_nan =np.isnan(value)
        temp_infs = np.isinf(value)
        if any(temp_nan) :
            print('there is a nan in '+key)
        if any(temp_infs) :
            print('there is an infinite in '+key)
        
    # apply PCA
    sg_df = pd.DataFrame(vec_to_plot).T
    pca = PCA(n_components=2)
    components = pca.fit_transform(sg_df)
    
    # plot the vectors
    fig = px.scatter(components, x=0, y=1, color=sg_df.index, 
                     title=f'{emb_type} BERT embeddings {title_addition}',
                     color_discrete_sequence=px.colors.qualitative.Alphabet)
    fig.update_layout(
    autosize=False,
    width=700,
    height=700,)
    fig.show()
    
def get_book_entities(entities_file_path):
    '''Given a book ID, computes the list of textually-close BookEntities and more laxly
    match BookEntities, and returns them.

    Parameters
    ----------
    entities_file_path : string
        Path to find file containing entities to use
        
    Returns
    -------
    textually_close_merged_book_entities: list
        A list of the textually-close matched BookEntity instances
    lax_merged_book_entities: list
        A list of the more laxly matched BookEntity instances
    '''
    
    entities_df = pd.read_csv(entities_file_path, skiprows=[0],
                          names = ['total_word_index'])

    final_entity_list = entities_df['total_word_index'].tolist() 
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

def construct_groups_tokens(name_df_path, book_pg_id, max_chunk_len = 512, grouped_entities = False, entity_window = 2) :
    '''Given a dataframe path, computes a collection of grouped tokens defining supposedly a character, 
            by looking at the before context of each entities

    Parameters
    ----------
    name_df_path : string
        Data path of the CSV to extract the single tokens from
    book_pg_id : string
        Book ID of the corresponding corpus extracting grouped tokens from
    max_chunk_len = 512 : Int, optional
        Maximum character-level length of each sentence passed to the model (default is 512)
    grouped_entities = False : Boolean, optional
        Flag indicating whether the NER pipeline is configured to output grouped_entities or not 
        (default is False)
    entity_window = 2 : Int, optional
        The entity context window size, in number of words forward (i.e. a entity_window 
        of 3 will return a context of 4 words (3 + 1)) (default is 2)
        
    Returns
    -------
    textually_close_merged_book_entities: list
        A list of the textually-close matched BookEntity instances
    lax_merged_book_entities: list
        A list of the more laxly matched BookEntity instances
    '''
    # entity_df = pd.read_csv(f'../data/book_dfs/rouge_noir_df_grouped.csv', skiprows=[0],
      #                    names = ['full_word','sentence_word_index','total_word_index','score'])
    
    # read the file containing extracted single tokens
    entity_df = pd.read_csv(name_df_path, skiprows=[0],
                          names = ['full_word','sentence_word_index','total_word_index','score'])
    
    # read list of french stopwords
    french_stopwords = []
    with open('../data/stopwords-fr.txt', 'r') as f:
        french_stopwords = [*map(str.strip, f.readlines())]
    
    # keep only the most popular single tokens, i.e. the ones that were extracted strictly more than 4 times
    entity_df = entity_df.groupby(entity_df.full_word).agg(
        {'sentence_word_index':list, 'total_word_index':list, 'score':'count'}).reset_index()
    entity_df = entity_df[entity_df.score>4]
    
    # loead model and tokenizer
    ner_model = 'Jean-Baptiste/camembert-ner'
    tokenizer = CamembertTokenizer.from_pretrained(ner_model, max_length = max_chunk_len)
    model = CamembertForTokenClassification.from_pretrained(ner_model, max_length = max_chunk_len)
    
    # initialize nlp pipeline
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities = grouped_entities)
    
    # for each popular entity find all the contexts of each entities in the corpus
    entities_windows = entity_df.total_word_index.apply(
        lambda x: from_name_window_to_entities(
            get_all_name_windows_for_entities(x, book_pg_id, window_size=entity_window), nlp))
    
    # explode list of contexts entities
    entities_windows = entities_windows.explode().dropna()
    
    return entities_windows.reset_index()['total_word_index']