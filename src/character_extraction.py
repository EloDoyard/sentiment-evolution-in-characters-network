import pandas as pd
import re
from tqdm import tqdm, trange
from nltk.tokenize import word_tokenize
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification,LukeTokenizer, LukeForEntitySpanClassification, CamembertTokenizer, CamembertForTokenClassification
import spacy



def get_book_text(gutenberg_id):
    '''Given a book ID, returns the book's text, excluding Project Gutenberg's header and outro.
    
     Parameters
    ----------
    gutenberg_id : int
        The book's Project Gutenberg ID

    Returns
    -------
    book_text : str
        The book's text, excluding Project Gutenberg's header and outro
    '''
    
    context = ''
    with open(f'../data/book/PG-{gutenberg_id}.txt', mode='r', encoding='utf-8') as f:
        context = f.read()
    return ' '.join([l for l in (context.split('End of the Project Gutenberg EBook of ')[0]
                                        .split('*** END OF THE PROJECT GUTENBERG EBOOK')[0]
                                        .split('\n')) if l][16:])

def get_line_entities(l, ner_entities_tokens, ner_entities_words, sentence_index, tokenizer, nlp,
                     grouped_entities):
    '''Given a line, lists for tokens and words, and word index at the end of the sentence, as well as
    the tokenizer and nlp model instances (from huggingface's transformers), updates the tokens and
    words lists and the word index to include the given line.

    Parameters
    ----------
    l : str
        The line to analyze
    ner_entities_tokens : list
        A list containing all the Person tokens found so far, across all the previous lines
    ner_entities_words : list
        A list containing dictionary entries of all the Person entities found so far (the full 
        word corresponding to them (i.e. not separated tokens), their index in the sentence and 
        in the book overall, and their PER-entity classification score, a number between 0.0 and 
        1.0), across all the previous lines
    sentence_index : int
        The overall (book-wise) index of the first word of the sentence
    tokenizer : AutoTokenizer
        huggingface's tokenizer being used in the NER pipeline
    nlp : pipeline
        huggingface's NER pipeline object
    grouped_entities : bool
        Flag indicating whether the NER pipeline is configured to output grouped_entities or not

    Returns
    -------
    ner_entities_tokens : list
        A list containing all the Person tokens found so far, across this and all the previous lines
    ner_entities_words : list
        A list containing dictionary entries of all the Person entities found so far (the full 
        word corresponding to them (i.e. not separated tokens), their index in the sentence and 
        in the book overall, and their PER-entity classification score, a number between 0.0 and 
        1.0), across this and all the previous lines
    sentence_index : int
        The overall (book-wise) index of the first word of the next sentence
    '''
    french_stopwords = pd.read_csv('../data/stopwords-fr.txt', header = None)[0].values.tolist()
    
    new_entity_tokens = []
    if grouped_entities:
        new_entity_tokens = [e for e in nlp(l) if 'PER' in e['entity_group']]
    else:
        new_entity_tokens = [e for e in nlp(l) if 'PER' in e['entity']]
    ner_entities_tokens += new_entity_tokens
    
    tokenized_line = tokenizer(l)
    line_words = [w if w != 'word_tokenize_splits_cannot_into_2_words' else 'cannot'
                    for w in word_tokenize(
                                           re.sub(r'[^a-zA-Z0-9À-ÿ]', ' \g<0> ', 
                                                  l).replace('cannot', 
                                                            'word_tokenize_splits_cannot_into_2_words'))]
    
    # go from token to word with
    for et in new_entity_tokens:
        if grouped_entities:
            # find index of grouped entity
            reconstructed_line = ' '.join([lw.lower() for lw in line_words])
            first_word = word_tokenize(re.sub(r'[^a-zA-Z0-9À-ÿ]', ' \g<0> ', et['word']))[0]
            if et['word'][0] == '#':
                first_word = word_tokenize(re.sub(r'[^a-zA-Z0-9À-ÿ]', ' \g<0> ', et['word'][2:]))[0]
            
            word_index = len(reconstructed_line[:reconstructed_line.index(first_word)].split())

            if et['word'] not in french_stopwords and et['word'].isalpha():
                # record grouped entity
                ner_entities_words += [{'full_word': et['word'], 
                                        'sentence_word_index': word_index, 
                                        'total_word_index': sentence_index+word_index,
                                        'score': et['score']}]
        else:
            # record non-grouped entity
            word_index = tokenized_line.word_ids()[et['index']]
            if line_words[word_index] not in french_stopwords and line_words[word_index].isalpha():
                ner_entities_words += [{'full_word': line_words[word_index], 
                                        'sentence_word_index': word_index, 
                                        'total_word_index': sentence_index+word_index,
                                        'score': et['score']}]
    sentence_index += len(line_words)
    return ner_entities_tokens, ner_entities_words, sentence_index

def get_person_entities(gutenberg_id, grouped_entities=False, max_chunk_len=512, split_chunk_len=256):
    '''Given a book ID, returns its text (excluding Project Gutenberg's intro and outro), all its 
    tokens classified as PER (Person) entities, and all the words corresponding to those tokens, as 
    well as their index in the sentence and in the book, and their classification score as a PER entity.
    
    Parameters
    ----------
    gutenberg_id : int
        The book's Project Gutenberg ID
    grouped_entities : bool, optional
        Flag indicating whether the NER pipeline is configured to outout grouped_entities or not 
        (default is False)
    max_chunk_len : int, optional
        Maximum character-level length of each sentence passed to the model (default is 512)
    split_chunk_len : int, optional
        Maximum character-level length of each sub-sentence passed to the model, when splitting an
        overly big sentence into smaller sub-sentences (default is 256)

    Returns
    -------
    book_text : str
        The book's text, excluding Project Gutenberg's header and outro
    ner_entities_tokens : list
        A list containing all the Person tokens found across the whole book
    ner_entities_words : list
        A list containing dictionary entries of all the Person entities found across the whole book 
        (the full word corresponding to them (i.e. not separated tokens), their index in the sentence 
        and in the book overall, and their PER-entity classification score, a number between 0.0 and 
        1.0)
    '''
    # code is correct, but gives a warning about the model not having a predefined maximum length,
    # suppressing those warnings to not interfere with tdqm progress bar
    import warnings
    from transformers import logging
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    logging.set_verbosity_error()
    
    # read in gutenberg book
    book_text =  get_book_text(gutenberg_id)

    # load NER model and tokenizer
    # ner_model = 'mrm8488/mobilebert-finetuned-ner'
    # tokenizer = AutoTokenizer.from_pretrained(ner_model, max_length = max_chunk_len)
    # model = AutoModelForTokenClassification.from_pretrained(ner_model, max_length = max_chunk_len)
    
    ner_model = 'Jean-Baptiste/camembert-ner'
    tokenizer = CamembertTokenizer.from_pretrained(ner_model)
    model = CamembertForTokenClassification.from_pretrained(ner_model)
    
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities = grouped_entities)
    
    # prepare for iteration over the book
    sentence_level_book = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', book_text)
    ner_entities_tokens = []
    ner_entities_words = []
    sentence_index = 0
    
    # iterate over sentence-level chunks
    for l in tqdm(sentence_level_book):
        l = l.lower()
        if len(l) > max_chunk_len:
            for m in range(len(l) // split_chunk_len + 1):
                new_l = ' '.join(l.split(' ')[m*split_chunk_len:][:(m+1)*split_chunk_len])
                ner_entities_tokens, ner_entities_words, sentence_index = get_line_entities(new_l, 
                                                                                            ner_entities_tokens, 
                                                                                            ner_entities_words,
                                                                                            sentence_index, 
                                                                                            tokenizer,
                                                                                            nlp,
                                                                                            grouped_entities)
        else:
            ner_entities_tokens, ner_entities_words, sentence_index = get_line_entities(l, 
                                                                                        ner_entities_tokens, 
                                                                                        ner_entities_words,
                                                                                        sentence_index, 
                                                                                        tokenizer, 
                                                                                        nlp,
                                                                                        grouped_entities)

    return book_text, ner_entities_tokens, ner_entities_words


def get_LUKE_tokens_labels(l, tokenizer, model, nlp):
    '''Given a line, and the relevant NER and Laguage models and tokenizer, returns a list of tuples
    containing all the identified entities in that line.
    
    Parameters
    ----------
    l : str
        The line to analyze
    tokenizer : LukeTokenizer
        The LUKE NER model's tokenizer
    model : LukeForEntitySpanClassification
        The LUKE NER model instance
    nlp : Language
        spaCy's language model (code written for the "en_core_web_sm" version)

    Returns
    -------
    joint_per_entities : list
        A list of tuples, each tuple containing the joint entity (a single string with its several 
        tokens), and the (ordered and consecutive) indices of all its tokens
    line_len : int
        The length, in tokens, of l
    '''
    doc = nlp(l)

    entity_spans = []
    original_word_spans = []
    for token_start in doc:
        for token_end in doc[token_start.i:]:
            entity_spans.append((token_start.idx, token_end.idx + len(token_end)))
            original_word_spans.append((token_start.i, token_end.i + 1))

    inputs = tokenizer(l, entity_spans=entity_spans, return_tensors="pt", padding=True)
#     inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    max_logits, max_indices = logits[0].max(dim=1)
    predictions = []
    for logit, index, span in zip(max_logits, max_indices, original_word_spans):
        if index != 0:  # the span is not NIL
            predictions.append((logit, span, model.config.id2label[int(index)]))
            
    joint_per_entities = []            
    for _, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
        if label == 'PER':
            joint_per_entities.append((' '.join([str(doc[i]) for i in range(span[0], span[1])]),
                                       [i for i in range(span[0], span[1])]))

    return joint_per_entities, len(doc)

def get_LUKE_line_entities(l, ner_entities_words, sentence_index, tokenizer, model, nlp):
    '''Given a line, and the relevant NER and Laguage models and tokenizer, updates the entities'
    words list to reflect new entities found in that line, and the overall sentence index,

    Parameters
    ----------
    l : str
        The line to analyze
    ner_entities_words : list
        A list containing dictionary entries of all the Person entities found so far (the full 
        word corresponding to them (i.e. not separated tokens), and their index in the sentence 
        and in the book overall), across all the previous lines
    sentence_index : int
        The overall (book-wise) index of the first word of the sentence
    tokenizer : LukeTokenizer
        The LUKE NER model's tokenizer
    model : LukeForEntitySpanClassification
        The LUKE NER model instance
    nlp : Language
        spaCy's language model (code written for the "en_core_web_sm" version)

    Returns
    -------
    ner_entities_words : list
        A list containing dictionary entries of all the Person entities found so far (the full 
        word corresponding to them (i.e. not separated tokens), and their index in the sentence 
        and in the book overall), across this and all the previous lines
    sentence_index : int
        The overall (book-wise) index of the first word of the next sentence
    '''
    
    # guarantee line is not empty
    if not l.strip():
        return ner_entities_words, sentence_index
    
    # process line tokens and labels
    entities, sentence_len = get_LUKE_tokens_labels(l, tokenizer, model, nlp)
    for ent in entities:
        ner_entities_words += [{'full_word': ent[0], 
                                'sentence_word_index': ent[1][0], 
                                'total_word_index': sentence_index + ent[1][0]}]
    sentence_index += sentence_len
    return ner_entities_words, sentence_index

def get_LUKE_person_entities(gutenberg_id, checkpoint_directory_path, max_chunk_len=256, 
                             split_chunk_len=42, last_checkpoint=-1, ner_entities_words=[]):
    '''Given a book ID, returns its text (excluding Project Gutenberg's intro and outro), 
    all its tokens classified as PER (Person) entities, and all the words corresponding to 
    those token, as well as their index in the sentence and in the book.

    Parameters
    ----------
    gutenberg_id : int
        The book's Project Gutenberg ID
    checkpoint_directory_path : str
        The directory in which to save checkpoints at every 500th iteration
    max_chunk_len : int, optional
        Maximum character-level length of each sentence passed to the model (default is 512)
    split_chunk_len : int, optional
        Maximum word-level length (i.e. maximum number of words) of each sub-sentence passed 
        to the model, when splitting an overly big sentence into smaller sub-sentences (default 
        is 42)
    last_checkpoint : int, optional
        Checkpoint to resume from (default is -1)
    ner_entities_words : list, optional
        List containing dictionary entries of all the Person entities found up to 'last_checkpoint' 
        (default is [])

    Returns
    -------
    book_text : str
        The book's text, excluding Project Gutenberg's header and outro
    ner_entities_words : list
        A list containing dictionary entries of all the Person entities found across the whole book 
        (the full word corresponding to them (i.e. not separated tokens), and their index in the sentence 
        and in the book overall)
    '''
    
    # read in gutenberg book
    book_text = get_book_text(gutenberg_id)

    # Load the model checkpoint and tokenizer
    model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003", )
    model.eval()
#     model.to("cuda")
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

    # prepare for iteration over the book
    sentence_level_book = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', book_text)
    sentence_index = 0
    
    # iterate over sentence-level chunks
    nlp = spacy.load("en_core_web_sm")
    for i, l in enumerate(tqdm(sentence_level_book)):
        
        if i <= last_checkpoint:
            sentence_index += len(nlp(l))
            continue
            
        if len(l) > max_chunk_len:
            for m in range(len(l) // split_chunk_len + 1):
                new_l = ' '.join(l.split(' ')[m*split_chunk_len:][:split_chunk_len])
                ner_entities_words, sentence_index = get_LUKE_line_entities(new_l, ner_entities_words,
                                                                            sentence_index, tokenizer, 
                                                                            model, nlp)
        else:
            ner_entities_words, sentence_index = get_LUKE_line_entities(l, ner_entities_words,
                                                                        sentence_index, tokenizer, 
                                                                        model, nlp)
    
        if i % 500 == 0:
            # save a snapshot of the current progress
            tmp_df = pd.DataFrame(ner_entities_words)
            tmp_df.to_csv(f'{checkpoint_directory_path}luke_tmp_df_{i:05d}.csv', index=False)

    return book_text, ner_entities_words


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
              for w in word_tokenize(re.sub(r'[^a-zA-Z0-9]', ' \g<0> ', 
                                            ' '.join(re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', book_text))
                                           ).replace('cannot', 'word_tokenize_splits_cannot_into_2_words'))]
    
    word_window = book_words[max(total_word_index - window_size, 0): total_word_index + window_size]
    return re.sub(r' ([^a-zA-Z0-9]) ', '[_]\g<0>[_] ', ' '.join(word_window)).replace('[_] ', '')

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
              for w in word_tokenize(re.sub(r'[^a-zA-Z0-9]', ' \g<0> ', 
                                            ' '.join(re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', book_text))
                                           ).replace('cannot', 'word_tokenize_splits_cannot_into_2_words'))]
    
    result = []
    for total_word_index in tqdm(total_word_indexes):
        word_window = book_words[max(total_word_index - window_size, 0): total_word_index + window_size]
        result.append(re.sub(r' ([^a-zA-Z0-9]) ', '[_]\g<0>[_] ', ' '.join(word_window)).replace('[_] ', ''))
    
    return result