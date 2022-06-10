# Characters Relationship Evolution

This repo is associated with Eloïse Doyard's student master project at EPFL. The subject of this project is "Dictionary of all novels' characters" and the name of her project is "Characters Relationship Evolution".

## Abstract

The task of automatically extracting novels' characters from French literature yet remains unsolved. Moreover, the analysis of the evolution of interpersonal relations between characters using embeddings can reveal interesting results as suggested by the literature. This project is a case study of “Le Rouge et le Noir" written by Stendhal, and has the following goals : implement a pipeline to automatically identify and extract characters of a novel for French literature; and, propose an exploratory analysis of the evolution of relationships between the characters through the study and interpretation of embeddings clusters obtained from interactions between characters. This work gives satisfying results for the extraction of characters, on the one hand, and appears to capture a new interesting way of studying characters relationships, on the other hand.

## Folder structure

Folders:
- `scripts/`: contains all the notebook :
    + `previous project/`: is a folder containing the files of the previous projects on "Dictionary of all novels' characters".
    + `french_character_extraction.ipynb` : notebook for extracting the characters from French Literature
    + `interaction.ipynb`: notebook for extracting interactions between characters and studying the progression of those interactions in different ways.
    + `ner_evaluation.ipynb`: notebook for evaluating the performances of the character extraction.
- `src/`: contains all the python files :
    + `book_entity.py`: functions used in the task of extracting BookEntities
    + `book_entities_embeddings_getters.py`: functions to get different embeddings from different methods 
    + `character_extraction.py`: functions used to perform the characters extraction
    + `embeddings.py`: functions used to compute embeddings and functions around embeddings
    + `evaluation_metrics.py`: functions used in the evaluation of the model for extracting characters from French written novels.
- `data/`: all the data needed to run the code and all the data outputed by the code.
    + `book/` : folder containing the original corpuses of the book studied
    + `book_dfs/`: folder containing the result of the name entity extraction
    + `book_entities/`: folder containing the result of the final characters extraction
    + `dim_reduc/`: folder containing the result of the dimensional reduction of the embeddings for different two-characters relationships
- `report/`: folder containing the report of this project
    + TODO
- `python_parser.py`: python code TODO
- `stendhal.yml`: yalm file containing all the requirements and dependencies used in this project.

## Python imports
To run the code, please install a conda environment using the following command :
```bash
conda env create --file=stendhal.yml
```
