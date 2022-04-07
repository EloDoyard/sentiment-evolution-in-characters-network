import collections
import csv
import fuzzy
from nameparser.config import Constants
from nameparser import HumanName
from pyjarowinkler.distance import get_jaro_distance
from libindic.soundex import Soundex
import editdistance
import string


honorifics = ['Ab"d', 'Admo"r', 'Adv', 'Advocate', 'BDS', 'Baron', 'Baroness', 'Baronet', 
              'Br', 'Brother', 'Cantor', 'Chancellor', 'Chief Executive', 'Chief Rabbi', 
              'Cl', 'Counsel', 'Countess', 'DDS', 'DMD', 'DO', 'DPhil', 'DVM', 'Dame', 
              'Dean', 'Director', 'Doc', 'Doctor', 'Dr', 'Earl', 'EdD', 'Elder', 'Emi', 
              'Eminent', 'Esq', 'Esq.', 'Esquire', 'Eur Ing', 'Excellence', 'Excellency', 
              'Father', 'Fr', 'Gaava"d', 'Gentleman', 'Grand Rabbi', 'HAH', 'HE', 'HH', 
              'HMEH', 'Her Excellence', 'Her Excellency', 'Her Honour', 'His All Holiness', 
              'His Beatitude', 'His Eminence', 'His Excellence', 'His Excellency', 'His Grace', 
              'His Holiness', 'His Honour', 'His Most Eminent Highness', 'Holy Father', 'Hon.', 
              'Hāfiz', 'Hāfizah', 'Hājī', 'Imām', 'KC', "King\'s Counsel", 'Lady', 'Lord', 
              'Lordship', 'MBBS', 'MBChB', 'MD', "Ma\'am", 'Madam', 'Marchioness', 'Marquess', 
              'Master', 'Mawlānā', 'Miss', 'Missis', 'Missus', 'Mister', 'Mistress', 
              'Most Reverend Eminence', 'Most Reverend Excellency', 'Mr', 'Mrs', 'Mrs.', 
              'Ms', 'Ms.', 'Muftī.', 'Mx', 'My Lady', 'My Lord', 'Mz.', 'Nun', 'OD', 'Pastor', 
              'PhD', 'PharmD', 'Pope', 'Pope Emeritus', 'Pr', 'President', 'Principal', 'Prof', 
              'Professor', 'Provost', 'QC', "Queen\'s Counsel", 'Qārī', 'Raava"d', 'Rabbi', 
              'Rav', 'Rebbe', 'Rebbetzin', 'Rector', 'Regent', 'Rev.', 'Reverend', 'Rt Hon', 
              'SCl', 'Saint', 'Sayyid', 'Sayyidah', 'Senior Counsel', 'Sharif', 'Shaykh', 
              'Sir', 'Sire', 'Sis', 'Sister', 'Sr', "The Hon\'ble", 'The Hon.', 'The Honorable', 
              'The Honourable', 'The Most Blessed', 'The Most Honourable', 'The Most Rev', 
              'The Most Revd', 'The Most Reverend', 'The Rev', 'The Revd', 'The Reverend', 
              'The Right Honourable', 'The Right Reverend', 'The Rt Rev', 'Ven', 'Venerable', 
              'Vice-Chancellor', 'Viscount', 'Viscountess', 'Warden', 'Yeoman', 'Your All Holiness', 
              'Your Beatitude', 'Your Eminence', 'Your Excellence', 'Your Excellency', 
              'Your Grace', 'Your Holiness', 'Your Honour', 'Your Ladyship', 'Your Most Eminent Highness']

## french_honorifics = ['Mr', 'Mrs','Miss','Ms','Dr','Amiral','Air Comm','Ambassadeur', 'Ambassadrice','Baron','Baronne', 'Brigadier', 'Brigadière', 'Frère', 'Chanoine','Capitaine', 'Chef','Cllr','Colonel','Commandeur','Commandeur & Mme','Consul', 'Consule', 'Consul Général', 'Consule Générale','Comte','Comtesse', 'Comtesse de','Cpl','Dame', 'Député','Dr & Mme', 'Drs', 'Duchesse','Duc','Père','Général','Sa Grâce','Son Excellence','Ing', 'Ingénieur', 'Ingénieure','Juge','Justice', 'Lady', 'Madame','Lic','Llc','Lord','Lord & Lady','Lt','Lt Col','Lt Cpl','M','Major', 'Major Général', 'Marquise', 'Marquis', 'Ministre','Mme','M & Dr','M & Mme','M & Miss','Prince','Princesse','Professeur', 'Professeure','Prof','Prof & Dr', 'Prof & Mme','Prof & Rev','Prof Dame','Prof Dr','Pvt','Rabbin','Contre-amiral','Rev','Rev & Mme','Rev Chanoine', 'Rev Dr','Senateur', 'Senatrice','Sgt', 'Shérif', 'Shériffe','Sir','Sir & Lady','Soeur','Sqr. Leader','Le Comte de', "L'Honorable","L'Honorable Dr", "L'Honorable Dame", "Monseigneur", "L'Honorable Monseigneur", "L'honorable Madame", "L'honorable Monsieur",'The Honourable','Le très honorable','Le très honorable Docteur', 'Le très honorable monseigneur','le très honorable monsieur','le très honorable vicomte','vicomte', 'vicomtesse']

french_gender_from_honorifics = {'Mr':'M', 'Mrs':'F','Miss':'F','Ms':'F','Dr':'M', 'Docteur':None, 'Doctoresse':'F' ,'Amiral':'M', 'Amirale':'F','Ambassadeur':'M', 'Ambassadrice':'F','Baron':'M','Baronne':'F', 'Brigadier':'M', 'Brigadière':'F', 'Frère':'M', 'Chanoine':None,'Capitaine':None, 'Chef':'M', 'Cheffe' : 'F','Colonel':None,'Commandeur':None,'Commandeur & Mme':None,'Consul':'M', 'Consule':'F', 'Consul Général':'M', 'Consule Générale':'F' ,'Comte':'M','Comtesse':'F', 'Comtesse de':'F','Dame':'F', 'Député':None,'Dr & Mme':None,
        'Duchesse':'F','Duc':'M','Père':'M', 'Mère':'F', 'Général':None,'Sa Grâce':None,'Son Excellence':None,'Ing' : None, 'Ingénieur' : 'M', 'Ingénieure':'F','Juge':None,'Justice':None, 'Lady':'F', 'Madame':'F','Lord':'M','Lord & Lady' :None, 'Lieutenant':None,'Lieutenant Colonel' :None,'Caporal':None, 'Lieutenant Caporal' : None,'M':'M','Major':None, 'Major Général':None, 'Marquise' : 'F', 'Marquis':'M',
        'Ministre' : None,'Mme':'F','M & Dr':None,'M & Mme':None,'M & Miss':None,'Prince' : 'M','Princesse':'F', 'Professeur':None, 'Professeure' :'F','Prof':None,'Prof & Dr':None,
        'Prof & Mme':None,'Prof & Rev':None,'Révérant':'M','Révérante':'F','Prof Dame':'F','Prof Dr':None,'Rabbin':None ,'Contre-amiral':None,'Rev':None,'Rev & Mme':None,'Rev Chanoine':None,
        'Rev Dr':None,'Sénateur':'M', 'Sénatrice':'F', 'Shérif':None, 'Shériffe':'F','Sir':'M','Sir & Lady':None,'Soeur':'F', 'tante':'F', 'oncle':'M','Le Comte de':'M',
        "L'Honorable":None,"L'Honorable Dr":None, "L'Honorable Dame":'F', "Monseigneur":'M', "L'Honorable Monseigneur":'M', "L'honorable Madame":'F',
        "L'honorable Monsieur":'M','Le très honorable':'M', 'La très honorable':'M','Le très honorable Docteur':'M','La très honorable docteur':'F',
        'Le très honorable monseigneur':'M','le très honorable monsieur':'M','le très honorable vicomte':'M','vicomte':'M', 'vicomtesse':'F', 'la très honorable vicomtesse':'F', 'M de la': 'M', 'M de':'M', 'Mlle de la':'F', 'M de la':'M', 'Mme de': 'F', 'Mme de la':'F', 'Abbé':'M', 'Marquis de la':'M', 'Marquis de':'M', 'Marquise de la':'F', 'Marquise de':'F','Messieurs':'M'
                                ,'Abesse':'F'}
french_honorific = french_gender_from_honorifics.keys()
        
gender_from_honorific = {'Ab"d': 'M', 
                         'Admo"r': 'M', 
                         'Adv': 'M', 
                         'Advocate': 'M', 
                         'BDS': 'M', 
                         'Baron': 'M', 
                         'Baroness': 'F', 
                         'Baronet': 'M', 
                         'Br': 'M', 
                         'Brother': 'M', 
                         'Cantor': 'M', 
                         'Chancellor': None, 
                         'Chief Executive': None, 
                         'Chief Rabbi': 'M',  
                         'Cl': 'M', 
                         'Counsel': 'M', 
                         'Countess': 'F', 
                         'DDS': None, #'M', 
                         'DMD': None, #'M', 
                         'DO': None, #'M', 
                         'DPhil': None, 
                         'DVM': None, #'M', 
                         'Dame': 'F', 
                         'Dean': None,
                         'Director': None,
                         'Doc': None, 'Doctor': None, 'Dr': None, 
                         'Earl': 'M', 
                         'EdD': None, 
                         'Elder': None, 
                         'Emi': 'M', 
                         'Eminent': 'M', 
                         'Esq': 'M', 
                         'Esq.': 'M', 
                         'Esquire': 'M', 
                         'Eur Ing': None, #'M',
                         'Excellence': None, 
                         'Excellency': None, 
                         'Father': 'M', 
                         'Fr': 'M', 
                         'Gaava"d': 'M', 
                         'Gentleman': 'M', 
                         'Grand Rabbi': 'M', 
                         'HAH': 'M',
                         'HE': 'M', 
                         'HH': 'M', 
                         'HMEH': 'M', 
                         'Her Excellence': 'F', 
                         'Her Excellency': 'F', 
                         'Her Honour': 'F', 
                         'His All Holiness': 'M', 
                         'His Beatitude': 'M', 
                         'His Eminence': 'M', 
                         'His Excellence': 'M', 
                         'His Excellency': 'M', 
                         'His Grace': 'M', 
                         'His Holiness': 'M', 
                         'His Honour': 'M', 
                         'His Most Eminent Highness': 'M', 
                         'Holy Father': 'M', 
                         'Hon.': 'M', 
                         'Hāfiz': 'M', 
                         'Hāfizah': 'F', 
                         'Hājī': 'M', 
                         'Imām': 'M', 
                         'KC': 'M', 
                         "King\'s Counsel": None, #'M', 
                         'Lady': 'F', 
                         'Lord': 'M', 
                         'Lordship': 'M', 
                         'MBBS': 'M', 
                         'MBChB': 'M', 
                         'MD': None, 
                         "Ma\'am": 'F', 
                         'Madam': 'F', 
                         'Marchioness': 'F', 
                         'Marquess': 'M', 
                         'Master': 'M', 
                         'Mawlānā': 'M', 
                         'Miss': 'F', 
                         'Missis': 'F', 
                         'Missus': 'F', 
                         'Mister': 'M', 
                         'Mistress': 'F', 
                         'Most Reverend Eminence': 'M', 
                         'Most Reverend Excellency': 'M', 
                         'Mr': 'M', 
                         'Mr.': 'M', 
                         'Mrs': 'F', 
                         'Mrs.': 'F', 
                         'Ms': 'F', 
                         'Ms.': 'F', 
                         'Muftī.': 'M', 
                         'Mx': None, 
                         'Mx.': None, 
                         'My Lady': 'F', 
                         'My Lord': 'M', 
                         'Mz.': 'M', 
                         'Nun': 'F', 
                         'OD': 'M', 
                         'Pastor': 'M', 
                         'PhD': None, 
                         'PharmD': None, 
                         'Pope': 'M', 
                         'Pope Emeritus': 'M', 
                         'Pr': 'M', 
                         'President': 'M', 
                         'Principal': 'M', 
                         'Prof': 'M', 
                         'Professor': 'M', 
                         'Provost': 'M', 
                         'QC': 'M', 
                         "Queen\'s Counsel": None, #'M', 
                         'Qārī': 'M', 
                         'Raava"d': 'M', 
                         'Rabbi': 'M', 
                         'Rav': 'M', 
                         'Rebbe': 'M', 
                         'Rebbetzin': 'M', 
                         'Rector': 'M', 
                         'Regent': 'M', 
                         'Rev.': 'M', 
                         'Reverend': 'M', 
                         'Rt Hon': 'M', 
                         'SCl': 'M', 
                         'Saint': 'M', 
                         'Sayyid': 'M', 
                         'Sayyidah': 'F', 
                         'Senior Counsel': 'M', 
                         'Sharif': 'M', 
                         'Shaykh': 'M', 
                         'Sir': 'M', 
                         'Sire': 'M', 
                         'Sis': 'F', 
                         'Sister': 'F', 
                         'Sr': 'F', 
                         "The Hon\'ble": 'M', 
                         'The Hon.': 'M', 
                         'The Honorable': 'M', 
                         'The Honourable': 'M', 
                         'The Most Blessed': 'M', 
                         'The Most Honourable': 'M', 
                         'The Most Rev': 'M', 
                         'The Most Revd': 'M', 
                         'The Most Reverend': 'M', 
                         'The Rev': 'M', 
                         'The Revd': 'M', 
                         'The Reverend': 'M', 
                         'The Right Honourable': 'M', 
                         'The Right Reverend': 'M', 
                         'The Rt Rev': 'M', 
                         'Ven': None, 
                         'Venerable': None,
                         'Vice-Chancellor': None, #'M', 
                         'Viscount': 'M', 
                         'Viscountess': 'F', 
                         'Warden': 'M', 
                         'Yeoman': 'M', 
                         'Your All Holiness': 'M', 
                         'Your Beatitude': 'M', 
                         'Your Eminence': 'M', 
                         'Your Excellence': None, 
                         'Your Excellency': None, 
                         'Your Grace': 'M', 
                         'Your Holiness': 'M', 
                         'Your Honour': None, 
                         'Your Ladyship': 'F', 
                         'Your Most Eminent Highness': None,
                        }

male_names = []
with open('../male_names.txt', 'r') as f:
    male_names = [*map(str.strip, f.readlines())]

female_names = []
with open('../female_names.txt', 'r') as f:
    female_names = [*map(str.strip, f.readlines())]

class NameDenormalizer(object):
    def __init__(self, filename=None):
        filename = filename or '../french_names.csv'
        lookup = collections.defaultdict(list)
        with open(filename) as f:
            reader = csv.reader(f)
            for line in reader:
                matches = set(line)
                for match in matches:
                    lookup[match].append(matches)
        self.lookup = lookup

    def __getitem__(self, name):
        name = name.lower()
        if name not in self.lookup:
            raise KeyError(name)
        names = set().union(*self.lookup[name])
        if name in names:
            names.remove(name)
        return names

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default
        
constants = Constants()
constants.titles.add(*[h for h in french_honorific])

french_stopwords = []
with open('../data/stopwords-fr.txt', 'r') as f:
    french_stopwords = [*map(str.strip, f.readlines())]
    
# stop_words = list(get_stop_words('en')) # about 900 stopwords
# nltk_words = list(stopwords.words('english')) # about 150 stopwords
# stop_words.extend(nltk_words)


def count_reduce(obj1, obj2):
    '''Applies a reductor to the two tuples passed, counting their total number.

    Parameters
    ----------
    obj1 : tuple
        A (key, count) tuple
    obj2 : tuple
        A (key, count) tuple

    Returns
    -------
    new_tup : tuple
        The reduced (key, count) tuple
    '''
    return (obj1[0], obj1[1], obj1[2]+obj2[2])

def remove_non_ascii_chars(n):
    '''Removes non-ascii characters from a given string.

    Parameters
    ----------
    n : str
        The string from which to remove non-ascii characters

    Returns
    -------
    new_n : str
        The string with only ascii characters
    '''
    printable = set(string.printable)
    
    n = n.replace('‘', '\'').replace('’', '\'')
    return ''.join(filter(lambda x: x in printable, n))

def is_nickname(n1, n2):
    '''Given two names, checks whether any of them is a nickname or diminutive
    form of the other.

    Parameters
    ----------
    n1 : str
        A name
    n2 : str
        A name

    Returns
    -------
    is_nickname : bool
        Whether any of them is a nickname or diminutive form of the other
    '''
    
    nickname_finder = NameDenormalizer()
    n1_alts = nickname_finder.get(n1)
    n2_alts = nickname_finder.get(n2)
    
    if n1_alts != None:
        if any([n1_alt == n2_form for n1_alt in n1_alts for n2_form in re.split('[^a-zA-ZÀ-ÿ]', n2)]):
            return True
    if n2_alts != None:
        if any([n2_alt == n1_form for n2_alt in n2_alts for n1_form in re.split('[^a-zA-ZÀ-ÿ]', n1)]):
            return True
    
    return False

def phonetically_matches(n1, n2, stricter=True):
    '''Given two names, checks whether they phonetically match (i.e. are phonetically
    very similar).

    Parameters
    ----------
    n1 : str
        A name
    n2 : str
        A name
    stricter : bool, optional
        A flag indicating whether the names have to pass all tests (True) or at least 
        just one of them (default is True)

    Returns
    -------
    is_nickname : bool
        Whether the two names phonetically match
    '''
    
    soundex = Soundex()
    dmeta = fuzzy.DMetaphone()
    
    if stricter:
        return (soundex.compare(n1, n2) >= 0)and\
               (editdistance.eval(fuzzy.nysiis(n1), fuzzy.nysiis(n2)) <= 1) and\
               (editdistance.eval(dmeta(remove_non_ascii_chars(n1))[0], dmeta(remove_non_ascii_chars(n2))[0]) <= 1)
    
    return (soundex.compare(n1, n2) >= 0) or\
           (editdistance.eval(fuzzy.nysiis(n1), fuzzy.nysiis(n2)) <= 1) or\
           (editdistance.eval(dmeta(remove_non_ascii_chars(n1))[0], dmeta(remove_non_ascii_chars(n2))[0]) <= 1)
    
def textually_matches(n1, n2, stricter=True):
    '''Given two names, checks whether they textually match (i.e. are textually
    very similar).

    Parameters
    ----------
    n1 : str
        A name
    n2 : str
        A name
    stricter : bool, optional
        A flag indicating whether the names have to pass all tests (True) or at least 
        just one of them (default is True)

    Returns
    -------
    is_nickname : bool
        Whether the two names textually match
    '''
    
    if stricter:
        return (get_jaro_distance(n1, n2) > 0.85) and\
               (editdistance.eval(n1, n2) <= 1)

    return (get_jaro_distance(n1, n2) > 0.85) or\
           (editdistance.eval(n1, n2) <= 1)

def names_fuzzy_match(n1, n2):
    '''Given two names, checks whether they fuzzily match (whether they are 
    considered very similar by any metric - whether they are equal strings,
    a nickname or diminutive of the other, or phonetically or textually very
    similar).

    Parameters
    ----------
    n1 : str
        A name
    n2 : str
        A name

    Returns
    -------
    is_nickname : bool
        Whether the two names textually match
    '''
    
    if n1.lower() == n2.lower():
        return True
    
    # handle nicknames
    if is_nickname(n1, n2):
        return True
    
    # handle phonetic matching
    if phonetically_matches(n1, n2):
        return True
    
    # handle similar strings (string distances)
    if textually_matches(n1, n2):
        return True
    
    return False

def get_gender(honorific, names):
    '''Given an entity's honorific and names, tries to extract their gender.

    Parameters
    ----------
    honorific : str
        The entity's honorific
    names : list
        A list of the entity's names

    Returns
    -------
    gender : str
        The inferred gender ('M' or 'F') or None if unable to infer it
    '''
    
    if honorific != '' and honorific.title() in french_gender_from_honorifics:
        return french_gender_from_honorifics[honorific.title()] 
    for name in names:
        if name != '':
            if name.title() in male_names:
                return 'M'
            if name.title() in female_names:
                return 'F'
    return None

class BookListEntity:
    '''
    A class used to represent an entity as a list of honorifics, first, middle
    and last names, suffixes, nicknames, all their addresses and their inferred
    gender.

    ...

    Attributes
    ----------
    honorific : list
        a list of the entity's honorifics
    first_name : list
        a list of the entity's first names
    middle_name : list
        a list of the entity's middle names
    last_name : list
        a list of the entity's last names
    suffix : list
        a list of the entity's suffixes
    nickname : list
        a list of the entity's nicknames
    gender : str
        the entity's inferred gender
    all_addresses : list
        a list of the all the entity's addresses and mentions
        
    Methods
    -------
    get_universal_address()
        Returns the entity's "short name" string representation, a consistent but
        more readable output than simply stringifying the object.
    exactly_references_entity(check_name)
        Checks whether the given name exactly references this entity.
    textually_close_references_entity(check_name)
        Checks whether the given name matches this entity in a textually-close manner.
    references_entity(check_name)
        Checks whether the given name matches this entity in a general, approximated, 
        manner. Still requires a very high level of similarity, however.
    fuzzy_references_entity(check_name)
        Checks whether the given name matches this entity in a fuzzy manner,
        by comparing name comparisons not covered by stricter functions.
    merge(new_name, strictness='high')
        Merges a new form of address to the current entity.
    '''
    
    def __init__(self, name_str):
        '''
        Parameters
        ----------
        name_str : str
            The entity's name
        '''
        
        hn = HumanName(name_str, constants=constants)
        
        self.honorific = [t.lower() for t in hn.title_list]
        self.first_name = [n.lower() for n in hn.first_list]
        self.middle_name = [n.lower() for n in hn.middle_list]
        self.last_name = [n.lower() for n in hn.last_list]
        self.suffix = [s.lower() for s in hn.suffix_list]
        self.nickname = [n.lower() for n in hn.nickname_list]
        self.gender = get_gender(self.honorific[0] if self.honorific else '', [n for n in self.first_name + 
                                                                       self.middle_name + self.nickname])
        self.all_addresses = [name_str]
        
    def __str__(self):
        '''Returns the BookListEntity's string representation.

        Returns
        -------
        ent_str : str
            The BookListEntity's string representation
        '''
        return f"""<BookListEntity : [
	honorific: '{self.honorific}'
	first: '{self.first_name}' 
	middle: '{self.middle_name}' 
	last: '{self.last_name}' 
	suffix: '{self.suffix}'
	nickname: '{self.nickname}'
	gender: '{self.gender}'
	all adresses: '{self.all_addresses}'
]>"""
    
    def get_universal_address(self):
        '''Returns the entity's "short name" string representation, a consistent but
        more readable output than simply stringifying the object.

        Returns
        -------
        address : str
            the entity's short name
        '''
        address = ''
        if self.honorific != []:
            address = self.honorific[0] + ' '
        if self.first_name != []:
            valid_names = [first_n for first_n in self.first_name if first_n != '']
            if valid_names != []:
                address = address + valid_names[0] + ' '
        if self.last_name != []:
            valid_names = [last_n for last_n in self.last_name if last_n != '']
            if valid_names != []:
                address = address + valid_names[0] + ' '
        return address
    
    def exactly_references_entity(self, check_name):
        '''Checks whether the given name exactly references this entity.

        Parameters
        ----------
        check_name : str
            The name against which to check

        Returns
        -------
        matches : bool
            Whether the given name exactly references this entity
        '''
    
        hn = HumanName(check_name, constants=constants)
        check_name_gender = get_gender(hn.title, [n for n in hn.first_list + hn.middle_list + hn.nickname_list])

        # first, if possible, check gender matches
        if self.gender != None and check_name_gender != None and self.gender != check_name_gender:
            return False
        
        # account for all names
        last_name_matches = [(last_names.lower() == check_vals.lower())# or textually_matches(last_names, check_vals) 
                              for last_names in self.last_name
                                for check_vals in [hn.first, hn.middle, hn.nickname, hn.last] 
                                if last_names != '' and check_vals != '']
        first_name_matches = [(first_names.lower() == check_vals.lower())# or textually_matches(first_names, check_vals) 
                              for first_names in self.first_name
                                for check_vals in [hn.first, hn.middle, hn.nickname, hn.last] 
                                if first_names != '' and check_vals != '']
        middle_name_matches = [(middle_names.lower() == check_vals.lower())# or textually_matches(middle_names, check_vals) 
                               for middle_names in self.middle_name
                                for check_vals in [hn.first, hn.middle, hn.nickname, hn.last] 
                                if middle_names != '' and check_vals != '']
        nickname_matches = [(nicknames.lower() == check_vals.lower())# or textually_matches(nicknames, check_vals) 
                            for nicknames in self.nickname
                                for check_vals in [hn.first, hn.middle, hn.nickname, hn.last] 
                                if nicknames != '' and check_vals != '']
        
        
        if (last_name_matches != [] and any(last_name_matches)) or\
           (first_name_matches != [] and any(first_name_matches)) or\
           (middle_name_matches != [] and any(middle_name_matches)) or\
           (nickname_matches != [] and any(nickname_matches)):
            return True
        
        honorific_matches = [(hn.title.lower() == honorific.lower()) for honorific in self.honorific]
        if (last_name_matches == [] and first_name_matches == [] and 
            middle_name_matches == [] and nickname_matches == []) and any(honorific_matches):
            return True

        return False
        
    def textually_close_references_entity(self, check_name):
        '''Checks whether the given name matches this entity in a textually-close manner.

        Parameters
        ----------
        check_name : str
            The name against which to check

        Returns
        -------
        matches : bool
            Whether the given name matches this entity in a textually-close manner
        '''
        
        hn = HumanName(check_name, constants=constants)
        check_name_gender = get_gender(hn.title, [n for n in hn.first_list + hn.middle_list + hn.nickname_list])

        # first, if possible, check gender matches
        if self.gender != None and check_name_gender != None and self.gender != check_name_gender:
            return False
        
        # account for all names
        last_name_matches = [textually_matches(last_names.lower(), check_vals.lower()) 
                              for last_names in self.last_name
                                for check_vals in [hn.first, hn.middle, hn.nickname, hn.last] 
                                if last_names != '' and check_vals != '']
        first_name_matches = [textually_matches(first_names.lower(), check_vals.lower()) 
                              for first_names in self.first_name
                                for check_vals in [hn.first, hn.middle, hn.nickname, hn.last] 
                                if first_names != '' and check_vals != '']
        middle_name_matches = [textually_matches(middle_names.lower(), check_vals.lower()) 
                               for middle_names in self.middle_name
                                for check_vals in [hn.first, hn.middle, hn.nickname, hn.last] 
                                if middle_names != '' and check_vals != '']
        nickname_matches = [textually_matches(nicknames.lower(), check_vals.lower())
                            for nicknames in self.nickname
                                for check_vals in [hn.first, hn.middle, hn.nickname, hn.last] 
                                if nicknames != '' and check_vals != '']
        
        if (last_name_matches != [] and any(last_name_matches)) or\
           (first_name_matches != [] and any(first_name_matches)) or\
           (middle_name_matches != [] and any(middle_name_matches)) or\
           (nickname_matches != [] and any(nickname_matches)):
            return True
        
        honorific_matches = [(hn.title.lower() == honorific.lower()) for honorific in self.honorific]
        if (last_name_matches == [] and first_name_matches == [] and 
            middle_name_matches == [] and nickname_matches == []) and any(honorific_matches):
            return True

        return False
        
    def references_entity(self, check_name):
        '''Checks whether the given name matches this entity in a general, approximated, 
        manner. Still requires a very high level of similarity, however.

        Parameters
        ----------
        check_name : str
            The name against which to check

        Returns
        -------
        matches : bool
            Whether the given name matches this entity in a general, approximated, 
            manner
        '''
        
        hn = HumanName(check_name, constants=constants)
        check_name_gender = get_gender(hn.title, [n for n in hn.first_list + hn.middle_list + hn.nickname_list])

        # first, if possible, check gender matches
        if self.gender != None and check_name_gender != None and self.gender != check_name_gender:
            return False
        
        # then, if possible, check last names match
        last_name_match = [names_fuzzy_match(last_name, hn.last) for last_name in self.last_name 
                           if last_name != '' and hn.last != '']
        if last_name_match != [] and not any(last_name_match):
            return False
        
        # then, for all non-last names check that non-null equivalent entries on both sides fuzzy-match
        # (accounting for nicknames and diminuitive names as much as possible)
        first_name_matches = [names_fuzzy_match(first_names, check_vals) for first_names in self.first_name
                                for check_vals in [hn.first, hn.middle, hn.nickname] 
                                if first_names != '' and check_vals != '']
        middle_name_matches = [names_fuzzy_match(middle_names, check_vals) for middle_names in self.middle_name
                                for check_vals in [hn.first, hn.middle, hn.nickname] 
                                if middle_names != '' and check_vals != '']
        nickname_matches = [names_fuzzy_match(nicknames, check_vals) for nicknames in self.nickname
                                for check_vals in [hn.first, hn.middle, hn.nickname] 
                                if nicknames != '' and check_vals != '']
        
        
        if (first_name_matches != [] and any(first_name_matches)) or\
           (middle_name_matches != [] and any(middle_name_matches)) or\
           (nickname_matches != [] and any(nickname_matches)) or\
           (hn.first == '' and  hn.middle == '' and hn.nickname == '' and 
            (self.first_name == [''] or self.first_name == []) and 
            (self.middle_name == [''] or self.middle_name == []) and 
            (self.nickname == [''] or self.nickname == [])):
            return True

        # to do the merging below, we first need to extract gender from name as much as possible
        # and make it a necessary condition that genders match
        if (((self.gender != None and check_name_gender != None and self.gender == check_name_gender) 
             and any(last_name_match)) and
            ((self.first_name == [''] or self.first_name == [] or hn.first == '') and 
            (self.middle_name == [''] or self.middle_name == [] or hn.middle == '') and 
            (self.nickname == [''] or self.nickname == [] or hn.nickname == ''))):
            return True

        return False
        
    def fuzzy_references_entity(self, check_name):
        '''Checks whether the given name matches this entity in a fuzzy manner,
        by comparing name comparisons not covered by stricter functions.

        Parameters
        ----------
        check_name : str
            The name against which to check

        Returns
        -------
        matches : bool
            Whether the given name matches this entity in a fuzzy manner
        '''
        
        hn = HumanName(check_name, constants=constants)
        check_name_gender = get_gender(hn.title, [n for n in hn.first_list + hn.middle_list + hn.nickname_list])

        # first, if possible, check gender matches
        if self.gender != None and check_name_gender != None and self.gender != check_name_gender:
            return False
        
        
        # scenario #1 - first name was recognized as last name
        scenario1_match = [names_fuzzy_match(first_name, hn.last) for first_name in self.first_name 
                           if first_name != '' and hn.last != '']
        if scenario1_match != [] and any(scenario1_match):
            return True
        
        # scenario #2 - last name was recognized as first one
        scenario2_match = [names_fuzzy_match(last_name, hn.first) for last_name in self.last_name 
                           if last_name != '' and hn.first != '']
        if scenario2_match != [] and any(scenario2_match):
            return True
        
        # scenario #3 - middle name was recognized as last name
        scenario3_match = [names_fuzzy_match(middle_name, hn.last) for middle_name in self.middle_name 
                           if middle_name != '' and hn.last != '']
        if scenario3_match != [] and any(scenario3_match):
            return True
        
        # scenario #4 - middle name was recognized as first name
        scenario4_match = [names_fuzzy_match(middle_name, hn.first) for middle_name in self.middle_name 
                           if middle_name != '' and hn.first != '']
        if scenario4_match != [] and any(scenario4_match):
            return True
        
        # scenario #5 - last name was recognized as middle name
        scenario5_match = [names_fuzzy_match(last_name, hn.middle) for last_name in self.last_name 
                           if last_name != '' and hn.middle != '']
        if scenario5_match != [] and any(scenario5_match):
            return True
        
        # scenario #6 - first name was recognized as middle name
        scenario6_match = [names_fuzzy_match(first_name, hn.middle) for first_name in self.first_name 
                           if first_name != '' and hn.middle != '']
        if scenario6_match != [] and any(scenario6_match):
            return True
        
        # scenario #7 - nickname was recognized as first name
        scenario7_match = [names_fuzzy_match(nickname, hn.first) for nickname in self.nickname 
                           if nickname != '' and hn.first != '']
        if scenario7_match != [] and any(scenario7_match):
            return True
        
        # scenario #8 - nickname was recognized as last name
        scenario8_match = [names_fuzzy_match(nickname, hn.last) for nickname in self.nickname 
                           if nickname != '' and hn.last != '']
        if scenario8_match != [] and any(scenario8_match):
            return True
        
        return False
        
    def merge(self, new_name, strictness='high'):
        '''Merges a new form of address to the current entity.

        Parameters
        ----------
        new_name : str
            The new form of address
        strictness : str, optional 
            The strictness to enforce in the merge (between 'high', 'fuzzy',
            'exact' and 'textually_close') (default is 'high')
        '''
        
        # check genders, honorifics, and names
        if strictness == 'high' and not self.references_entity(new_name):
            return
        elif strictness == 'fuzzy' and not self.fuzzy_references_entity(new_name):
            return
        elif strictness == 'exact' and not self.exactly_references_entity(new_name):
            return
        elif strictness == 'textually_close' and not self.textually_close_references_entity(new_name):
            retrun
        
#         # check not already taken into account
#         # (commented out because these come in handy for the second class, (non-list) BookEntity)
#         if new_name in self.all_addresses:
#             return
        
        # decide which names are being added to entity
        hn = HumanName(new_name, constants=constants)
        
        if self.gender == None:
            updated_gender = gender_from_honorific[hn.title.title()] if hn.title != '' and\
                                hn.title.title() in honorifics else None
            self.gender = updated_gender
            
        # add all non-null entries of new_names not already present in entity
        if strictness == 'high' or strictness == 'exact':
            if hn.title.lower() not in self.honorific and hn.title != '':
                self.honorific += [hn.title.lower()]
            if hn.first.lower() not in self.first_name and hn.first != '':
                self.first_name += [hn.first.lower()]
            if hn.middle.lower() not in self.middle_name and hn.middle != '':
                self.middle_name += [hn.middle.lower()]
            if hn.last.lower() not in self.last_name and hn.last != '':
                self.last_name += [hn.last.lower()]
            if hn.suffix.lower() not in self.suffix and hn.suffix != '':
                self.suffix += [hn.suffix.lower()]
            if hn.nickname.lower() not in self.nickname and hn.nickname != '':
                self.nickname += [hn.nickname.lower()]
        
        elif strictness == 'fuzzy':
            # accurately match non-misplaceable names
            if hn.title.lower() not in self.honorific and hn.title != '':
                self.honorific += [hn.title.lower()]
            if hn.suffix.lower() not in self.suffix and hn.suffix != '':
                self.suffix += [hn.suffix.lower()]
                
            # fuzzy match all possibly misplaced names
            for name in [hn.first, hn.middle, hn.last, hn.nickname]:
                if name != '':
                    if any([names_fuzzy_match(first_name, name) for first_name in self.first_name 
                               if first_name != '']) and name.lower() not in self.first_name:
                        self.first_name += [name.lower()]
                    if any([names_fuzzy_match(middle_name, name) for middle_name in self.middle_name
                               if middle_name != '']) and name.lower() not in self.middle_name:
                        self.middle_name += [name.lower()]
                    if any([names_fuzzy_match(last_name, name) for last_name in self.last_name 
                               if last_name != '']) and name.lower() not in self.last_name:
                        self.last_name += [name.lower()]
                    if any([names_fuzzy_match(nickname, name) for nickname in self.nickname 
                               if nickname != '']) and name.lower() not in self.nickname:
                        self.nickname += [name.lower()]
                        
        # save added form of address
        self.all_addresses.append(new_name)
        
        
class BookEntity:
    '''
    A class used to represent an entity as a list of honorifics, first, middle
    and last names, suffixes, nicknames, all their addresses and their inferred
    gender.

    ...

    Attributes
    ----------
    honorific : str
        The character's main honorific
    first_name : str
        The character's main first name
    middle_names : list
        A list of the all the entity's middle names
    last_name : str
        The character's main last name
    suffix : str
        The character's main suffix
    nicknames : list
        A list of the all the entity's nicknames
    gender : str
        The character's inferred gender
    all_addresses : list
        A list of the all the entity's addresses and mentions
        
    Methods
    -------
    get_shortname()
        Returns the entity's "short name" string representation, by doing a frequency
        analysis to identify its most common forms of address.
        
    @staticmethod
    from_list_entity(list_book_entity)
        Converts a BookListEntity into a BookEntity one.
    '''
    
    def __init__(self, honorific='', first_name='', middle_names=[], last_name='', suffix='',
                 nicknames=[], gender=None, all_addresses=[]):
        '''
        Parameters
        ----------
        honorific : str, optional
            The character's main honorific (default is '')
        first_name : str, optional
            The character's main first name (default is '')
        middle_names : list, optional
            A list of the all the entity's middle names (default is [])
        last_name : str, optional
            The character's main last name (default is '')
        suffix : str, optional
            The character's main suffix (default is '')
        nicknames : list, optional
            A list of the all the entity's nicknames (default is [])
        gender : str, optional
            The character's inferred gender (default is None)
        all_addresses : list, optional
            A list of the all the entity's addresses and mentions (default is [])
        '''
        
        self.honorific = honorific
        self.first_name = first_name
        self.middle_names = middle_names
        self.last_name = last_name
        self.suffix = suffix
        self.nicknames = nicknames
        self.gender = gender
        self.all_addresses = all_addresses
        
    def __str__(self):
        '''Returns the BookEntity's string representation.

        Returns
        -------
        ent_str : str
            The BookListEntity's string representation
        '''
        return f"""<BookEntity : [
	honorific: '{self.honorific}'
	first: '{self.first_name}' 
	middle: '{' '.join(self.middle_names)}' 
	last: '{self.last_name}' 
	suffix: '{self.suffix}'
	nickname: '{self.nicknames}'
	gender: '{self.gender}'
	all adresses: '{self.all_addresses}'
]>"""

    def get_shortname(self):
        '''Returns the entity's "short name" string representation, by doing a frequency
        analysis to identify its most common forms of address.

        Returns
        -------
        address : str
            the entity's short name
        '''
        return ' '.join([self.honorific, self.first_name, self.last_name]).strip().replace('  ', ' ')
    
    @staticmethod
    def from_list_entity(list_book_entity):
        '''Converts a BookListEntity into a BookEntity one.

        Parameters
        ----------
        list_book_entity : BookListEntity
            The BookListEntity to convert into a BookEntity one
        '''
        
        # prepare data to create new book entity
        # (if several addresses, choose most common one to be the main one)
        honorific = ''
        if list_book_entity.honorific != []:
            honorifics_by_freq = [tup[0] for tup in sorted([(hon, sum([hon in s.lower() 
                                    for s in list_book_entity.all_addresses])) 
                                    for hon in list_book_entity.honorific], key=lambda x: x[1])]
            honorific = honorifics_by_freq[0]
        
        suffix = ''
        if list_book_entity.suffix != []:
            suffixes_freq = [(suf, sum([suf.lower() in s.lower() for s in list_book_entity.all_addresses])) 
                               for suf in list_book_entity.suffix]
            suffix = suffixes_freq[np.argmax(suffixes_freq)][0]
            
        # do an n-gram analysis to get (most common) first and last names, sending the others
        # into either the middle names list or to the nicknames list
        main_names = set(list_book_entity.first_name + list_book_entity.middle_name + 
                        list_book_entity.last_name)
        
        all_addresses_by_freq = [tup[0] for tup in sorted([(address, sum([address.lower() in s.lower() 
                                                                          for s in list_book_entity.all_addresses]))
                                                           for address in list_book_entity.all_addresses], 
                                                          key=lambda x: x[1])]
        
        # if addressed as Mr/Mr. we have an easy way of getting the last name with certainty
        last_name = ''
        for address in all_addresses_by_freq:
            if 'mr' in address.lower():
                last_name = address.split()[-1]
                break
                     
        # handle first, middle and last names
        first_name = ''
        middle_names = []
        nicknames = []
        if len(main_names) > 1:
            if last_name == '':
                if len(list_book_entity.last_name) >= 1:
                    last_names_per_freq = [tup[0] for tup in sorted([(ln, sum([ln.lower() in s.lower() 
                                                                          for s in list_book_entity.all_addresses]))
                                                           for ln in list_book_entity.last_name], 
                                                          key=lambda x: x[1])]
                    last_name = last_names_per_freq[0]
                    
                else: # len(main_names) > 1:
                    end_name_per_freq = [tup[0] for tup in sorted([(name.split()[-1], 
                                                                    sum([name.split()[-1].lower() in s.lower() 
                                                                         for s in list_book_entity.all_addresses]))
                                                           for name in main_names], 
                                                          key=lambda x: x[1])]
                    last_name = end_name_per_freq[0]
            
            # last name is defined, now handle first name
            if len(list_book_entity.first_name) >= 1:
                first_names_per_freq = [tup[0] for tup in sorted([(fn, sum([fn.lower() in s.lower() 
                                                                      for s in list_book_entity.all_addresses]))
                                                       for fn in list_book_entity.first_name
                                                       if fn.lower() != last_name.lower()], 
                                                      key=lambda x: x[1])]
                first_name = first_names_per_freq[0]

            else: # len(main_names) > 1:
                first_name_per_freq = [tup[0] for tup in sorted([(name.split()[0], 
                                                                sum([name.split()[0].lower() in s.lower() 
                                                                     for s in list_book_entity.all_addresses]))
                                                       for all_names in main_names for name in all_names.split()
                                                       if name != '' and 
                                                          name.lower() != last_name.lower() and 
                                                          not any(name in list_book_entity.honorific)], 
                                                      key=lambda x: x[1])]
                first_name = first_name_per_freq[0]
            
            # all remaining names "must be" middle names
            middle_names = [name for name in main_names 
                            if (name.lower() != last_name.lower()) and (name.lower() != first_name.lower())]
            
            nicknames = list_book_entity.nickname
            
        elif len(list_book_entity.nickname) > 1:
            # extract names from nicknames field
            nicknames_per_freq = sorted([(nn, sum([nn.lower() in s.lower() for s in list_book_entity.all_addresses]),
                                          sum([s.index(nn)  for s in list_book_entity.all_addresses if nn in s]))
                                         for nn in list_book_entity.nickname], key=lambda x: x[1])
            
            first_name = nicknames_per_freq[0][0] if (nicknames_per_freq[0][2]/nicknames_per_freq[0][1] > 
                                                      nicknames_per_freq[1][2]/nicknames_per_freq[1][1]) else nicknames_per_freq[1][0]
            last_name = nicknames_per_freq[0][0] if nicknames_per_freq[0][0] != first_name else nicknames_per_freq[1][0]
            
            
            nicknames = [nick for nick in list_book_entity.nickname 
                         if (nick.lower() != last_name.lower()) and (nick.lower() != first_name.lower())]
            
        elif last_name == '':
            only_available_name = (main_names.pop() if len(main_names) == 1 else 
                                   list_book_entity.nickname[0] if len(list_book_entity.nickname) == 1 else
                                   '')
            if honorific == '':
                first_name = only_available_name
            else:
                last_name = only_available_name
        
            
        return BookEntity(honorific.strip().title(), first_name.strip().title(), 
                          [mn.strip().title() for mn in middle_names], 
                          last_name.strip().title(), suffix.strip(), 
                          [nn.strip().title() for nn in nicknames], 
                          list_book_entity.gender, all_addresses_by_freq)