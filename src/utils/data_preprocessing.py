import twitter

from genderize import Genderize

import re

import nltk
nltk.download('punkt')

from nltk.tokenize import TweetTokenizer

import pandas

from spellchecker import SpellChecker

import numpy as np

# Initialization phase to use the Twitter API
api = twitter.Api(
    consumer_key='',
    consumer_secret='',
    access_token_key='',
    access_token_secret=''
)
api.VerifyCredentials()

def tweet_tokenize(df, shuffle=True):
    """
        TODO: documentation
    """
    df.text = df.text.apply(lambda x: str(x).lower())

    # Dataset shuffle 
    if shuffle:
        df = df.sample(frac=1)

    tknzr = TweetTokenizer(reduce_len=True)

    TEMP = df.text.to_numpy()
    DS_X = []
    for entry in TEMP:
        DS_X.append(np.array(tknzr.tokenize(entry)))

    #DS_Y = np.column_stack((df.misogynous.to_numpy(), df.aggressiveness.to_numpy()))

    return DS_X

spell = SpellChecker()
spell.word_frequency.load_dictionary("_utils_files/word_frequency_dictionary.json")

def correct(word):
    """
        TODO: documentation
    """
    return spell.correction(word)

def candidates_words(word):
    """
        TODO: documentation
    """
    return spell.candidates(word)

def __get_gender(names):
    """
        TODO: documentation
    """
    try:
        genders = Genderize().get(names, country_id="IT")
        male = 0
        male_count = 0
        female = 0
        female_count = 0
        for gender in genders:
            if gender['gender'] != None:
                if gender['gender'] == "female":
                    female_count += 1
                    female += gender['probability']
                else:
                    male_count += 1
                    male += gender['probability']
        if male > 0:
            male = male / male_count
        if female > 0:
            female = female / female_count
        if male > 0 or female > 0:
            if female > male:
                return " Donna "
            else:
                return " Uomo "
        else:
            return " PERSONA "
    except:
        print(names)
        return " PERSONARICHIESTAFALLITA "

def profile_tag_apply(x):
    """
        TODO: documentation
    """
    names = []
    try:
        user = api.GetUser(screen_name=x)
        for name in user.name.split(" "):
            names.append(re.sub('[0-9]+', '', name))
    except:
        name_x = re.sub('[^A-Za-z0-9]+', '', x)
        for name in name_x.split(" "):
            names.append(re.sub('[0-9]+', '', name))
    return __get_gender(names)

def profile_tag_processing(df):
    """
        TODO: documentation
    """
    tag_reg = re.compile('@\w+', re.IGNORECASE)
    df.text = df.text.apply(lambda x: re.sub(tag_reg, lambda pattern: profile_tag_apply(pattern.group(0)), str(x)))
    return df

def remove_special_characters(df):
    """
        TODO: documentation
    """
    # Hashtag
    hashtag_reg = re.compile('#', re.IGNORECASE)
    df.text = df.text.apply(lambda x: re.sub(hashtag_reg, ' ', str(x)))

    # Remove special character
    special_characters_reg = re.compile('[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇèàòùì@£+\-=€$%_ ]', re.IGNORECASE)
    df.text = df.text.apply(lambda x: re.sub(special_characters_reg, ' ', str(x)))
    return df

def remove_letters_repetition(df):
    """
        TODO: documentation
    """
    df.text = df.text.apply(lambda x: re.sub(r'(.)(\1)\1+', r"\g<1>\g<2>", x))
    return df

def remove_link(df):
    """
        TODO: documentation
    """
    # Remove link:
        #   The presence of link can induce a unintendend bias due to the fact that
        #   the presence of links is not balanced w.r.t. classes
    link_reg = re.compile(
        '(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
        , 
        re.IGNORECASE
    )
    df.text = df.text.apply(lambda x: re.sub(link_reg, '', str(x)))
    return df