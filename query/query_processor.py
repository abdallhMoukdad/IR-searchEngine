import string
from itertools import chain
from typing import List

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from spellchecker import SpellChecker
from nltk.corpus import wordnet
from textblob import TextBlob

cureent_index = 0

'''   Expand Query:

    Uses wordnet to find synonyms and expand the query.
    Adds synonyms to the original query to broaden the search.

Spell Correction:

    Uses textblob to correct spelling mistakes in the query.
    Ensures the query is free from spelling errors.

Refine Query:

    Combines spell correction and query expansion to refine the query.
    Provides a refined query for better search results.'''


class QueryProcessor:
    def process_query(self, query):
        # Basic query processing (e.g., lowercasing)

        return query.lower()

    def expand_query(self, query):
        expanded_query = query
        for word in query.split():
            synonyms = wordnet.synsets(word)
            lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
            if lemmas:
                expanded_query += ' ' + ' '.join(lemmas)
        return expanded_query

    def suggest_query_refinement(self, query):
        synonyms = set()
        for word in query.split():
            for synset in wordnet.synsets(word):
                synonyms.update(synset.lemma_names())
        return list(synonyms)

    def spell_correction(self, query):
        corrected_query = str(TextBlob(query).correct())
        return corrected_query

    def refine_query(self, query):
        corrected_query = self.spell_correction(query)
        expanded_query = self.expand_query(corrected_query)
        return expanded_query

    def complete_process_query(self, text):
        # text = "The boys are running and the leaves are falling."
        # if (text is list):
        #     [print(t.lower()) for t in text]
        # else:
        #     text.lower()
        # print(text)
        global cureent_index
        cureent_index += 1
        # print(f'\n-------------------------curent_index ={cureent_index}---------------- /n')
        text.lower()
        # print(text)
        # Tokenize into words
        words = word_tokenize(text)
        # print(words)
        words = self.remove_punctuation(words)
        # print(words)

        words = self.remove_stop_wrods(words)
        # print(words)

        # words = self.correct_sentence_spelling(words)
        # print(words)

        # Stemming
        stemmed_words = self.stemming(words)
        return stemmed_words
        # POS tagging
        # pos_tags = pos_tag(words)
        # lemmatized_words = self.lemmatization(words)
        # return lemmatized_words

    def stemming(self, words):
        # Stemming
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in words]

        # print(stemmed_words)
        return stemmed_words

    def lemmatization(self, words):
        # POS tagging
        pos_tags = pos_tag(words)

        # Lemmatization
        lemmatizer = WordNetLemmatizer()

        lemmatized_words = [lemmatizer.lemmatize(word, pos=self.get_wordnet_pos(tag)) for word, tag in pos_tags]

        lemmatized_words
        return lemmatized_words

    def correct_sentence_spelling(self, tokens: List[str]) -> List[str]:
        spell = SpellChecker()
        misspelled = spell.unknown(tokens)
        for i, token in enumerate(tokens):
            if token in misspelled:
                corrected = spell.correction(token)
                if corrected is not None:
                    tokens[i] = corrected
        return tokens

    def get_wordnet_pos(self, tag_parameter):
        tag = tag_parameter[0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def remove_stop_wrods(self, text):
        filtered_text = []

        for word in text:
            if word not in stopwords.words('english'):
                filtered_text.append(word)

        return filtered_text

    def remove_punctuation(self, text: List[str]):
        new_tokens = []
        for token in text:
            new_tokens.append(token.translate(str.maketrans('', '', string.punctuation)))
        return new_tokens
