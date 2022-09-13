from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc)

from natasha.doc import DocSpan, DocToken
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm") 

from wikidata.client import Client
import wikidata

# from IPython.core.debugger import set_trace

class Kostil_phrase_normalization():
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()

        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.doc = None
        
        
        
    def phrase_preprocess(self, phrase):
        self.doc = Doc(phrase)
        self.doc.segment(self.segmenter)
        self.doc.tag_morph(self.morph_tagger)
        
        
        
    def get_tokens(self, phrase, tokens):
        local_tokens = phrase.split()
        result_tokens = []
        for token in tokens:
            if token.text in local_tokens:
                result_tokens.append(token)
        return result_tokens
    
    
    def normalize(self, phrase):
        self.phrase_preprocess(phrase)
        
        tokens = self.get_tokens(phrase, self.doc.tokens)
        span = DocSpan('0', '2', type='LOC', text=phrase, tokens=tokens)
        span.normalize(self.morph_vocab)
        return span.normal

def get_nouns(text):
    text = text.replace('?', '')
    
    doc = nlp(text)
    import spacy

    lemmas = []
    for token in doc:
        lemmas.append(str(token.lemma_))
    lemmatized_text = " ".join(lemmas)
        
    ents = [str(ent) for ent in doc.ents]
    if '"' in text:
        text3 = text[text.find('"')+1:]
        text3 = text3[0:text3.find('"')]
        ents += [str(text3)]
        
    if '«' in text:
        text4 = text[text.find('«')+1:]
        text4 = text4[0:text4.find('»')]
        ents += [str(text4)]
        

    doc = nlp(text)
    ents = [token.lemma_ for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"]

    bigrams = [" ".join(b) for b in zip(text.split(" ")[:-1], text.split(" ")[1:])]
    ents += bigrams
    bigrams = [" ".join(b) for b in zip(lemmatized_text.split(" ")[:-1], lemmatized_text.split(" ")[1:])]
    ents += bigrams

    bigrams = [" ".join(b) for b in zip(text.split(" ")[:-2], text.split(" ")[1:], text.split(" ")[2:])]
    ents += bigrams
    bigrams = [" ".join(b) for b in zip(lemmatized_text.split(" ")[:-2], lemmatized_text.split(" ")[1:], lemmatized_text.split(" ")[2:])]
    ents += bigrams
        
    nouns_set = set(ents)
    if "" in nouns_set:
        nouns_set.remove("")
    return list(nouns_set)


def mp_get_second_hop_entities_by_idd(idd,d):
    try:
        client = wikidata.client.Client()
        entity = client.get(idd, load = True)
        second_hop_qp = []
        for x in list(entity): # Iterate over properties
            prop = client.get(x.id, load = True)
            try:
                if type(prop) is wikidata.entity.Entity and type(entity[prop]) is wikidata.entity.Entity:
                    second_hop_qp.append((str(entity[prop].id),str(prop.id)))
            except:
                pass
        d[idd] = second_hop_qp
    except:
        d[idd] = []