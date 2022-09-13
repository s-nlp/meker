import numpy as np
from tqdm import tqdm
import wikidata
import pickle
from wikidata.client import Client
from ner import get_nouns
from wikidataintegrator import wdi_core

def presearch_sq(max_presearch, questions, type_='val'):
    client = wikidata.client.Client()

    candidate_ents = []
    for q in tqdm(questions):
        cands = []
        for noun in get_nouns(q):
            cands.extend(wdi_core.WDItemEngine.get_wd_search_results(noun)[0:max_presearch])
        candidate_ents.append(cands)
    candidate_ents = np.array(candidate_ents)
    np.save(f'data/candidate_entities_sq_{type_}.npy', candidate_ents)

    candidate_ents = np.load(f'data/candidate_entities_sq_{type_}.npy', allow_pickle=True)

    def flatten(xss):
        return [x for xs in xss for x in xs]

    total_ents = set(flatten(candidate_ents))

    entity_neighborhoods = {}

    for entity in tqdm(list(total_ents)):
        loaded_entity = client.get(entity, load = True)

        props_with_objects = []
        for prop in list(loaded_entity):
            try:
                if type(prop) is wikidata.entity.Entity and type(loaded_entity[prop]) is wikidata.entity.Entity:
                    props_with_objects.append((str(loaded_entity[prop].id),str(prop.id)))
            except:
                pass

        entity_neighborhoods[entity] = props_with_objects

    with open(f'data/entity_subgraphs_sq_{type_}.pickle', 'wb') as handle:
        pickle.dump(entity_neighborhoods, handle, protocol=pickle.HIGHEST_PROTOCOL)

    presearched = []
    for cands in candidate_ents:
        q_presearched = {}
        for cand in cands:
            q_presearched[cand] = entity_neighborhoods[cand]
        presearched.append(q_presearched)
    
    presearched = np.array(presearched)
    
    np.save(f'data/presearched_fixed_sq_{type_}.npy', presearched)
    
def presearch_rubq(max_presearch, questions, type_='val'):
    client = wikidata.client.Client()

    candidate_ents = []
    for q in tqdm(questions):
        cands = []
        for noun in get_nouns(q):
            cands.extend(wdi_core.WDItemEngine.get_wd_search_results(noun)[0:max_presearch])
        candidate_ents.append(cands)
    candidate_ents = np.array(candidate_ents)
    np.save(f'data/candidate_entities_rubq_{type_}.npy', candidate_ents)

    candidate_ents = np.load(f'data/candidate_entities_rubq_{type_}.npy', allow_pickle=True)

    def flatten(xss):
        return [x for xs in xss for x in xs]

    total_ents = set(flatten(candidate_ents))

    entity_neighborhoods = {}

    for entity in tqdm(list(total_ents)):
        loaded_entity = client.get(entity, load = True)

        props_with_objects = []
        for prop in list(loaded_entity):
            try:
                if type(prop) is wikidata.entity.Entity and type(loaded_entity[prop]) is wikidata.entity.Entity:
                    props_with_objects.append((str(loaded_entity[prop].id),str(prop.id)))
            except:
                pass

        entity_neighborhoods[entity] = props_with_objects

    with open(f'data/entity_subgraphs_rubq_{type_}.pickle', 'wb') as handle:
        pickle.dump(entity_neighborhoods, handle, protocol=pickle.HIGHEST_PROTOCOL)

    presearched = []
    for cands in candidate_ents:
        q_presearched = {}
        for cand in cands:
            q_presearched[cand] = entity_neighborhoods[cand]
        presearched.append(q_presearched)
    
    presearched = np.array(presearched)
    
    np.save(f'data/presearched_fixed_rubq_{type_}.npy', presearched)