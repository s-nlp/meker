from wikidataintegrator import wdi_core
from wikidata.client import Client
import wikidata
import en_core_web_sm
nlp = en_core_web_sm.load()

client = wikidata.client.Client()
#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#
#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#
def get_triplets_by_idd(idd):
    '''
    This function returns a list of triples for a given entity by its id
        
        input:  (str) wikidata id, e.x. 'Q2'
        output: (list) list of lists of strings
    '''
    client = wikidata.client.Client()
    entity = client.get(idd, load = True)
    triples = []
    for x in list(entity): # Iterate over properties
        triple = [str(entity.label).lower()]
        prop = client.get(x.id, load = True)
        triple.append(str(prop.label).lower())
        try:
            if type(entity[prop]) is wikidata.entity.Entity:
                triple.append(str(entity[prop].label).lower())
            else:
                triple.append(str(entity[prop]).lower())
        except Exception as e:
            try:
                param = str(e).split("unsupported type: ")[1].replace("'",'"')
                d = json.loads(param)
                triple.append(str(d["value"]["amount"]).lower())
            except:
#                 print (e)
                triple.append("BAD_TYPE")
        triples.append(triple)
    return triples
#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#
#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#

def get_description_name(idd):
    '''
    This function returns a name of an entity and its description given WikiData id
    
        input:  (str) wikidata id, e.x. 'Q2'
        output: (str) concatenated 'name, description' of a given entity
    '''
    entity = client.get(idd, load=True)
    name = "None"
    description = "None"
    try:
        name = entity.data["labels"]["en"]["value"]
        description = entity.data["descriptions"]["en"]["value"]
    except:
        pass
    return name + ", " + description