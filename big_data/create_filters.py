from general_functions1 import create_filter, hr

import pickle


def main():
    
    path_data = "Link_Prediction_Data/RuBIQ8M/"
    
    

    train_triples = pickle.load(open(path_data + 'train', 'rb'))
    valid_triples = pickle.load(open(path_data + 'valid', 'rb'))
    test_triples = pickle.load(open(path_data + 'test', 'rb'))
    
    print ("start to create filters")
    
    test_filter = create_filter(test_triples, test_triples + valid_triples + train_triples)

    with open('/notebook/data/Link_Prediction_Data/RuBIQ8M/test_filter.pkl', 'wb') as handle:
        pickle.dump(test_filter, handle)
        
    valid_filter = create_filter(valid_triples, test_triples + valid_triples + train_triples)

    with open('/notebook/data/Link_Prediction_Data/RuBIQ8M/valid_filter.pkl', 'wb') as handle:
        pickle.dump(valid_filter, handle)
    
if __name__ == "__main__":
    main()
