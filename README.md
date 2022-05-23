### MEKER embedding modelling

This is an example code for MEKER embedding modelling in python.

Create proper conda enviroment
* conda env create -f MEKER_env.yml
* conda activate MEKER_env

To run the Fb15k237 experiment:

* install requirements.txt OR create proper conda enviroment
* cd ./gpu
* run python3 gcp_torch.py  --batch_size=156 --how_many=6 --l2=0 --lr=0.01 --n_epoch=65 --opt_type=adamw --scheduler_gamma=0.8 --scheduler_step=3 --seed 55

To run the big dataset experiment:

* install requirements.txt OR create proper conda enviroment
* create folder with train, test, valid triples and filters
* cd ./big_data
* specify in gcp_torch.py path to folder with data
* run python3 gcp_torch.py


Wiki5m dataset (entity and relation mapping to tensor indexes, triples and filters) is on https://zenodo.org/deposit/6574179.
