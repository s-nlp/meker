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

### Citation
```
@inproceedings{chekalina-etal-2022-meker,
    title = "{MEKER}: Memory Efficient Knowledge Embedding Representation for Link Prediction and Question Answering",
    author = "Chekalina, Viktoriia  and
      Razzhigaev, Anton  and
      Sayapin, Albert  and
      Frolov, Evgeny  and
      Panchenko, Alexander",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-srw.27",
    doi = "10.18653/v1/2022.acl-srw.27",
    pages = "355--365",
    abstract = "Knowledge Graphs (KGs) are symbolically structured storages of facts. The KG embedding contains concise data used in NLP tasks requiring implicit information about the real world. Furthermore, the size of KGs that may be useful in actual NLP assignments is enormous, and creating embedding over it has memory cost issues. We represent KG as a 3rd-order binary tensor and move beyond the standard CP decomposition (CITATION) by using a data-specific generalized version of it (CITATION). The generalization of the standard CP-ALS algorithm allows obtaining optimization gradients without a backpropagation mechanism. It reduces the memory needed in training while providing computational benefits. We propose a MEKER, a memory-efficient KG embedding model, which yields SOTA-comparable performance on link prediction tasks and KG-based Question Answering.",
}
```
