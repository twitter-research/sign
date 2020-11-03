# SIGN: Scalable Inception Graph Networks [[arXiv](https://arxiv.org/abs/2004.11198)]

This repository contains the code to run the SIGN model on the ogbn-papers100M dataset, the largest, publicly available node classification benchmark at the time of writing.
If you want to know more about SIGN checkout its [arXiv page](https://arxiv.org/abs/2004.11198) and its [ICML 2020 GRL+ Workshop version](https://grlplus.github.io/papers/77.pdf).

## Requirements

Dependencies with python 3.8.5:
```
torch==1.5.0
torch_geometric==1.6.1
torch_scatter==2.0.5
torch_sparse==0.6.7
ogb==1.2.3
```

## Preprocessing, Training & Evaluation

```
# Generate SIGN features and save them as sign_333_embeddings.pt
python preprocessing.py --file_name sign_333_embeddings --undirected --directed --directed_asymm_norm --undirected_set_diag --directed_remove_diag

# Train SIGN model based on the preprocessed features generated above and write results at sign_results.txt
python sign_training.py --dropout 0.3 --lr 0.00005 --hidden_channels 512 --num_layers 3 --embeddings_file_name sign_333_embeddings.pt --result_file_name sign_results.txt

# Train SIGN-xl model based on the preprocessed features generated above and write results at sign-xl_results.txt
python sign_training.py --dropout 0.5 --lr 0.00005 --hidden_channels 2048 --num_layers 3 --embeddings_file_name sign_333_embeddings.pt --result_file_name sign-xl_results.txt
```

## Cite us

```bibtex
@inproceedings{sign_icml_grl2020,
    title={SIGN: Scalable Inception Graph Neural Networks},
    author={Fabrizio Frasca and Emanuele Rossi and Davide Eynard and Benjamin Chamberlain and Michael Bronstein and Federico Monti},
    booktitle={ICML 2020 Workshop on Graph Representation Learning and Beyond},
    year={2020}
}
```
