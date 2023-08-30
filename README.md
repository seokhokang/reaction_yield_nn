# reaction_yield_nn
PyTorch implementation of the model described in the paper [Uncertainty-Aware Prediction of Chemical Reaction Yields with Graph Neural Networks](https://doi.org/10.1186/s13321-021-00579-z)

## Components
- **data/*** - dataset files used
- **data/get_data.py** - script for dataset file generation
- **model/*** - model files used
- **run_code.py** - script for model training/evaluation
- **dataset.py** - data structure & functions
- **model.py** - model architecture & functions
- **util.py**

## Data
- The datasets used in the paper can be downloaded from
  - https://github.com/rxn4chemistry/rxn_yields/

## Dependencies
- **Python**
- **PyTorch**
- **DGL**
- **RDKit**

## Citation
```
@Article{Kwon2022,
  title={Uncertainty-aware prediction of chemical reaction yields with graph neural networks},
  author={Kwon, Youngchun and Lee, Dongseon and Choi, Youn-Suk and Kang, Seokho},
  journal={Journal of Cheminformatics},
  volume={14},
  pages={2},
  year={2022},
  doi={10.1186/s13321-021-00579-z}
}
```
