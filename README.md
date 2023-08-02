# ADRNet: Multi-label Adverse Drug Reaction Prediction via Drug Descriptor-aware Collaborative Filtering.
## Usage

- Create and activate a python environment using anaconda:

    `conda env create -f py37env1.yml`
    
    `conda activate py37env1`
    
    ```pip install qpsolvers==1.0.5```



- To generate K-Fold data:

    `python main.py -i`

- To run and evaluate a model:

    `python main.py -d DATA_NAME -m MODEL_NAME -f FEATURE_TYPE` 
    
    For example:
    `python main.py -d AEOLUS -m DrugNCF -f 2
    `

    Evaluation results containing AUC, AUPR and STDERR are stored in "./final_results" folder.


- To obtain options for DATA_NAME and MODEL_NAME and FEATURE_TYPE:

    `python main.py -h`


## Data

All input data is available in the "./data" folder:


## Reference
```
@article{nguyen2021survey,
  title={A survey on adverse drug reaction studies: data, tasks and machine learning methods},
  author={Nguyen, Duc Anh and Nguyen, Canh Hao and Mamitsuka, Hiroshi},
  journal={Briefings in Bioinformatics},
  volume={22},
  number={1},
  pages={164--177},
  year={2021},
  publisher={Oxford University Press}
}
```

