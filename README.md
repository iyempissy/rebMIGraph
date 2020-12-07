# MIAGraph

We present 2 noteboks for both settings (TSTF and TSTS) described in our paper. This is anonymized for reviewers to run our code and reproduce the results recorded in the paper.

**TSTF:** Train on a subgraph and test on the full graph. The testing subgraph includes the full structure of the entire graph including the structure information of the training nodes. 

**TSTS:** Train on a subgraph, test on another subgraph. Different subgraphs are used for training and testing the model. Only the structural infomation of selected graph is used during testing.

The **model_type** and **data_type** can be changed to view different performance across different models and dataset.

We also included the **.py** file as the **.ipynb** does not render well in the anonymous repository.

### Performance on Cora (TSTF setting)
Model | Shadow Train | Shadow Test | Attack Precision | Attack Recall | Attack AUROC
--- | --- | --- | --- |--- |---
GCN | 0.82 ± 0.01 | 0.85 ± 0.02 | 0.76 ± 0.02 | 0.75 ± 0.02 | 0.754 ± 0.02
GAT | 0.76 ± 0.02 | 0.82 ± 0.01 | 0.69 ± 0.03 | 0.68 ± 0.03 | 0.678 ± 0.03
SGC | 0.72 ± 0.01 | 0.84 ± 0.01 | 0.77 ± 0.02 | 0.78 ± 0.01 | 0.777 ± 0.01
SAGE | 0.99 ± 0.001 | 0.76 ± 0.02 |0.78 ± 0.02  |0.77 ± 0.02  | 0.773 ± 0.02




### Performance on Cora (TSTS setting)
Model | Shadow Train | Shadow Test | Attack Precision | Attack Recall | Attack AUROC
--- | --- | --- | --- |--- |---
GCN | 0.81 ± 0.01 | 0.61 ± 0.01 | 0.70 ± 0.02  | 0.69 ± 0.02  | 0.685 ± 0.02 
GAT | 0.76 ± 0.01 | 0.55 ± 0.02 | 0.69 ± 0.02  | 0.68 ± 0.02  | 0.675 ± 0.02 
SGC | 0.72 ± 0.02 | 0.56 ± 0.02 | 0.68 ± 0.02  | 0.68 ± 0.02  | 0.676 ± 0.02 
SAGE | 0.99 ± 0.002 | 0.70 ± 0.02 |0.83 ± 0.01   |0.82 ± 0.01   |0.819 ± 0.01
