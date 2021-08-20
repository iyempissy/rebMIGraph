# MIAGraph

We present 2 noteboks for both settings (TSTF and TSTS) described in our paper. This is anonymized for reviewers to run our code and reproduce the results recorded in the paper.

**TSTF:** Train on a subgraph and test on the full graph. The testing subgraph includes the full structure of the entire graph including the structure information of the training nodes. 

**TSTS:** Train on a subgraph, test on another subgraph. Different subgraphs are used for training and testing the model. Only the structural infomation of selected graph is used during testing.

The **model_type**, **data_type** and **defense_type**can be changed to view different performance across different models and dataset.

We also included the **.py** file as the **.ipynb** does not render well in the anonymous repository.
