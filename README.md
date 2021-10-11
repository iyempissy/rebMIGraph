# MIAGraph

This is the repository for the paper: Membership Inference Attack on Graph Neural Network https://arxiv.org/abs/2101.06570.

**TSTF:** Train on a subgraph and test on the full graph. The testing subgraph includes the full structure of the entire graph including the structure information of the training nodes. 

**TSTS:** Train on a subgraph, test on another subgraph. Different subgraphs are used for training and testing the model. Only the structural infomation of selected graph is used during testing.

The **model_type**, **data_type** and **defense_type** can be changed to view different performance across different models and dataset.

# Detailed Tables:

## TSTF

![Performance in TSTF setting](https://user-images.githubusercontent.com/9529101/136162397-f4684917-74a0-4cf1-913b-32f91d86c0ec.png)

## TSTS

![Performance in TSTS setting](https://user-images.githubusercontent.com/9529101/136162389-3f4785b6-50d4-4837-aac7-d1bf195638af.png)

# Relaxing Asusmptions

## Attack performance without knowledge of exact  hyperparameters
![Relaxing knowledge of hyperparameter assumption](https://user-images.githubusercontent.com/9529101/136159584-d61e0f9b-8388-4f01-8af4-d799e80f470b.png)
**Fig. 7.** Relaxing the knowledge of the hyperparameter assumption. We varied the number of hidden neurons. The original
shadow model was trained with 256 hidden neurons.

## Attack performance without the knowledge of target model's  architecture
![Relaxing knowledge of target model](https://user-images.githubusercontent.com/9529101/136159578-5010ef6f-314b-4a2a-ae87-4b9451a34245.png)
**Fig. 8.** Relaxing the knowledge of target model. O = Original performance when target and shadow model have the same architecture. S = using SGC as shadow model, G = using GCN as shadow model. We observe a similar trend on the precision and recall.

