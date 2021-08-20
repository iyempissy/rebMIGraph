#!/usr/bin/env python3
'''
# Train on subgraph and Test on Full graph. Thus TSTF
'''

'''
To run relaxation:
1. Set relax_target_data = False (Normal run)
2. Run for each model_type for Cora and CiteSeer. This saves the corresponding shadow and target files to train n test attacks
3. Set relax_target_data = True
4. Change data_type to CiteSeer
5. End

'''

'''
To run vanpd, set isoutput_perturb_defense = True
To run lbp_defense, set  use_binning = True, use_lbp = True and num_bins
To run nsd_defense, set use_nsd = True and how_many_edges_k
'''
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Reddit, Flickr
from torch_geometric.nn import GCNConv, SAGEConv, SGConv, GATConv
from tqdm import tqdm
import networkx as nx

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch_geometric.data import NeighborSampler

from torch_geometric.utils import subgraph
from torch_geometric.data import Data
import random
import sys
import  itertools
# from itertools import tee
import statistics

from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve, roc_auc_score, f1_score

from torch.nn.utils import clip_grad_norm_
from scipy.spatial.distance import jensenshannon

# No need for this, condor auto assigns
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # change this cos sometimes port 0 is full


# # Reset params of trained model
# for layer in model.children():
#     if hasattr(layer, "reset_parameters"):
#         layer.reset_parameters()


if torch.cuda.is_available():
    global_path = "/dstore/home/iyiola/"
else:
    global_path = "./"

num_of_runs = 2#11 #  # this runs the program 10 times



# Different defense mechanisms
def lbp_defense(pred, num_
                s, beta, use_lbp):
    # #Binned version========>
    pred = pred.cpu().detach().numpy()

    for i in range(0, len(pred)):
        each_pred = pred[i]
        idx = np.array([x for x in range(pred.shape[1])])

        p = np.random.permutation(len(each_pred))

        # shuffled versions
        each_pred = each_pred[p]
        idx = idx[p]

        # divide into bins
        random_split_pred = np.array_split(each_pred, num_bins)
        random_split_idx = np.array_split(idx, num_bins)
        # print("random_split_pred", random_split_pred, "random_split_idx", random_split_idx)
        # print("random_split_idx====>", random_split_idx[0], random_split_idx[1], random_split_idx[2])

        for bin in range(num_bins):
            if use_lbp:
                ''' version on bins. use beta '''
                beta = 1 / beta
                random_split_pred[bin] += np.random.laplace(0, beta,
                                                            1)  # multiply each partition by the corresponding noise value
            else:
                ''' This is the normal version that use the mean of each partition as noise '''
                mean_each_split_pred = np.mean(random_split_pred[bin])  # calculate mean of each partition
                # print("mean", mean_each_split_pred)

                random_split_pred[
                    bin] += mean_each_split_pred  # multiply each partition by the corresponding mean value


            # Loop through and set value of pred back to the new one with their bins
            for w, k in zip(random_split_idx[bin], random_split_pred[bin]):
                # print("This is w", w, "k", k)
                pred[i][w] = k

    # turn back to float tensor
    pred = torch.FloatTensor(pred)
    return pred

def multiply_perturbation_defense(pred, beta):
    # Original========> i.e no binning
    # just multiply each posterior by 0.1 i,e alpha
    pred = pred.cpu().detach().numpy()

    for i in range(0, len(pred)):
        # changed it to \beta * U(-1, 1) whwre alpha is the. Bes
        # sample_noise_uniform
        alpha = round(random.uniform(0.1, 0.2), 10) #random.uniform(0.1, 0.2) #multiply each row by different numbers between 0.1 and 1.0 with floating precision of 10 e.g 0.7812920741
        # print("alpha", alpha)
        # That is, samoling from unifirm distribution of numbers between 0.1 and 1.0 and they have equal probability of appearing
        for j in range(0, len(pred[i])):
            pred[i][j] *= alpha#alpha # only use multiplication if you are perturbing the shadow as well (alpha*beta) 0.1 beta is better

    # turn back to float tensor
    pred = torch.FloatTensor(pred)

    return pred



def vanpd_defense(pred, beta):
    pred = pred.cpu().detach().numpy()
    # add laplacian noise
    beta = 1 / beta # 1 here is the sensitivity
    for i in range(0, len(pred)):
        for j in range(0, len(pred[i])):
            pred[i][j] += np.random.laplace(0, beta, 1)

    # turn back to float tensor
    pred = torch.FloatTensor(pred)

    return pred

def normalize_posterior(pred):
    pred = torch.exp(pred)

    # convert to numpy
    pred = pred.cpu().detach().numpy()
    norm_pred = pred / pred.sum(axis=1,keepdims=True)  # scale or divide each element by the sum of each row
    # convert back to torch
    norm_pred = torch.FloatTensor(norm_pred).to(device)
    return norm_pred



def clip_logits_norm(parameters, max_norm=4, norm_type=2, beta=0.1, add_laplacian_noise=False, add_gaussian_noise=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    # if isinstance(parameters, torch.Tensor):
    #     parameters = [parameters]
    # parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    # if norm_type == inf:
    #     total_norm = max(p.grad.detach().abs().max() for p in parameters)
    # else:
    # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)

    # print("b4 parameters", parameters)

    total_norm = torch.norm(parameters, norm_type)
    # print("total_norm", total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    # print("clip_coef", clip_coef)
    if clip_coef < 1:
        for i in range(0, len(parameters)):
            # print("p", p)
            # p.grad.detach().mul_(clip_coef)
            parameters[i].mul(clip_coef)

    # print("after multiplying parameters", parameters)
    # add Gaussian noise
    if add_gaussian_noise:
        # print("torch.normal(mean=0.0, std=beta * max_norm, size=(parameters.shape[0], parameters.shape[1]))", torch.normal(mean=0.0, std=beta * max_norm, size=(parameters.shape[0], parameters.shape[1])))
        parameters += torch.normal(mean=0.0, std=beta * max_norm, size=(parameters.shape[0], parameters.shape[1])).to(device)

    elif add_laplacian_noise:
        # laplacian noise
        for i in range(0, len(parameters)):
            for j in range(0, len(parameters[i])):
                # k = torch.normal(2, 3, size=(1, 1))
                # print("k", k)
                beta = 1 / beta  # 1 here is the sensitivity
                # print("np.random.laplace(0, beta, 1)", np.random.laplace(0, beta, 1))
                lap_noise = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([beta]))
                parameters[i][j] += lap_noise.sample().item()
                # noisy_logits
    else:
        print("no noise")
    # print("noisy parameters", parameters)
    return parameters




def delete_tensor(tensor, indices):
    # deletes all tensors except those that were selected i.e only keep the indices
    mask = torch.zeros(tensor[0].numel(), dtype=torch.bool)
    mask[indices] = True
    # Remove from both direction of edge index
    tensor_0 = tensor[0][mask]
    tensor_1 = tensor[1][mask]

    stack_tensor = torch.stack([tensor_0, tensor_1])
    return stack_tensor



def k_edge_index(edge_index, k):
    print("B4 k edge index", len(edge_index[0]))
    # node:neighbors_index
    node_list_index_list = []
    node_list_index_dict = {}

    for i in edge_index[0]:
        # print(i)
        node_list_index = (edge_index[0] == i).nonzero(as_tuple=True)[0].tolist() # returns the index of all the occurences of the node. We will use that to "delete" the node later
        node_list_index_dict[i.item()] = node_list_index

    print("len(node_list_index_dict)", len(node_list_index_dict))

    for node in node_list_index_dict:
        # loop through and only select top k
        if len(node_list_index_dict[node]) > k:
            # randomly select k
            selected = random.sample(node_list_index_dict[node], k)
            node_list_index_list.append(selected)

    # join list of list
    all_selected_node_index = list(itertools.chain.from_iterable(node_list_index_list))
    # print("all_selected_node_index", all_selected_node_index)
    print("After k edge index", len(all_selected_node_index))

    # delete all except those that were selected
    # delete_tensor()
    k_edge_index = delete_tensor(edge_index, all_selected_node_index)

    return k_edge_index


# confidence score distortion
def compute_confidence_distortion(noisy_posterior, non_noisy_posteriors):
    targetmember_noisy = pd.read_csv(noisy_posterior, header=None, sep=" ")
    print("targetmember_noisy", targetmember_noisy.shape)
    # convert to tensor
    targetmember_noisy = torch.tensor(targetmember_noisy.values)

    targetmember_nonoise = pd.read_csv(non_noisy_posteriors, header=None, sep=" ")
    print("targetmember_nonoise", targetmember_nonoise.shape)
    targetmember_nonoise = torch.tensor(targetmember_nonoise.values)

    target_member_nonoise_noise_overall_sum = []
    for i in range(len(targetmember_nonoise)):
        target_member_nonoise_noisy_js = jensenshannon(targetmember_nonoise[i], targetmember_noisy[i],
                                                       base=2)  # rel_entr(targetmember_nonoise[i], targetmember_noisy[i])
        sum_each_dim = target_member_nonoise_noisy_js  # sum(target_member_nonoise_noisy_js)
        print(sum_each_dim)
        target_member_nonoise_noise_overall_sum.append(sum_each_dim.item())

    final_member_nonoise_noisy_js = sum(target_member_nonoise_noise_overall_sum) / len(targetmember_nonoise)

    print("final_member_nonoise_noisy_js", final_member_nonoise_noisy_js)
    return final_member_nonoise_noisy_js

for global_model_type in ["Cora", "CiteSeer", "PubMed", "Flickr"]: #["Cora"]:#, "CiteSeer", "PubMed", "Flickr"]:
    # for global_k in [0, 1, 2, 3]:
    # for global_bin in [2, 3, 4]:
    #     for global_eps in [0.2, 0.5, 1.2, 2.0, 3, 10]:
            # random_data = [1050154401, 87952126, 461858464, 2251922041, 2203565404, 2569991973, 569824674, 2721098863, 836273002,2935227127]
            random_data = [1050154401]

            if torch.cuda.is_available():
                home_root = "/dstore/home/iyiola/"
            else:
                home_root = "./"

            all_results = []

            result_file_average = open(home_root + "Average_resultfile_TSTF_" +".txt", "a")

            target_train_loss_acc_per_run = []
            target_approx_train_acc_per_run = []
            target_train_acc_per_run = []
            target_test_acc_per_run = []
            target_macro_acc_per_run = []
            target_micro_acc_per_run = []


            shadow_train_loss_acc_per_run = []
            shadow_approx_train_acc_per_run = []
            shadow_train_acc_per_run = []
            shadow_test_acc_per_run = []
            shadow_macro_acc_per_run = []
            shadow_micro_acc_per_run = []

            precision_per_run = []
            auroc_per_run = []
            recall_per_run = []
            f1_score_per_run = []

            total_time_per_run = []



            # we run each data_type e.g cora against all model type

            '''
            Set parameters here ===================================================================================?????????
            '''

            model_type = "GCN"  # GCN, GAT, SAGE, SGC
            shadow_model_type = "GCN"
            target_num_neurons = 256
            shadow_num_neurons = 256  # 256 64

            # relaxation
            relax_target_data = False

            # defense
            isdp_logits = False  # Adds noise to the logits (logits = layer b4 the softmax)
            add_laplacian_to_logits = False
            add_gaussian_to_logits = False
            max_norm_logits = 2

            isoutput_perturb_defense = False  # This adds noise only to the posterior of the target model. The shadow is left untouched!. Note, noise should be addded at the test (when u query the target model!)
            beta = 0.2#global_eps #0.2

            perturb_shadow = False  # perturb shadow model with the same perturbation as the target
            ismultiply_by_beta = False  # True  # multiply posterior by beta. 0.2. Works better than 0.5. If set to false, then its normal i.e no multiplication

            # ismultiply_by_beta needs to be set to True to use  use_binning and use_lbp

            # if use_binning is set to False, then there is no binning and just do multiplicative with the value of beta
            use_binning = False  # meanbdp Set to True to use the binned version. If set to true, it uses the mean of each bin and adds the corresponding value. No need for beta
            use_lbp = False  # lbdp Also binned but add noise drawn from Laplacian distribution to each bin. It use beta here. Also use_binning needs be True
            num_bins = 2#global_bin #2

            use_prior_inductive_split = False  # True #False # set to true if you wanna use the prior method of splitting the graph. Setting to false is a lot faster and better method!

            remove_edge_at_test_index = False#True  # When an attacker send a query, do not use the entire neighbor
            use_nsd = False # nsd to use this remove_edge_at_test_index needs to be set to False
            how_many_edges_k = 2#global_k #3 # e.g selects 3 random neighbors {mask on the adjacency matrix}

            data_type = global_model_type #"Cora"  # CiteSeer, Cora, PubMed, Flickr, Reddit
            # # CiteSeer (binary features), Cora (binary features), PubMed(float where necessary), Flickr(integers), Reddit(float rep)

            # catering for only using Cora as target and CiteSeer as shadow and vice versa.
            # The data_type becomes the shadow and the opposite becomes the target
            # e.g if Cora is the data_type, then CiteSeer is the target n Cora is the shadow
            if relax_target_data:
                if data_type == "Cora":
                    relax_target_data_type = "CiteSeer"
                else:
                    relax_target_data_type = "Cora"
            else:
                relax_target_data_type = data_type  # the same as data type

            print("Target data type for relaxation", relax_target_data_type)
            print("Shadow data type", data_type)

            mode = "TSTF"  # train on subgraph, test on full grah


            if isdp_logits == True and add_laplacian_to_logits == True:
                def_t = "vanlapdp_logits"
                defense_type = def_t + str(max_norm_logits)+str(beta)
            elif isdp_logits == True and add_gaussian_to_logits == True:
                def_t = "vangaudp_logits"
                defense_type = def_t + str(max_norm_logits) + str(beta)
            elif isoutput_perturb_defense == True:
                def_t = "vanpd_posterior"
                defense_type = def_t + str(beta) #original vanilla

            elif ismultiply_by_beta and use_binning and use_lbp:
                def_t = "lbdp"
                defense_type = def_t + str(beta) + str(num_bins)

            elif ismultiply_by_beta and use_binning:
                def_t = "meanbdp"
                defense_type = def_t + str(num_bins)
            elif use_nsd == True:
                def_t = "neigbhordist"
                defense_type = def_t +str(how_many_edges_k)
            else:
                def_t = "normal"
                defense_type = def_t # no defense

            print("defense type", defense_type)


            # writing all print to file
            old_stdout = sys.stdout
            log_file = open("message"+data_type+model_type+defense_type+".txt","w")

            sys.stdout = log_file

            #TODO Run after


            save_shadow_OutTrain = "posteriorsShadowOut_" + mode + "_" + data_type + "_" + model_type + defense_type + ".txt"
            save_shadow_InTrain = "posteriorsShadowTrain_" + mode + "_" + data_type + "_" + model_type + defense_type + ".txt"
            save_target_OutTrain = "posteriorsTargetOut_" + mode + "_" + data_type + "_" + model_type + defense_type + ".txt"
            save_target_InTrain = "posteriorsTargetTrain_" + mode + "_" + data_type + "_" + model_type + defense_type + ".txt"

            save_correct_incorrect_homophily_prediction = "correct_incorrect_homo_pred" + mode + "_" + data_type + "_" + model_type + defense_type + ".txt"
            save_global_true_homophily = "true_homophily" + mode + "_" + data_type + "_" + model_type + defense_type + ".txt"
            save_global_pred_homophily = "pred_homophily" + mode + "_" + data_type + "_" + model_type + defense_type + ".txt"

            save_target_InTrain_nodes_neigbors = "nodesNeigborsTargetTrain_" + mode + "_" + data_type + "_" + model_type + defense_type + ".npy"  # TODO N
            save_target_OutTrain_nodes_neigbors = "nodesNeigborsTargetOut_" + mode + "_" + data_type + "_" + model_type + defense_type + ".npy"  # TODO N







            for which_run, rand_state in enumerate(random_data): #range(1, num_of_runs)
                # result_file = open(global_path + "resultfile_" + mode + "_" + model_type + "_" + shadow_model_type + defense_type +".txt", "a")
                result_file = open(global_path + "resultfile_" + mode + "_" + model_type + "_" + shadow_model_type + def_t +".txt", "a")
                # random_data = os.urandom(4)

                # rand_state = 1050154401 #int.from_bytes(random_data, byteorder="big") # 4123696913 #10, #3469326556, 959554842 1048906271 2784139507 2676276030(Better), 75126506#

                # rand_state = [1050154401, 87952126, 461858464, 2251922041, 2203565404, 2569991973, 569824674, 2721098863, 836273002,2935227127]
                # rand_state = [1050154401]
                print("rand_state", rand_state)
                torch.manual_seed(rand_state)
                torch.cuda.manual_seed(rand_state)
                random.seed(rand_state)
                np.random.seed(seed=rand_state)
                torch.cuda.manual_seed_all(rand_state)  # if you are using multi-GPU.
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                # torch.set_default_dtype(torch.float64) #set default torch to 64 floating

                start_time = time.time()


                '''
                ######################################## Data ##############################################
                '''
                if data_type == "Reddit":
                    ###################################### Reddit ##################################

                    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
                    path = global_path+"data/"+data_type #"/dstore/home/iyiola/data/Reddit"
                    dataset = Reddit(path)
                    # data = dataset[0]
                    # print("len(dataset)", len(dataset)) # Reddit dataset consists of 1 graph
                    # print("data", data) # Data(edge_index=[2, 114615892], test_mask=[232965], train_mask=[232965], val_mask=[232965], x=[232965, 602], y=[232965])
                    # print("Total Num of nodes in dataset", data.num_nodes) # 232965
                    # print("Total Num of edges in dataset", data.num_edges) # 114615892
                    # print("Total Num of node features in dataset", data.num_node_features) # 602
                    # print("Total Num of features in dataset", dataset.num_features) # same as node features # 602
                    # print("Num classes", dataset.num_classes) #41

                    # Reduced this cos it's taking too long to create subgraph
                    num_train_Train_per_class = 500 #500  # 1000 50 for redditSmall
                    num_train_Shadow_per_class = 500 #500  # 1000
                    num_test_Target = 20500 #20500  # 41000
                    num_test_Shadow = 20500 #20500  # 41000

                    # normal train test of target model. For comparing with distorted / the defensed method
                    # note: all normal_precision, normal_recall and normal_auroc is wrt to not using prior inductive split for fair comparison

                    if model_type == "GCN":
                        normal_test = 0
                        normal_precision = 0
                        normal_recall = 0
                        normal_auroc = 0
                    elif model_type == "GAT":
                        normal_test = 0
                        normal_precision = 0
                        normal_recall = 0
                        normal_auroc = 0
                    elif model_type == "SGC":
                        normal_test = 0
                        normal_precision = 0
                        normal_recall = 0
                        normal_auroc = 0

                    else:
                        normal_test = 0 #SAGE
                        normal_precision = 0
                        normal_recall = 0
                        normal_auroc = 0

                elif data_type == "Flickr":

                    ###################################### Flickr ##################################

                    path = global_path+"data/"+data_type #"/dstore/home/iyiola/Flickr"
                    dataset = Flickr(path)
                    data = dataset[0]
                    # print("len(dataset)", len(dataset))  # Flikr dataset consists of 1 graph
                    # print("data",
                    #       data)  # Data(edge_index=[2, 899756], test_mask=[89250], train_mask=[89250], val_mask=[89250], x=[89250, 500], y=[89250])
                    # print("Total Num of nodes in dataset", data.num_nodes)  # 89250
                    # print("Total Num of edges in dataset", data.num_edges)  # 899756
                    # print("Total Num of node features in dataset", data.num_node_features)  # 500
                    # print("Total Num of features in dataset", dataset.num_features)  # same as node features # 500
                    # print("Num classes", dataset.num_classes)  # 7

                    num_train_Train_per_class = 1500  # cos min of all classes only have 3k nodes
                    num_train_Shadow_per_class = 1500
                    num_test_Target = 10500
                    num_test_Shadow = 10500

                    # normal train test of target model. For comparing with distorted / the defensed method
                    if model_type == "GCN":
                        normal_test = 0.18
                        normal_precision = 0.871
                        normal_recall = 0.881
                        normal_auroc = 0.871
                    elif model_type == "GAT":
                        normal_test = 0.14
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.
                    elif model_type == "SGC":
                        normal_test = 0.10
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.
                    else:
                        normal_test = 0.20 #SAGE
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.


                elif data_type == "Cora":

                    ###################################### Cora ##################################
                    path = global_path+"data/"+data_type #'/dstore/home/iyiola/Cora'
                    dataset = Planetoid(root=path, name="Cora", split="random")  # set test to 1320 to match train
                    num_train_Train_per_class = 90  # 180
                    num_train_Shadow_per_class = 90
                    num_test_Target = 630
                    num_test_Shadow = 630


                    # normal train test of target model. For comparing with distorted / the defensed method
                    if model_type == "GCN":
                        normal_test = 0.84
                        normal_precision = 0.815
                        normal_recall = 0.812
                        normal_auroc = 0.811
                    elif model_type == "GAT":
                        normal_test = 0.83
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.
                    elif model_type == "SGC":
                        normal_test = 0.84
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.
                    else:
                        normal_test = 0.76 #SAGE
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.

                elif data_type == "CiteSeer":

                    ###################################### CiteSeer ##################################
                    path = global_path+"data/"+data_type #'/dstore/home/iyiola/CiteSeer'
                    dataset = Planetoid(root=path, name="CiteSeer", split="random")
                    num_train_Train_per_class = 100
                    num_train_Shadow_per_class = 100
                    num_test_Target = 600
                    num_test_Shadow = 600


                    # normal train test of target model. For comparing with distorted / the defensed method
                    if model_type == "GCN":
                        normal_test = 0.75
                        normal_precision = 0.887
                        normal_recall = 0.879
                        normal_auroc = 0.879
                    elif model_type == "GAT":
                        normal_test = 0.73
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.
                    elif model_type == "SGC":
                        normal_test = 0.74
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.
                    else:
                        normal_test = 0.67 #SAGE
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.


                elif data_type == "PubMed":

                    ###################################### PubMed ##################################
                    path = global_path + "data/" + data_type #'/dstore/home/iyiola/PubMed'
                    dataset = Planetoid(root=path, name="PubMed", split="random")

                    num_train_Train_per_class = 1500
                    num_train_Shadow_per_class = 1500
                    num_test_Target = 4500
                    num_test_Shadow = 4500


                    # normal train test of target model. For comparing with distorted / the defensed method
                    if model_type == "GCN":
                        normal_test = 0.80
                        normal_precision = 0.689
                        normal_recall = 0.678
                        normal_auroc = 0.678
                    elif model_type == "GAT":
                        normal_test = 0.76
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.
                    elif model_type == "SGC":
                        normal_test = 0.78
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.
                    else:
                        normal_test = 0.78 #SAGE
                        normal_precision = 0.
                        normal_recall = 0.
                        normal_auroc = 0.


                else:
                    print("Error: No data specified")

                data = dataset[0]
                print("data", data)


                '''
                ############################################# Target and Shadow Models ###############################################
                '''


                class TargetModel(torch.nn.Module):
                    def __init__(self, dataset):
                        super(TargetModel, self).__init__()

                        if model_type == "GCN":
                            # GCN
                            self.conv1 = GCNConv(dataset.num_node_features, target_num_neurons)
                            self.conv2 = GCNConv(target_num_neurons, dataset.num_classes)
                        elif model_type == "SAGE":
                            # GraphSage
                            # self.conv1 = SAGEConv(dataset.num_node_features, 256)
                            # self.conv2 = SAGEConv(256, dataset.num_classes)

                            # TODO SAGE
                            self.num_layers = 2

                            self.convs = torch.nn.ModuleList()
                            self.convs.append(SAGEConv(dataset.num_node_features, target_num_neurons))
                            self.convs.append(SAGEConv(target_num_neurons, dataset.num_classes))

                        elif model_type == "SGC":
                            # SGC
                            self.conv1 = SGConv(dataset.num_node_features, target_num_neurons, K=2, cached=False)
                            self.conv2 = SGConv(target_num_neurons, dataset.num_classes, K=2, cached=False)

                        elif model_type == "GAT":
                            # GAT
                            self.conv1 = GATConv(dataset.num_features, target_num_neurons, heads=8, dropout=0.1)
                            # On the Pubmed dataset, use heads=8 in conv2.
                            if data_type == "PubMed":
                                self.conv2 = GATConv(target_num_neurons * 8, dataset.num_classes, heads=8, concat=False)
                                # self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=8, concat=False, dropout=0.1)
                            else:
                                self.conv2 = GATConv(target_num_neurons * 8, dataset.num_classes, heads=1, concat=False)
                        else:
                            print("Error: No model selected")

                    def forward(self, x, edge_index):
                        # print("xxxxxxx", x.size())

                        # TODO SAGE
                        if model_type == "SAGE":
                            all_node_and_neigbors = []  # {} # lookup table dictionary #changed to list cos it;s not possible to slice esp for TSTS TODO C
                            all_nodes = []

                            # the edge_index here is quite different (it is a list cos we will be passing train_loader). edge index is different based on batch data.
                            # Note edge_index here is a bipartite graph. meaning all the edges retured are connected
                            for i, (edge_ind, _, size) in enumerate(edge_index):
                                # print("iiiiiiiiiiiiiiii", i)

                                edges_raw = edge_ind.cpu().numpy()
                                # print("edges_raw", edges_raw)
                                edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

                                G = nx.Graph()
                                G.add_nodes_from(list(range(
                                    data.num_nodes)))
                                G.add_edges_from(edges)


                                # getting the neigbors of a particular node.
                                for n in range(0, x.size(
                                        0)):
                                    all_nodes.append(n)  # get all nodes
                                    # all_node_and_neigbors[n] = [node for node in G.neighbors(n)]  # set the value of the dict to the corresponding value if it has a neighbor else put empty list
                                    all_node_and_neigbors.append((n, [node for node in G.neighbors(n)]))  # TODO C
                                    # print("n:", n, "neighbors:", [n for n in G.neighbors(n)])  # G.adj[n] # gets the neighbors of a particular n


                                x_target = x[:size[1]]  # Target nodes are always placed first.
                                x = self.convs[i]((x, x_target), edge_ind)
                                if i != self.num_layers - 1:
                                    x = F.relu(x)
                                    # x = F.dropout(x, p=0.5, training=self.training)
                            # print("Final all nodes and neighbors", all_node_and_neigbors)
                            return x.log_softmax(dim=-1), all_node_and_neigbors

                        else:
                            edges_raw = edge_index.cpu().numpy()
                            # print("edges_raw", edges_raw)
                            edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

                            G = nx.Graph()
                            G.add_nodes_from(list(range(
                                data.num_nodes)))
                            G.add_edges_from(edges)

                            all_node_and_neigbors = []  # {} # lookup table list
                            all_nodes = []


                            for n in range(0, x.size(
                                    0)):
                                all_nodes.append(n)  # get all nodes
                                all_node_and_neigbors.append((n, [node for node in G.neighbors(n)]))  #



                            x, edge_index = x, edge_index
                            x = self.conv1(x, edge_index)
                            x = F.relu(x)
                            # x = F.dropout(x, p=0.5, training=self.training)
                            # x = F.normalize(x, p=2, dim=-1)
                            x = self.conv2(x, edge_index)

                            # clip logits here and add noise
                            if isdp_logits:
                                x = clip_logits_norm(x, max_norm=max_norm_logits, norm_type=2, beta=beta, add_laplacian_noise=add_laplacian_to_logits,
                                                  add_gaussian_noise=add_gaussian_to_logits)
                            # x = F.relu(x)
                            # x = F.normalize(x, p=2, dim=-1)

                            return F.log_softmax(x, dim=1), all_node_and_neigbors



                    def inference(self, x_all):

                        for i in range(self.num_layers):
                            xs = []
                            all_node_and_neigbors = []
                            all_nodes = []

                            for batch_size, n_id, adj in all_graph_loader:
                                edge_index, _, size = adj.to(device)

                                x = x_all[n_id].to(device)

                                edges_raw = edge_index.cpu().numpy()
                                # print("edges_raw inference", edges_raw)
                                edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

                                G = nx.Graph()
                                G.add_nodes_from(list(range(
                                    data.num_nodes)))
                                G.add_edges_from(edges)


                                for n in range(0, x.size(
                                        0)):
                                    all_nodes.append(n)  # get all nodes

                                    all_node_and_neigbors.append((n, [node for node in G.neighbors(n)]))
                                x_target = x[:size[1]]
                                x = self.convs[i]((x, x_target), edge_index)
                                if i != self.num_layers - 1:
                                    x = F.relu(x)
                                xs.append(x.cpu())

                            x_all = torch.cat(xs, dim=0)

                        return F.log_softmax(x_all, dim=1), all_node_and_neigbors


                class ShadowModel(torch.nn.Module):
                    def __init__(self, dataset):
                        super(ShadowModel, self).__init__()

                        if shadow_model_type == "GCN":
                            # GCN
                            self.conv1 = GCNConv(dataset.num_node_features, shadow_num_neurons)
                            self.conv2 = GCNConv(shadow_num_neurons, dataset.num_classes)
                        elif shadow_model_type == "SAGE":
                            # GraphSage
                            # self.conv1 = SAGEConv(dataset.num_node_features, 256)
                            # self.conv2 = SAGEConv(256, dataset.num_classes)

                            # TODO SAGE
                            self.num_layers = 2

                            self.convs = torch.nn.ModuleList()
                            self.convs.append(SAGEConv(dataset.num_node_features, shadow_num_neurons))
                            self.convs.append(SAGEConv(shadow_num_neurons, dataset.num_classes))

                        elif shadow_model_type == "SGC":
                            # SGC
                            self.conv1 = SGConv(dataset.num_node_features, shadow_num_neurons, K=2, cached=False)
                            self.conv2 = SGConv(shadow_num_neurons, dataset.num_classes, K=2, cached=False)

                        elif shadow_model_type == "GAT":
                            # GAT
                            self.conv1 = GATConv(dataset.num_features, shadow_num_neurons, heads=8, dropout=0.1)
                            # On the Pubmed dataset, use heads=8 in conv2.
                            if data_type == "PubMed":
                                self.conv2 = GATConv(shadow_num_neurons * 8, dataset.num_classes, heads=8, concat=False)
                                # self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=8, concat=False, dropout=0.1)
                            else:
                                self.conv2 = GATConv(shadow_num_neurons * 8, dataset.num_classes, heads=1, concat=False)
                        else:
                            print("Error: No model selected")

                    def forward(self, x, edge_index):
                        # print("xxxxxxx", x.size())

                        if shadow_model_type == "SAGE":
                            all_node_and_neigbors = []
                            all_nodes = []

                            for i, (edge_ind, _, size) in enumerate(edge_index):
                                # print("iiiiiiiiiiiiiiii", i)

                                edges_raw = edge_ind.cpu().numpy()
                                # print("edges_raw", edges_raw)
                                edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

                                G = nx.Graph()
                                G.add_nodes_from(list(range(
                                    data.num_nodes)))
                                G.add_edges_from(edges)

                                for n in range(0, x.size(
                                        0)):
                                    all_nodes.append(n)  # get all nodes
                                    all_node_and_neigbors.append((n, [node for node in G.neighbors(n)]))  # TODO C
                                x_shadow = x[:size[1]]  # Shadow nodes are always placed first.
                                x = self.convs[i]((x, x_shadow), edge_ind)
                                if i != self.num_layers - 1:
                                    x = F.relu(x)
                                    # x = F.dropout(x, p=0.5, training=self.training)
                            # print("Final all nodes and neighbors", all_node_and_neigbors)
                            return x.log_softmax(dim=-1), all_node_and_neigbors

                        else:
                            edges_raw = edge_index.cpu().numpy()
                            # print("edges_raw", edges_raw)
                            edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

                            G = nx.Graph()
                            G.add_nodes_from(list(range(
                                data.num_nodes)))
                            G.add_edges_from(edges)

                            all_node_and_neigbors = []
                            all_nodes = []


                            for n in range(0, x.size(
                                    0)):
                                all_nodes.append(n)
                                all_node_and_neigbors.append((n, [node for node in G.neighbors(n)]))


                            x, edge_index = x, edge_index
                            x = self.conv1(x, edge_index)
                            x = F.relu(x)
                            # x = F.dropout(x, p=0.5, training=self.training)
                            # x = F.normalize(x, p=2, dim=-1)
                            x = self.conv2(x, edge_index)
                            # x = F.relu(x)
                            # x = F.normalize(x, p=2, dim=-1)

                            return F.log_softmax(x, dim=1), all_node_and_neigbors



                    def inference(self, x_all):

                        for i in range(self.num_layers):
                            xs = []
                            all_node_and_neigbors = []
                            all_nodes = []

                            for batch_size, n_id, adj in all_graph_loader:
                                edge_index, _, size = adj.to(device)

                                x = x_all[n_id].to(device)

                                edges_raw = edge_index.cpu().numpy()
                                # print("edges_raw inference", edges_raw)
                                edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

                                G = nx.Graph()
                                G.add_nodes_from(list(range(
                                    data.num_nodes)))
                                G.add_edges_from(edges)

                                for n in range(0, x.size(
                                        0)):
                                    all_nodes.append(n)
                                    all_node_and_neigbors.append((n, [node for node in G.neighbors(n)]))
                                x_shadow = x[:size[1]]
                                x = self.convs[i]((x, x_shadow), edge_index)
                                if i != self.num_layers - 1:
                                    x = F.relu(x)
                                xs.append(x.cpu())

                            x_all = torch.cat(xs, dim=0)

                        return F.log_softmax(x_all, dim=1), all_node_and_neigbors


                '''
                ##################################### Data Processing Inductive Split ######################################
                '''


                def get_inductive_spilt(data, num_classes, num_train_Train_per_class, num_train_Shadow_per_class, num_test_Target,
                                        num_test_Shadow):
                    # -----------------------------------------------------------------------
                    # target_train, target_out
                    # shadow_train, shadow_out
                    '''
                    Randomly choose 'num_train_Train_per_class' and 'num_train_Shadow_per_class' per classes for training Target and shadow models respectively
                    Random choose 'num_test_Target' and 'num_test_Shadow' for testing (out data) Target and shadow models respectively

                    '''

                    st_time = time.time()

                    # convert all label to list
                    label_idx = data.y.numpy().tolist()
                    print("label_idx", len(label_idx))
                    if os.path.isfile(global_path + "/target_train_idx" + mode + "_" + data_type + "_" + str(rand_state)+ ".pt"):
                        target_train_idx = torch.load(global_path + "/target_train_idx" + mode + "_" + data_type + "_" + str(rand_state)+ ".pt")
                        shadow_train_idx = torch.load(global_path + "/shadow_train_idx" + mode + "_" + data_type + "_" + str(rand_state)+ ".pt")
                        target_test_idx = torch.load(global_path + "/target_test_idx" + mode + "_" + data_type + "_" + str(rand_state)+ ".pt")
                        shadow_test_idx = torch.load(global_path + "/shadow_test_idx" + mode + "_" + data_type + "_" + str(rand_state)+ ".pt")


                    else:

                        target_train_idx = []
                        shadow_train_idx = []
                        # for i in range(num_classes):
                        #     c = [x for x in range(len(label_idx)) if label_idx[x] == i]
                        #     print("c", len(c)) #the min is 180 which is 7th class
                        #     sample = random.sample(range(c),num_train_Train_per_class)
                        #     target_train_idx.extend(sample)

                        for c in range(num_classes):
                            idx = (data.y == c).nonzero().view(-1)
                            sample_train_idx = idx[torch.randperm(idx.size(0))]
                            sample_target_train_idx = sample_train_idx[:num_train_Train_per_class]
                            target_train_idx.extend(sample_target_train_idx)

                            print("idx.size(0)", idx.size(0))  # this is the total number of data in each class
                            sample_shadow_train_idx = sample_train_idx[
                                                      num_train_Train_per_class:num_train_Train_per_class + num_train_Shadow_per_class]
                            shadow_train_idx.extend(sample_shadow_train_idx)

                        torch.save(target_train_idx, global_path + "/target_train_idx" + mode + "_" + data_type + "_" + str(rand_state)+ ".pt")
                        torch.save(shadow_train_idx, global_path + "/shadow_train_idx" + mode + "_" + data_type + "_" + str(rand_state)+ ".pt")

                        print("shadow_train_idx", len(shadow_train_idx))
                        print("Target_train_idx", len(target_train_idx))

                        others = [x for x in range(len(label_idx)) if x not in set(target_train_idx) and x not in set(shadow_train_idx)]
                        print("done others")
                        target_test_idx = random.sample(others, num_test_Target)
                        print("done target test")
                        shadow_test = [x for x in others if x not in set(target_test_idx)]
                        shadow_test_idx = random.sample(shadow_test, num_test_Shadow)
                        print("done shadow test")

                        torch.save(target_test_idx,
                            global_path + "/target_test_idx" + mode + "_" + data_type + "_" + str(rand_state)+ ".pt")
                        torch.save(shadow_test_idx,
                            global_path + "/shadow_test_idx" + mode + "_" + data_type + "_" + str(rand_state)+ ".pt")

                        print("target_test_idx", target_test_idx)
                        print("shadow_test_idx", shadow_test_idx)





                    # ----------set values for mask--------------------------------


                    if model_type == "SAGE":
                        target_train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
                        for i in target_train_idx:
                            target_train_mask[i] = 1

                        shadow_train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
                        for i in shadow_train_idx:
                            shadow_train_mask[i] = 1

                    else:
                        # Other GNN
                        target_train_mask = torch.ones(len(target_train_idx), dtype=torch.uint8)
                        shadow_train_mask = torch.ones(len(shadow_train_idx), dtype=torch.uint8)

                    # ---test-mask---
                    target_test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
                    for i in target_test_idx:
                        target_test_mask[i] = 1
                    # ---val-mask-----
                    shadow_test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
                    for i in shadow_test_idx:
                        shadow_test_mask[i] = 1

                    '''
                    get all nodes and corresponding edge_index information
                    '''
                    # This is for creating subgraphs

                    # For target
                    target_x_inductive = data.x[target_train_idx]
                    target_y_inductive = data.y[target_train_idx]
                    # target_edge_index_inductive, _ = subgraph(target_train_idx, data.edge_index)

                    # For shadow
                    shadow_x_inductive = data.x[shadow_train_idx]
                    shadow_y_inductive = data.y[shadow_train_idx]
                    # shadow_edge_index_inductive, _ = subgraph(shadow_train_idx, data.edge_index)

                    if use_prior_inductive_split:
                        # upis = use_prior_inductive_split
                        if os.path.isfile(global_path + "/target_edge_index_inductive" + mode + "_" + data_type + "_upis" + str(rand_state) + ".pt"):
                            target_edge_index_inductive = torch.load(global_path + "/target_edge_index_inductive" + mode + "_" + data_type + "_upis" + str(rand_state) + ".pt")
                            shadow_edge_index_inductive = torch.load(global_path + "/shadow_edge_index_inductive" + mode + "_" + data_type + "_upis" + str(rand_state) + ".pt")

                        else:
                            # run again
                            target_edge_index_inductive, _ = subgraph(target_train_idx, data.edge_index)
                            shadow_edge_index_inductive, _ = subgraph(shadow_train_idx, data.edge_index)

                            '''
                            in this part use a vertex_map to get a correct target_edge_index_inductive
                            '''
                            # ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*
                            '''
                            get new edge_index information, because some nodes were removed from orginal nodes set. so there
                            are some edge_index that will disappear. If we dont do that, it will cause error: out of index 193
                            '''

                            target_vertex_map = {}
                            ind = -1
                            for i in range(data.num_nodes):
                                if i in target_train_idx:
                                    ind += 1
                                    target_vertex_map[i] = ind
                            for i in range(target_edge_index_inductive.shape[1]):
                                target_edge_index_inductive[0, i] = target_vertex_map[target_edge_index_inductive[0, i].tolist()]
                                target_edge_index_inductive[1, i] = target_vertex_map[target_edge_index_inductive[1, i].tolist()]

                            shadow_vertex_map = {}
                            ind = -1
                            for i in range(data.num_nodes):
                                if i in shadow_train_idx:
                                    ind += 1
                                    shadow_vertex_map[i] = ind
                            for i in range(shadow_edge_index_inductive.shape[1]):
                                shadow_edge_index_inductive[0, i] = shadow_vertex_map[shadow_edge_index_inductive[0, i].tolist()]
                                shadow_edge_index_inductive[1, i] = shadow_vertex_map[shadow_edge_index_inductive[1, i].tolist()]

                            torch.save(target_edge_index_inductive,
                                global_path + "/target_edge_index_inductive" + mode + "_" + data_type + "_upis" + str(
                                    rand_state) + ".pt")
                            torch.save(shadow_edge_index_inductive,
                                global_path + "/shadow_edge_index_inductive" + mode + "_" + data_type + "_upis" + str(
                                    rand_state) + ".pt")


                    else:
                        # This is fast so need to save
                        target_edge_index_inductive, _ = subgraph(target_train_idx, data.edge_index, relabel_nodes=True, num_nodes=len(label_idx))
                        shadow_edge_index_inductive, _ = subgraph(shadow_train_idx, data.edge_index, relabel_nodes=True, num_nodes=len(label_idx))

                        target_degree = target_edge_index_inductive.shape[1] / len(target_train_idx)
                        shadow_degree = shadow_edge_index_inductive.shape[1] / len(shadow_train_idx)

                        # print("target_degree", target_degree)
                        # print("shadow_degree", shadow_degree)




                    # ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*

                    # All graph data
                    all_x = data.x
                    all_y = data.y
                    all_edge_index = data.edge_index

                    '''
                    now we create a New data instances for save the all data with that we do inductive learning tasks
                    '''

                    data = Data(target_x=target_x_inductive, target_edge_index=target_edge_index_inductive,
                                target_y=target_y_inductive,
                                shadow_x=shadow_x_inductive, shadow_edge_index=shadow_edge_index_inductive,
                                shadow_y=shadow_y_inductive,
                                target_train_mask=target_train_mask, shadow_train_mask=shadow_train_mask, all_x=all_x,
                                all_edge_index=all_edge_index, all_y=all_y, target_test_mask=target_test_mask,
                                shadow_test_mask=shadow_test_mask)

                    en_time = time.time()
                    fi_time = en_time - st_time
                    print("Time for creating graph", fi_time)
                    return data


                def get_train_acc(data_new, pred, isTarget=True):
                    if isTarget:
                        train_acc = pred.eq(data_new.target_y).sum().item() / data_new.target_train_mask.sum().item()
                    else:
                        train_acc = pred.eq(data_new.shadow_y).sum().item() / data_new.shadow_train_mask.sum().item()
                    return train_acc


                def get_test_acc(data_new, pred, isTarget=True):
                    if isTarget:
                        test_acc = pred.eq(
                            data_new.all_y[data_new.target_test_mask]).sum().item() / data_new.target_test_mask.sum().item()
                    else:
                        test_acc = pred.eq(
                            data_new.all_y[data_new.shadow_test_mask]).sum().item() / data_new.shadow_test_mask.sum().item()
                    return test_acc


                def get_marco_f1(data_new, pred_labels, true_labels, label_list):
                    f1_marco = f1_score(true_labels, pred_labels, average='macro')
                    return f1_marco


                def get_micro_f1(data_new, pred_labels, true_labels, label_list):
                    f1_micro = f1_score(true_labels, pred_labels, average='micro')
                    return f1_micro


                '''
                ########################## End Data Processing Inductive Split ###########################
                '''

                ''' Data initalization '''

                # --- create labels_list---
                label_list = [x for x in range(dataset.num_classes)]

                # convert all label to list
                label_idx = data.y.numpy().tolist()

                data_new = get_inductive_spilt(data, dataset.num_classes, num_train_Train_per_class, num_train_Shadow_per_class,
                                               num_test_Target, num_test_Shadow)

                print("data", data)
                print("data new", data_new)
                print("data_new.shadow_test_mask.sum()", data_new.shadow_test_mask.sum())
                print("data_new.target_test_mask.sum()", data_new.target_test_mask.sum())

                print(dataset.num_classes)

                bool_tensor = torch.ones(num_test_Target, dtype=torch.bool)
                # print("bool_tensor", bool_tensor) #using this increases attack precision to 0.604 instead of 0.590

                target_train_loader = NeighborSampler(data_new.target_edge_index, node_idx=bool_tensor,
                                                      sizes=[25, 10], num_nodes=num_test_Target, batch_size=64,
                                                      shuffle=False)
                shadow_train_loader = NeighborSampler(data_new.shadow_edge_index, node_idx=bool_tensor,
                                                      sizes=[25, 10], num_nodes=num_test_Shadow, batch_size=64, shuffle=False)


                all_graph_loader = NeighborSampler(data_new.all_edge_index, node_idx=None, sizes=[-1],
                                                   batch_size=1024, num_nodes=data.num_nodes, shuffle=False)

                ''' Model initialization '''

                target_model = TargetModel(dataset)
                shadow_model = ShadowModel(
                    dataset)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                data_new = data_new.to(device)

                target_model = target_model.to(device)

                shadow_model = shadow_model.to(device)

                print("Target model", target_model)
                print("Shadow model", shadow_model)

                if model_type == "SAGE":
                    # better attack but slighly less test acc
                    if data_type == "PubMed":
                        target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.0001)
                        shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.0001)  # 01
                    else:
                        target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
                        shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.001)  # 01
                else:
                    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.0001)
                    if shadow_model_type =="SAGE":
                        shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.001)
                    else:
                        shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.0001)

                # for sampling only k edges # defense mechanism
                new_target_edge_index_k = k_edge_index(data_new.target_edge_index, how_many_edges_k) #target train
                new_all_edge_index_k = k_edge_index(data_new.all_edge_index, how_many_edges_k) # target test / all_edge_index

                '''
                Train and Test function for model
                '''


                # --*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
                # #----------------------------- TRAIN FUCNTION---------------------------
                def train(model, optimizer, isTarget=True):
                    model.train()
                    optimizer.zero_grad()
                    if isTarget:
                        out, nodes_and_neighbors = model(data_new.target_x,
                                                         data_new.target_edge_index)
                        loss = F.nll_loss(out, data_new.target_y)
                    else:
                        out, nodes_and_neighbors = model(data_new.shadow_x, data_new.shadow_edge_index)
                        loss = F.nll_loss(out, data_new.shadow_y)

                    pred = torch.exp(out)
                    loss.backward()
                    optimizer.step()

                    # approximate accuracy
                    if isTarget:
                        train_loss = loss.item() / int(data_new.target_train_mask.sum())
                        total_correct = int(pred.argmax(dim=-1).eq(data_new.target_y).sum()) / int(data_new.target_train_mask.sum())
                        # np.savetxt("save_Target_Train", pred.cpu().detach().numpy())
                    else:
                        train_loss = loss.item() / int(data_new.shadow_train_mask.sum())
                        total_correct = int(pred.argmax(dim=-1).eq(data_new.shadow_y).sum()) / int(data_new.shadow_train_mask.sum())
                        # np.savetxt("save_Shadow_Train", pred.cpu().detach().numpy())
                    return total_correct, train_loss




                def train_SAGE(model, optimizer, isTarget=True):
                    # print("================ Begin SAGE Train ==================")
                    model.train()

                    if isTarget:
                        total_loss = total_correct = 0
                        for batch_size, n_id, adjs in target_train_loader:
                            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                            adjs = [adj.to(device) for adj in adjs]
                            # print("adjs", adjs)

                            optimizer.zero_grad()
                            out, nodes_and_neighbors = model(data_new.target_x[n_id], adjs)

                            loss = F.nll_loss(out, data_new.target_y[n_id[:batch_size]])
                            loss.backward()
                            optimizer.step()

                            total_loss += float(loss)

                            total_correct += int(out.argmax(dim=-1).eq(data_new.target_y[n_id[:batch_size]]).sum())

                        loss = total_loss / len(target_train_loader)
                        approx_acc = total_correct / int(data_new.target_train_mask.sum())

                    else:
                        # Shadow training
                        total_loss = total_correct = 0
                        for batch_size, n_id, adjs in shadow_train_loader:
                            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                            adjs = [adj.to(device) for adj in adjs]
                            # print("adjs", adjs)

                            optimizer.zero_grad()
                            out, nodes_and_neighbors = model(data_new.shadow_x[n_id], adjs)

                            loss = F.nll_loss(out, data_new.shadow_y[n_id[:batch_size]])
                            loss.backward()
                            optimizer.step()

                            total_loss += float(loss)
                            total_correct += int(out.argmax(dim=-1).eq(data_new.shadow_y[n_id[:batch_size]]).sum())

                        loss = total_loss / len(shadow_train_loader)
                        approx_acc = total_correct / int(data_new.shadow_train_mask.sum())

                    # print("================ End SAGE Train ==================")
                    return approx_acc, loss


                # -----------------------------------------------------------------------------------------------
                # -----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*
                # -------------------------------- TEST FUNCTION -----------------------------------------
                def test(model, isTarget=True):
                    model.eval()
                    if isTarget:

                        '''InTrain Target'''

                        if remove_edge_at_test_index:
                            # dummy edges
                            pred, nodes_and_neighbors = model(data_new.target_x,
                                                              data_new.target_edge_index[:, :0])
                        elif use_nsd:
                            # sample only k edges
                            pred, nodes_and_neighbors = model(data_new.target_x,
                                                              new_target_edge_index_k)
                        else:
                            # Default i.e use full edge index
                            pred, nodes_and_neighbors = model(data_new.target_x,
                                                              data_new.target_edge_index)




                        if ismultiply_by_beta:

                            if use_binning:
                                pred = lbp_defense(pred, num_bins, beta, use_lbp)

                            else:
                                # #Original========> i.e no binning
                                pred = multiply_perturbation_defense(pred, beta)

                        if isoutput_perturb_defense:
                            pred = vanpd_defense(pred, beta)


                        pred_Intrain = pred.max(1)[1].to(device) # take maximum


                        # pred_Intrain = pred.max(1)[1].to(device)
                        # Actual probabilities


                        # scale to 1
                        if ismultiply_by_beta or isoutput_perturb_defense:
                            pred_Intrain_ps = normalize_posterior(pred)

                        else:
                            pred_Intrain_ps = torch.exp(pred)

                        np.savetxt(save_target_InTrain, pred_Intrain_ps.cpu().detach().numpy())

                        np.save(save_target_InTrain_nodes_neigbors, nodes_and_neighbors)

                        '''=============================== OutTrain Target===================================='''
                        if remove_edge_at_test_index:
                            # dummy edges
                            preds, nodes_and_neighbors = model(data_new.all_x,
                                                              data_new.all_edge_index[:, :0]) # [data_new.target_test_mask]
                        elif use_nsd:
                            # sample only k edges
                            preds, nodes_and_neighbors = model(data_new.all_x,
                                                              new_all_edge_index_k)
                        else:
                            # Default i.e use full edge index
                            preds, nodes_and_neighbors = model(data_new.all_x, data_new.all_edge_index)

                        nodes_and_neighbors = np.array(nodes_and_neighbors)

                        preds = preds[data_new.target_test_mask]
                        mask = data_new.target_test_mask.gt(
                            0)
                        mask = mask.cpu().numpy()
                        nodes_and_neighbors = nodes_and_neighbors[mask]

                        if ismultiply_by_beta:

                            if use_binning:
                                preds = lbp_defense(preds, num_bins, beta, use_lbp)

                            else:
                                #Original========> i.e no binning
                                # just multiply each posterior by 0.2 i.e beta
                                preds = multiply_perturbation_defense(preds, beta)



                        if isoutput_perturb_defense:

                            preds = vanpd_defense(preds, beta)


                        pred_out = preds.max(1)[1].to(device) # take maximum


                        # scale to 1
                        if ismultiply_by_beta or isoutput_perturb_defense:
                            pred_out_ps = normalize_posterior(preds)

                        else:
                            pred_out_ps = torch.exp(preds)


                        incremented_nodes_and_neighbors = []
                        for i in range(len(nodes_and_neighbors)):
                            res = nodes_and_neighbors[i][
                                1]
                            res_0 = nodes_and_neighbors[i][0]  # nodes_and_neighbors[i][0] + data.num_nodes
                            incremented_nodes_and_neighbors.append((res_0, res))


                        np.save(save_target_OutTrain_nodes_neigbors, incremented_nodes_and_neighbors)


                        nodes = []
                        for i in range(0, len(incremented_nodes_and_neighbors)):
                            nodes.append(incremented_nodes_and_neighbors[i][0])
                        nodes = np.array(nodes)
                        # print("nodesnodes",nodes)
                        preds_and_nodes = np.column_stack((pred_out_ps.cpu().detach().numpy(), nodes))
                        # print("preds_and_nodes", preds_and_nodes.shape)
                        np.savetxt(save_target_OutTrain, preds_and_nodes)  # pred_out_ps.cpu().detach().numpy()

                        # print("End OutTrain for target")

                        pred_labels = pred_out.tolist()
                        true_labels = data_new.all_y[data_new.target_test_mask].tolist()

                        train_acc = get_train_acc(data_new, pred_Intrain)
                        # Test n val are on full graph
                        test_acc = get_test_acc(data_new, pred_out)

                    else:

                        '''InTrain Shadow'''
                        pred, nodes_and_neighbors = model(data_new.shadow_x, data_new.shadow_edge_index)

                        if perturb_shadow:
                            pred = multiply_perturbation_defense(pred, beta)

                        pred_Intrain = pred.max(1)[1].to(device)

                        # Actual probabilities
                        # pred_Intrain_ps = torch.exp(model(data_new.shadow_x, data_new.shadow_edge_index)[data_new.shadow_train_mask])
                        if perturb_shadow:
                            pred_Intrain_ps = normalize_posterior(pred)
                        else:
                            pred_Intrain_ps = torch.exp(pred)

                        np.savetxt(save_shadow_InTrain, pred_Intrain_ps.cpu().detach().numpy())

                        '''OutTrain Shadow'''

                        # preds, nodes_and_neighbors = model(data_new.all_x, data_new.all_edge_index)[data_new.shadow_test_mask]
                        preds, nodes_and_neighbors = model(data_new.all_x, data_new.all_edge_index)  # [data_new.shadow_test_mask]
                        nodes_and_neighbors = np.array(nodes_and_neighbors)


                        preds = preds[data_new.shadow_test_mask]
                        mask = data_new.shadow_test_mask.gt(0)  # trick to
                        mask = mask.cpu().numpy()

                        nodes_and_neighbors = nodes_and_neighbors[mask]


                        # preds, nodes_and_neighbors = model(data_new.all_x, data_new.all_edge_index)[data_new.shadow_test_mask]
                        if perturb_shadow:
                            preds = multiply_perturbation_defense(preds, beta)

                        pred_out = preds.max(1)[1].to(device)

                        if perturb_shadow:
                            pred_out_ps = normalize_posterior(preds)
                        else:
                            pred_out_ps = torch.exp(preds)

                        np.savetxt(save_shadow_OutTrain, pred_out_ps.cpu().detach().numpy())

                        pred_labels = pred_out.tolist()
                        true_labels = data_new.all_y[data_new.shadow_test_mask].tolist()

                        # The train accuracy is not on the full graph. It's similar to approx_train_acc
                        train_acc = get_train_acc(data_new, pred_Intrain, False)
                        # Test n val are on full graph
                        test_acc = get_test_acc(data_new, pred_out, False)


                    # The f1 measures are on test dataset
                    f1_marco = get_marco_f1(data_new, pred_labels, true_labels, label_list)
                    f1_micro = get_micro_f1(data_new, pred_labels, true_labels, label_list)

                    return train_acc, test_acc, f1_marco, f1_micro

                def test_SAGE(model, isTarget=True):
                    # print("****************** SAGE test begin *************************")
                    model.eval()
                    if isTarget:

                        '''InTrain Target'''
                        pred = []
                        nodes_and_neighbors = []

                        total_target_train_correct = 0

                        for batch_size, n_id, adjs in target_train_loader:
                            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                            adjs = [adj.to(device) for adj in adjs]
                            # print("adjs test", adjs)
                            # print("n_id",len(n_id))

                            out, node_and_neigh = model(data_new.target_x[n_id], adjs)
                            out = torch.exp(out)
                            # print("out test", out.shape)
                            pred.append(out.cpu())
                            nodes_and_neighbors.append(node_and_neigh)

                            total_target_train_correct += int(out.argmax(dim=-1).eq(data_new.target_y[n_id[:batch_size]]).sum())

                        target_train_acc = total_target_train_correct / int(
                            data_new.target_train_mask.sum())  # similar to get_train_acc()


                        # Need to concat all preds cos it's per batch
                        pred_all_inTrain = torch.cat(pred, dim=0)


                        if ismultiply_by_beta:

                            if use_binning:
                                pred_all_inTrain = lbp_defense(pred_all_inTrain, num_bins, beta, use_lbp)

                            else:
                                # #Original========> i.e no binning
                                # # just multiply each posterior by 0.2 i,e beta
                                pred_all_inTrain = multiply_perturbation_defense(pred_all_inTrain, beta)



                        if isoutput_perturb_defense:

                            pred_all_inTrain = vanpd_defense(pred_all_inTrain, beta)



                        pred_Intrain = pred_all_inTrain.max(1)[1].to(device) # take maximum


                        # pred_Intrain = pred_all_inTrain.max(1)[1].to(device)
                        # Actual probabilities


                        # scale to 1
                        if ismultiply_by_beta or isoutput_perturb_defense:
                            pred_Intrain_ps = normalize_posterior(pred_all_inTrain)

                        else:
                            pred_Intrain_ps = pred_all_inTrain  # torch.exp(pred) #torch.exp(pred_all_inTrain)



                        np.savetxt(save_target_InTrain, pred_Intrain_ps.cpu().detach().numpy())

                        np.save(save_target_InTrain_nodes_neigbors, nodes_and_neighbors)  # TODO N

                        '''OutTrain Target'''
                        # This is where the difference is
                        out, nodes_and_neighbors = model.inference(data_new.all_x)


                        y_true = data_new.all_y.cpu().unsqueeze(-1)
                        y_pred = out.argmax(dim=-1, keepdim=True)


                        nodes_and_neighbors = np.array(nodes_and_neighbors)
                        preds = out[data_new.target_test_mask]
                        mask = data_new.target_test_mask.gt(
                            0)
                        mask = mask.cpu().numpy()

                        nodes_and_neighbors = nodes_and_neighbors[
                                              :data.num_nodes]

                        nodes_and_neighbors = nodes_and_neighbors[mask]

                        if ismultiply_by_beta:

                            if use_binning:
                                preds = lbp_defense(preds, num_bins, beta, use_lbp)


                            else:
                                # Original========> i.e no binning
                                # just multiply each posterior by 0.2 i.e beta
                                preds = multiply_perturbation_defense(preds, beta)


                        if isoutput_perturb_defense:
                            preds = vanpd_defense(preds, beta)

                        pred_out = preds.max(1)[1].to(device)  # take maximum

                        # scale to 1
                        if ismultiply_by_beta or isoutput_perturb_defense:
                            pred_out_ps = normalize_posterior(preds)

                        else:
                            pred_out_ps = torch.exp(preds)





                        incremented_nodes_and_neighbors = []
                        for i in range(len(nodes_and_neighbors)):
                            res = [x + num_test_Target for x in nodes_and_neighbors[i][1]]
                            res_0 = nodes_and_neighbors[i][0] + data.num_nodes
                            incremented_nodes_and_neighbors.append((res_0, res))


                        np.save(save_target_OutTrain_nodes_neigbors, incremented_nodes_and_neighbors)
                        # Simply the nodes are the last column of the "posterior"
                        nodes = []
                        for i in range(0, len(incremented_nodes_and_neighbors)):
                            nodes.append(incremented_nodes_and_neighbors[i][0])
                        nodes = np.array(nodes)
                        preds_and_nodes = np.column_stack((pred_out_ps.cpu().detach().numpy(), nodes))
                        np.savetxt(save_target_OutTrain, preds_and_nodes)
                        pred_labels = pred_out.tolist()
                        true_labels = data_new.all_y[data_new.target_test_mask].tolist()

                        train_acc = get_train_acc(data_new, pred_Intrain)
                        # Test n val are on full graph
                        test_acc = get_test_acc(data_new, pred_out)

                    else:

                        '''InTrain Shadow'''
                        pred = []
                        nodes_and_neighbors = []
                        for batch_size, n_id, adjs in shadow_train_loader:
                            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                            adjs = [adj.to(device) for adj in adjs]
                            # print("adjs test", adjs)
                            # print("n_id", len(n_id))

                            out, node_and_neigh = model(data_new.shadow_x[n_id], adjs)
                            out = torch.exp(out)
                            # print("out test", out.shape)
                            pred.append(out.cpu())
                            nodes_and_neighbors.append(node_and_neigh)

                        # Need to concat all preds cos it's per batch
                        pred_all_inTrain = torch.cat(pred, dim=0)

                        pred_Intrain = pred_all_inTrain.max(1)[1].to(device)

                        pred_Intrain_ps = pred_all_inTrain
                        np.savetxt(save_shadow_InTrain, pred_Intrain_ps.cpu().detach().numpy())


                        '''OutTrain Shadow'''

                        out, nodes_and_neighbors = model.inference(data_new.all_x)
                        # print("out OutTrain Shadow", out.shape)

                        y_true = data_new.all_y.cpu().unsqueeze(-1)
                        y_pred = out.argmax(dim=-1, keepdim=True)


                        nodes_and_neighbors = np.array(nodes_and_neighbors)
                        preds = out[data_new.shadow_test_mask]
                        # print("predsssssssssssssssss", torch.exp(preds).shape)
                        mask = data_new.shadow_test_mask.gt(
                            0)
                        mask = mask.cpu().numpy()


                        nodes_and_neighbors = nodes_and_neighbors[
                                              :data.num_nodes]

                        nodes_and_neighbors = nodes_and_neighbors[mask]


                        pred_out = preds.max(1)[1].to(device)
                        pred_out_ps = torch.exp(preds)

                        incremented_nodes_and_neighbors = []  # {}

                        for i in range(len(nodes_and_neighbors)):
                            # print(nodes_and_neighbors[i])  # list
                            # print(nodes_and_neighbors[i][0])
                            # print(nodes_and_neighbors[i][1])

                            res = [x + num_test_Shadow for x in nodes_and_neighbors[i][1]]  # increment from 630 num_test_Shadow
                            res_0 = nodes_and_neighbors[i][0] + data.num_nodes
                            incremented_nodes_and_neighbors.append((res_0, res))


                        nodes = []
                        for i in range(0, len(incremented_nodes_and_neighbors)):
                            nodes.append(incremented_nodes_and_neighbors[i][0])
                        nodes = np.array(nodes)
                        preds_and_nodes = np.column_stack((pred_out_ps.cpu().detach().numpy(), nodes))

                        np.savetxt(save_shadow_OutTrain,
                                   pred_out_ps.cpu().detach().numpy())
                        pred_labels = pred_out.tolist()
                        true_labels = data_new.all_y[data_new.shadow_test_mask].tolist()


                        train_acc = get_train_acc(data_new, pred_Intrain, False)
                        test_acc = get_test_acc(data_new, pred_out, False)


                    # The f1 measures are on test dataset
                    f1_marco = get_marco_f1(data_new, pred_labels, true_labels, label_list)
                    f1_micro = get_micro_f1(data_new, pred_labels, true_labels, label_list)

                    # print("****************** SAGE test End *************************")
                    return train_acc, test_acc, f1_marco, f1_micro


                # -----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*
                # -----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*
                # --------------------------TRAIN PROCESS-----------------------------------------------
                best_val_acc = best_val_acc = 0

                '''
                Training and testing target and shadow models
                '''

                if model_type == "SAGE":
                    if data_type == "CiteSeer" or data_type == "Cora":
                        if shadow_model_type == "SAGE" and model_type == "SAGE":
                            shadow_model_training_epoch = model_training_epoch = 16  # 301 #16 for CiteSeer n Cora, 101 for PubMed, 301 for Flickr n Reddit
                        else:
                            # train shadow as normal but target by 15 epochs
                            model_training_epoch = 16
                            shadow_model_training_epoch = 301
                    elif data_type == "PubMed":
                        if shadow_model_type == "SAGE" and model_type == "SAGE":
                            shadow_model_training_epoch = model_training_epoch = 101
                        else:
                            # train shadow as normal but target by 15 epochs
                            model_training_epoch = 101
                            shadow_model_training_epoch = 301
                    else:
                        shadow_model_training_epoch = model_training_epoch = 301
                else:
                    shadow_model_training_epoch = model_training_epoch = 301  # 301

                train_loss = approx_train_acc = train_acc = test_acc = marco =  micro = 0.0

                if not relax_target_data: # if I am relaxing the target model, then don't train the target cos it is useless.
                    # Target train
                    for epoch in range(1, model_training_epoch):
                        if model_type == "SAGE":
                            approx_train_acc, train_loss = train_SAGE(target_model, target_optimizer)
                            train_acc, test_acc, marco, micro = test_SAGE(target_model)
                        else:
                            approx_train_acc, train_loss = train(target_model, target_optimizer)
                            train_acc, test_acc, marco, micro = test(target_model)

                        log = 'TargetModel: {}-{} Epoch: {:03d}, Train Loss: {:.4f}, Approx Train: {:.4f}, Train: {:.4f}, Test: {:.4f},marco: {:.4f},micro: {:.4f}'
                        print(log.format(model_type,target_num_neurons, epoch, train_loss, approx_train_acc, train_acc, test_acc, marco, micro))
                        if epoch == model_training_epoch - 1:
                            result_file.write(log.format(model_type, target_num_neurons, epoch, train_loss, approx_train_acc, train_acc, test_acc, marco, micro) + "\n")

                target_train_loss, target_approx_train_acc, target_train_acc, target_test_acc, target_marco, target_micro = train_loss, approx_train_acc, train_acc, test_acc, marco, micro

                print()
                print("=========================================================End Target Train ==============================")

                train_loss = approx_train_acc = train_acc = test_acc = marco =  micro = 0.0

                # Shadow train
                for epoch in range(1, shadow_model_training_epoch):
                    if shadow_model_type == "SAGE":
                        approx_train_acc, train_loss = train_SAGE(shadow_model, shadow_optimizer, False)
                        train_acc, test_acc, marco, micro = test_SAGE(shadow_model, False)
                    else:
                        approx_train_acc, train_loss = train(shadow_model, shadow_optimizer, False)
                        train_acc, test_acc, marco, micro = test(shadow_model, False)

                    log = 'ShadowModel: {}-{} Epoch: {:03d}, Train Loss: {:.4f}, Approx Train: {:.4f}, Train: {:.4f}, Test: {:.4f},marco: {:.4f},micro: {:.4f}'
                    print(log.format(shadow_model_type,shadow_num_neurons, epoch, train_loss, approx_train_acc, train_acc, test_acc, marco, micro))
                    if epoch == shadow_model_training_epoch - 1:
                        result_file.write(log.format(shadow_model_type,shadow_num_neurons, epoch, train_loss, approx_train_acc, train_acc, test_acc, marco, micro) + "\n")

                shadow_train_loss, shadow_approx_train_acc, shadow_train_acc, shadow_test_acc, shadow_marco, shadow_micro = train_loss, approx_train_acc, train_acc, test_acc, marco, micro

                ''' =========================================== ATTACK ============================================== '''


                positive_attack_data = pd.read_csv(save_shadow_InTrain, header=None,
                                                   sep=" ")   #"posteriorsShadowTrain.txt"

                if relax_target_data:
                    # relaxation # ascending order sorting. No need for topn here except when cora
                    topn = 6
                    if data_type == "Cora": # if cora, then slice by topn because cora has 7 classes
                        print("bababa")
                        print("First positive_attack_data.shape", positive_attack_data.shape)
                        positive_attack_data = positive_attack_data.apply(np.sort, axis=1).apply(lambda x: x[-topn:]).apply(pd.Series)
                        print("Second positive_attack_data.shape", positive_attack_data.shape)
                    else:
                        positive_attack_data = positive_attack_data.apply(np.sort, axis=1).apply(lambda x: x[:]).apply(pd.Series)

                # target in and out data

                if relax_target_data:
                    # relaxation
                    relax_file_name = global_path+"posteriorsTargetTrain_"+mode+"_"+relax_target_data_type+ "_"+model_type+".txt"
                    target_data_for_testing_Intrain = pd.read_csv(relax_file_name, header=None,
                                                                  sep=" ")  # dataframe "posteriorsTargetTrain.txt"


                    # change for relaxation. Change target_data_for_testing_Intrain of cora / citeSeer to 6 posteriors
                    topn = 6
                    target_data_for_testing_Intrain = target_data_for_testing_Intrain.apply(np.sort, axis=1).apply(
                        lambda x: x[-topn:]).apply(pd.Series)
                    # target_data_for_testing_Intrain = target_data_for_testing_Intrain.apply(np.sort, axis=1).apply(lambda x: x[:]).apply(pd.Series)

                else:
                    target_data_for_testing_Intrain = pd.read_csv(save_target_InTrain, header=None,
                                                                  sep=" ")  # dataframe "posteriorsTargetTrain.txt"


                # Assign 1 to in data
                target_data_for_testing_Intrain["labels"] = 1

                # TODO N add nodeID from 0 to 629 (num_test_Target) == done

                if relax_target_data:
                    # relaxation i.e cora size or CiteSeer
                    target_data_for_testing_Intrain['nodeID'] = range(0,
                                                                      len(target_data_for_testing_Intrain))
                else:
                    target_data_for_testing_Intrain['nodeID'] = range(0,
                                                                      num_test_Target)

                if relax_target_data:
                    # relaxation
                    relax_file_name = global_path+"posteriorsTargetOut_"+mode+"_" + relax_target_data_type + "_"+model_type+ defense_type + ".txt"
                    target_data_for_testing_Outtrain_data = pd.read_csv(relax_file_name, header=None,
                                                                        sep=" ")  # dataframe "posteriorsTargetOut.txt"

                    target_data_for_testing_Outtrain = target_data_for_testing_Outtrain_data.iloc[:,
                                                       :-1]  # drop last. To be assigned later as nodeID

                    # relaxation. Change target_data_for_testing_Outtrain_data of cora to 6 posteriors
                    topn = 6
                    target_data_for_testing_Outtrain = target_data_for_testing_Outtrain.apply(np.sort, axis=1).apply(
                        lambda x: x[-topn:]).apply(pd.Series)
                    # target_data_for_testing_Outtrain = target_data_for_testing_Outtrain.apply(np.sort, axis=1).apply(
                    #     lambda x: x[:]).apply(pd.Series)

                else:
                    target_data_for_testing_Outtrain_data = pd.read_csv(save_target_OutTrain, header=None,
                                                                        sep=" ")  # dataframe "posteriorsTargetOut.txt"

                    target_data_for_testing_Outtrain = target_data_for_testing_Outtrain_data.iloc[:,
                                                       :-1]  # drop last. To be assigned later as nodeID




                target_data_for_testing_Outtrain['nodeID'] = target_data_for_testing_Outtrain_data.iloc[:, -1:].astype(
                    float).astype(
                    int)
                # Assign 0 to outdata
                target_data_for_testing_Outtrain["labels"] = 0

                # Assign 1 to training data
                positive_attack_data["labels"] = 1

                print("positive_attack_data.shape", positive_attack_data.shape)

                negative_attack_data = pd.read_csv(save_shadow_OutTrain, header=None, sep=" ")  # "posteriorsShadowOut.txt"

                if relax_target_data:
                    # relaxation # ascending order sorting. No need for topn here except when cora
                    topn = 6
                    if data_type == "Cora": # if cora, then slice by topn because cora has 7 classes
                        negative_attack_data = negative_attack_data.apply(np.sort, axis=1).apply(lambda x: x[-topn:]).apply(pd.Series)
                    else:
                        negative_attack_data = negative_attack_data.apply(np.sort, axis=1).apply(lambda x: x[:]).apply(pd.Series)



                # Assign 0 to out data
                negative_attack_data["labels"] = 0
                print("negative_attack_data.shape", negative_attack_data.shape)

                # Combine to single dataframe

                # combine them together
                attack_data_combo = [positive_attack_data, negative_attack_data]
                attack_data = pd.concat(attack_data_combo)

                target_data_for_testing_InAndOutTrain_combo = [target_data_for_testing_Intrain, target_data_for_testing_Outtrain]
                target_data_for_testing_InAndOutTrain = pd.concat(target_data_for_testing_InAndOutTrain_combo, sort=False)

                print("target_data_for_testing_InAndOutTrain", target_data_for_testing_InAndOutTrain.shape)

                print("attack_data.shape", attack_data.shape)

                X_attack = attack_data.drop("labels", axis=1)
                print("X_attack.shape", X_attack.shape)

                y_attack = attack_data["labels"]

                # let's do in and out for attack data (shadow)
                X_attack_InTrain = positive_attack_data.drop("labels", axis=1)
                y_attack_InTrain = positive_attack_data["labels"]

                X_attack_OutTrain = negative_attack_data.drop("labels", axis=1)
                y_attack_OutTrain = negative_attack_data["labels"]

                print("X_attack_InTrain", X_attack_InTrain.shape)
                print("X_attack_OutTrain", X_attack_OutTrain.shape)

                # For in train data (target)
                X_InTrain = target_data_for_testing_Intrain.drop(["labels", "nodeID"], axis=1)
                y_InTrain = target_data_for_testing_Intrain["labels"]
                nodeID_InTrain = target_data_for_testing_Intrain["nodeID"]

                # For Out train data
                X_OutTrain = target_data_for_testing_Outtrain.drop(["labels", "nodeID"], axis=1)
                y_OutTrain = target_data_for_testing_Outtrain["labels"]
                nodeID_OutTrain = target_data_for_testing_Outtrain["nodeID"]

                # For in out data
                X_InOutTrain = target_data_for_testing_InAndOutTrain.drop(["labels", "nodeID"], axis=1)
                print("X_InTrain.shape", X_InOutTrain.shape)

                y_InOutTrain = target_data_for_testing_InAndOutTrain["labels"]
                nodeID_InOutTrain = target_data_for_testing_InAndOutTrain["nodeID"]

                # convert to numpy
                # for shadow
                X_attack_InOut, y_attack_InOut = X_attack.to_numpy(), y_attack.to_numpy()

                X_attack_InTrain, X_attack_OutTrain = X_attack_InTrain.to_numpy(), X_attack_OutTrain.to_numpy()
                y_attack_InTrain, y_attack_OutTrain = y_attack_InTrain.to_numpy(), y_attack_OutTrain.to_numpy()

                # for target
                X_InTrain, y_InTrain, nodeID_InTrain = X_InTrain.to_numpy(), y_InTrain.to_numpy(), nodeID_InTrain.to_numpy()
                X_OutTrain, y_OutTrain, nodeID_OutTrain = X_OutTrain.to_numpy(), y_OutTrain.to_numpy(), nodeID_OutTrain.to_numpy()

                # for target
                X_InOutTrain, y_InOutTrain, nodeID_InOutTrain = X_InOutTrain.to_numpy(), y_InOutTrain.to_numpy(), nodeID_InOutTrain.to_numpy()

                # # Plot graphs
                #
                # plt.imshow(X_attack_InTrain, interpolation='nearest', aspect='auto')
                # plt.colorbar()
                # plt.tight_layout()
                # plt.title('Positive: In Train Posteriors')
                # plt.show()
                #
                # plt.imshow(X_attack_OutTrain, interpolation='nearest', aspect='auto')
                # plt.colorbar()
                # plt.tight_layout()
                # plt.title('Negative: Out Train Posteriors')
                # plt.show()
                #
                #
                # plt.imshow(X_InTrain, interpolation='nearest', aspect='auto')
                # plt.colorbar()
                # plt.tight_layout()
                # plt.title('Positive: Target Posteriors')
                # plt.show()
                #
                # plt.imshow(X_OutTrain, interpolation='nearest', aspect='auto')
                # plt.colorbar()
                # plt.tight_layout()
                # plt.title('Negative: Target Posteriors')
                # plt.show()

                attack_train_data_X, attack_test_data_X, attack_train_data_y, attack_test_data_y = train_test_split(X_attack,
                                                                                                                    y_attack,
                                                                                                                    test_size=50,
                                                                                                                    stratify=y_attack,
                                                                                                                    random_state=rand_state)
                print("baba")

                # convert series data to numpy array
                attack_train_data_X, attack_test_data_X, attack_train_data_y, attack_test_data_y = attack_train_data_X.to_numpy(), attack_test_data_X.to_numpy(), attack_train_data_y.to_numpy(), attack_test_data_y.to_numpy()

                # Attack_train
                attack_train_data = torch.utils.data.TensorDataset(torch.from_numpy(attack_train_data_X).float(), torch.from_numpy(
                    attack_train_data_y))  # convert to float to fix  uint8_t overflow error
                attack_train_data_loader = torch.utils.data.DataLoader(attack_train_data, batch_size=32, shuffle=True)

                # Attack_test = combo of targettrain and targetOut
                attack_test_data = torch.utils.data.TensorDataset(torch.from_numpy(attack_test_data_X).float(), torch.from_numpy(
                    attack_test_data_y))  # convert to float to fix  uint8_t overflow error
                attack_test_data_loader = torch.utils.data.DataLoader(attack_test_data, batch_size=32, shuffle=True)

                # Training InData
                target_data_for_testing_InTrain_data = torch.utils.data.TensorDataset(torch.from_numpy(X_InTrain).float(),
                                                                                      torch.from_numpy(y_InTrain),
                                                                                      torch.from_numpy(nodeID_InTrain))
                target_data_for_testing_InTrain_data_loader = torch.utils.data.DataLoader(target_data_for_testing_InTrain_data,
                                                                                          batch_size=64, shuffle=False)

                # Training OutData
                target_data_for_testing_OutTrain_data = torch.utils.data.TensorDataset(torch.from_numpy(X_OutTrain).float(),
                                                                                       torch.from_numpy(y_OutTrain),
                                                                                       torch.from_numpy(nodeID_OutTrain))
                target_data_for_testing_OutTrain_data_loader = torch.utils.data.DataLoader(target_data_for_testing_OutTrain_data,
                                                                                           batch_size=64, shuffle=False)

                # Training InOut Data
                target_data_for_testing_InOutTrain_data = torch.utils.data.TensorDataset(torch.from_numpy(X_InOutTrain).float(),
                                                                                         torch.from_numpy(y_InOutTrain),
                                                                                         torch.from_numpy(nodeID_InOutTrain))
                target_data_for_testing_InOutTrain_data_loader = torch.utils.data.DataLoader(
                    target_data_for_testing_InOutTrain_data,
                    batch_size=64, shuffle=True)

                class AttackModel(nn.Module):
                    def __init__(self):
                        super().__init__()

                        # inputs to hidden layer linear transformation
                        # Note, when using Linear, weight and biases are randomly initialized for you
                        self.hidden = nn.Linear(dataset.num_classes, 100)
                        self.hidden2 = nn.Linear(100, 50)
                        # output layer, 10 units - one for each digits
                        self.output = nn.Linear(50, 2)


                    def forward(self, x):
                        # Hidden layer with sigmoid activation
                        x = F.sigmoid(self.hidden(x))
                        x = F.sigmoid(self.hidden2(x))
                        # output layer with softmax activation
                        x = F.softmax(self.output(x), dim=1)
                        # print("xxxxxxxx", x)

                        return x


                class Net(nn.Module):
                    # define nn
                    def __init__(self):
                        super(Net, self).__init__()
                        if relax_target_data:
                            self.fc1 = nn.Linear(6, 100) #using only 6 posteriors relaxation
                        else:
                            self.fc1 = nn.Linear(dataset.num_classes, 100) #normal
                        self.fc2 = nn.Linear(100, 50)
                        self.fc3 = nn.Linear(50, 2)
                        self.softmax = nn.Softmax(dim=1)

                    def forward(self, X):
                        # print("attack X",X)
                        X = F.relu(self.fc1(X))
                        X = F.relu(self.fc2(X))
                        X = self.fc3(X)
                        X = self.softmax(X)

                        return X


                def init_weights(m):
                    if type(m) == nn.Linear:
                        torch.nn.init.xavier_uniform(m.weight)
                        m.bias.data.fill_(0.01)


                # create the ntwk
                attack_model = Net()  # AttackModel()
                attack_model = attack_model.to(device)
                attack_model.apply(init_weights)  # initialize weight rather than randomly
                print(attack_model)


                def attack_train(model, trainloader, testloader, criterion, optimizer, epochs, steps=0):
                    # train ntwk
                    final_train_loss = 0
                    train_losses, test_losses = [], []
                    posteriors = []
                    for e in range(epochs):
                        running_loss = 0
                        train_accuracy = 0

                        for features, labels in trainloader:
                            model.train()
                            features, labels = features.to(device), labels.to(device)


                            optimizer.zero_grad()

                            features = features.view(features.shape[0], -1)

                            logps = model(features)  # log probabilities
                            # print("labelsssss", labels.shape)
                            loss = criterion(logps, labels)

                            # Actual probabilities
                            ps = logps  # torch.exp(logps) #Only use this if the loss is nlloss


                            top_p, top_class = ps.topk(1,
                                                       dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
                            # print(top_p)
                            equals = top_class == labels.view(
                                *top_class.shape)  # making the shape of the label and top class the same
                            train_accuracy += torch.mean(equals.type(torch.FloatTensor))

                            loss.backward()
                            optimizer.step()

                            running_loss += loss.item()
                        else:

                            test_loss, test_accuracy, _, _, _, _, _, _, _ = attack_test(model, testloader, trainTest=True)

                            # set model back yo train model
                            model.train()
                            # scheduler.step()

                            train_losses.append(running_loss / len(trainloader))
                            test_losses.append(test_loss)

                            # get final train loss. To be returned at the end of the training loop
                            final_train_loss = running_loss / len(trainloader)

                            print("Epoch: {}/{}..".format(e + 1, epochs),
                                  "Training loss: {:.5f}..".format(running_loss / len(trainloader)),
                                  "Test Loss: {:.5f}..".format(test_loss),
                                  "Train Accuracy: {:.3f}".format(train_accuracy / len(trainloader)),
                                  "Test Accuracy: {:.3f}".format(test_accuracy)
                                  )

                    # # plot train and test loss
                    # plt.show()
                    # plt.plot(train_losses)
                    # plt.plot(test_losses)
                    # plt.title('Model Losses')
                    # plt.ylabel('loss')
                    # plt.xlabel('epoch')
                    # plt.legend(['train', 'val'], loc='upper left')
                    # plt.show()

                    return final_train_loss


                def attack_test(model, testloader, singleClass=False, trainTest=False):  # TODO N
                    test_loss = 0
                    test_accuracy = 0
                    auroc = 0
                    precision = 0
                    recall = 0
                    f_score = 0

                    posteriors = []
                    all_nodeIDs = []
                    true_predicted_nodeIDs_and_class = {}
                    false_predicted_nodeIDs_and_class = {}

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        # Doing validation

                        # set model to evaluation mode
                        model.eval()

                        if trainTest:
                            for features, labels in testloader:
                                features, labels = features.to(device), labels.to(device)
                                # features = features.unsqueeze(1)  # unsqueeze
                                features = features.view(features.shape[0], -1)
                                logps = model(features)
                                test_loss += criterion(logps, labels)

                                # Actual probabilities
                                ps = logps  # torch.exp(logps)
                                posteriors.append(ps)

                                # if singleclass=false
                                if not singleClass:
                                    y_true = labels.cpu().unsqueeze(-1)

                                    y_pred = ps.argmax(dim=-1, keepdim=True)


                                    # uncomment this to show AUROC
                                    auroc += roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
                                    # print("auroc", auroc)

                                    precision += precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                                    recall += recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                                    f_score += f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                                top_p, top_class = ps.topk(1,
                                                           dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
                                # print(top_p)
                                equals = top_class == labels.view(
                                    *top_class.shape)  # making the shape of the label and top class the same
                                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

                        else:
                            print("len(testloader.dataset)", len(testloader.dataset))
                            for features, labels, nodeIDs in testloader:
                                print("nodeIDs", nodeIDs)
                                features, labels = features.to(device), labels.to(device)
                                # features = features.unsqueeze(1)  # unsqueeze
                                features = features.view(features.shape[0], -1)
                                logps = model(features)
                                test_loss += criterion(logps, labels)

                                # Actual probabilities
                                ps = logps  # torch.exp(logps)
                                posteriors.append(ps)
                                all_nodeIDs.append(nodeIDs)

                                # if singleclass=false
                                if not singleClass:
                                    y_true = labels.cpu().unsqueeze(-1)
                                    # print("y_true", y_true)
                                    y_pred = ps.argmax(dim=-1, keepdim=True)
                                    # print("y_pred", y_pred)

                                    # uncomment this to show AUROC
                                    auroc += roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
                                    # print("auroc", auroc)

                                    precision += precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                                    recall += recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                                    f_score += f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                                top_p, top_class = ps.topk(1,
                                                           dim=1)
                                equals = top_class == labels.view(
                                    *top_class.shape)  # making the shape of the label and top class the same


                                print("equals", len(equals))
                                for i in range(len(equals)):
                                    if equals[i]:

                                        true_predicted_nodeIDs_and_class[nodeIDs[i].item()] = top_class[i].item()

                                    else:
                                        false_predicted_nodeIDs_and_class[nodeIDs[i].item()] = top_class[i].item()


                                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

                    test_accuracy = test_accuracy / len(testloader)
                    test_loss = test_loss / len(testloader)
                    final_auroc = auroc / len(testloader)
                    final_precision = precision / len(testloader)
                    final_recall = recall / len(testloader)
                    final_f_score = f_score / len(testloader)



                    return test_loss, test_accuracy, posteriors, final_auroc, final_precision, final_recall, final_f_score, true_predicted_nodeIDs_and_class, false_predicted_nodeIDs_and_class


                '''Initialization / params for attack model'''

                criterion = nn.CrossEntropyLoss()  # nn.NLLLoss() # cross entropy loss

                optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01)  # 0.01 #0.00001

                epochs = 100  # 1000

                '''==============Train and test Attack model ========== '''

                attack_train(attack_model, attack_train_data_loader, attack_test_data_loader, criterion, optimizer, epochs)

                # test to confirm using attack_test_data_loader
                _, test_accuracyConfirmTest, posteriors, auroc, precision, recall, f_score, _, _ = attack_test(attack_model,
                                                                                                               attack_test_data_loader,
                                                                                                               trainTest=True)  # TODO N
                # print(posteriors)

                # This is d result on the test set we used i.e split the attack data into train and test.
                # Size of the test = 50
                print("To confirm using attack_test_data_loader (50 test samples): {:.3f}".format(test_accuracyConfirmTest),
                      "AUROC: {:.3f}".format(auroc), "precision: {:.3f}".format(precision), "recall {:.3f}".format(recall))

                # test for InOut train target data
                # This is the one we are interested in
                _, test_accuracyInOut, posteriors, auroc, precision, recall, f_score, true_predicted_nodeIDs_and_class, false_predicted_nodeIDs_and_class = attack_test(
                    attack_model,
                    target_data_for_testing_InOutTrain_data_loader)



                # print(posteriors)
                print("true_predicted_nodeIDs_and_class", len(true_predicted_nodeIDs_and_class))
                print("false_predicted_nodeIDs_and_class", len(false_predicted_nodeIDs_and_class))
                print("Test accuracy with Target Train InOut: {:.3f}".format(test_accuracyInOut), "AUROC: {:.3f}".format(auroc),
                      "precision: {:.3f}".format(precision), "recall {:.3f}".format(recall), "F1 score {:.3f}".format(f_score),
                      "===> Interested!")

                interested_precision = precision
                interested_recall = recall
                interested_auroc = auroc


                result_file.write(
                    "Test accuracy with Target Train InOut: {:.3f} ".format(test_accuracyInOut) + " AUROC: {:.3f}".format(auroc) +
                    " precision: {:.3f}".format(precision) + " recall {:.3f}".format(recall) + " F1 score {:.3f}".format(f_score) +
                    "===> Interested! \n")



                target_train_loss_acc_per_run.append(round(target_train_loss, 4))
                target_approx_train_acc_per_run.append(round(target_approx_train_acc, 4))
                target_train_acc_per_run.append(round(target_train_acc, 4))
                target_test_acc_per_run.append(round(target_test_acc, 4))
                target_macro_acc_per_run.append(round(target_marco, 4))
                target_micro_acc_per_run.append(round(target_micro, 4))

                shadow_train_loss_acc_per_run.append(round(shadow_train_loss, 4))
                shadow_approx_train_acc_per_run.append(round(shadow_approx_train_acc, 4))
                shadow_train_acc_per_run.append(round(shadow_train_acc, 4))
                shadow_test_acc_per_run.append(round(target_test_acc, 4))
                shadow_macro_acc_per_run.append(round(target_marco, 4))
                shadow_micro_acc_per_run.append(round(target_micro, 4))

                precision_per_run.append(round(precision, 3))
                auroc_per_run.append(round(auroc, 3))
                recall_per_run.append(round(recall, 3))
                f1_score_per_run.append(round(f_score, 3))



                # test for Only In train target data
                _, test_accuracyIn, posteriors, _, precision, recall, f_score, _, _ = attack_test(attack_model,
                                                                                                  target_data_for_testing_InTrain_data_loader,
                                                                                                  True)
                print("Test accuracy with Target Train In: {:.3f}".format(test_accuracyIn), "precision: {:.3f}".format(precision),
                      "recall {:.3f}".format(recall), "F1 score {:.3f}".format(f_score))

                # test for Only Out train target data
                _, test_accuracyOut, posteriors, _, precision, recall, f_score, _, _ = attack_test(attack_model,
                                                                                                   target_data_for_testing_OutTrain_data_loader,
                                                                                                   True)
                print("Test accuracy with Target Train Out: {:.3f}".format(test_accuracyOut), "precision: {:.3f}".format(precision),
                      "recall {:.3f}".format(recall), "F1 score {:.3f}".format(f_score))

                result_file.write(
                    "Test accuracy with Target Train In: {:.3f} ".format(
                        test_accuracyIn) + " |=====| Test accuracy with Target Train Out: {:.3f}".format(test_accuracyOut) + "\n")

                print("data_type", data_type)
                print("model_type", model_type)
                end_time = time.time()

                total_time = round(end_time - start_time, 3)
                print("WhichRun", which_run, " Total time", total_time)

                label_loss = normal_test - target_test_acc #"nonoise - noisy" # Fill manually"nodefnse target test acc - currentrun" # some function

                # negative means percentage increase
                percentage_drop_precision = ((interested_precision-normal_precision)/normal_precision) * -100
                percentage_drop_recall = ((interested_recall-normal_recall)/normal_recall) * -100
                percentage_drop_auroc = ((interested_auroc-normal_auroc)/normal_auroc) * -100

                save_target_InTrainNonoise = "posteriorsTargetTrain_" + mode + "_" + data_type + "_" + model_type + "normal" + ".txt"
                save_target_InTrainNoisy = "posteriorsTargetTrain_" + mode + "_" + data_type + "_" + model_type + defense_type + ".txt"

                # Note, this is only for the mmeber nodes
                confidence_distortion = compute_confidence_distortion(save_target_InTrainNoisy, save_target_InTrainNonoise)


                # write defense type to file
                result_file.write(" || Defense Type: " + defense_type + " label loss: " + str(round(label_loss, 3)) + " confidence distortion: "+ str(round(confidence_distortion, 3)) + "|| \n")
                result_file.write(" || % Drop === AUROC: " + str(round(percentage_drop_auroc, 3)) + " Precision: " + str(
                    round(percentage_drop_precision, 3)) + " Recall: " + str(round(percentage_drop_recall, 3)) + "|| \n")

                # result_file.write("Data:"+data_type +" Model:"+ model_type+"\n\n\n")
                result_file.write(" ================ WhichRun: " + str(
                    which_run) + " || Data: " + data_type + " || Model: " + model_type + " || Time: " + str(
                    total_time) + " || rand_state: " + str(rand_state) + " ================== \n\n\n")

                result_file.close()


                total_time_per_run.append(total_time)



                if model_type != "SAGE" and relax_target_data!= False:
                    # TODO cos neighbor isnt good! fix b4 cuncomment for SAGE. Others are good!
                    print("==================== Begin Quick analysis===============================")

                    # Load
                    read_target_InTrain_nodes_neigbors_lookup = np.load(save_target_InTrain_nodes_neigbors,
                                                                        allow_pickle='TRUE')  # TODO C

                    read_target_OutTrain_nodes_neigbors_lookup = np.load(save_target_OutTrain_nodes_neigbors,
                                                                         allow_pickle='TRUE')  # TODO C

                    print("read_target_InTrain_nodes_neigbors_lookup", len(read_target_InTrain_nodes_neigbors_lookup))
                    print("read_target_OutTrain_nodes_neigbors_lookup", len(read_target_OutTrain_nodes_neigbors_lookup))
                    # Merge the 2 list #dictionary
                    all_nodes_lookup = np.concatenate(
                        (read_target_InTrain_nodes_neigbors_lookup, read_target_OutTrain_nodes_neigbors_lookup), axis=0)  # TODO C
                    print(len(all_nodes_lookup))

                    # for only train nodes {for stat}
                    train_nodes_lookup = read_target_InTrain_nodes_neigbors_lookup

                    print("all_nodes_lookup", all_nodes_lookup)

                    # convert list to dict
                    all_nodes_lookup_dict = {all_nodes_lookup[nodeID][0]: all_nodes_lookup[nodeID][1] for nodeID in
                                             range(0, len(all_nodes_lookup))}
                    print("all_nodes_lookup_dict", all_nodes_lookup_dict)


                    train_nodes_lookup_dict = {train_nodes_lookup[nodeID][0]: train_nodes_lookup[nodeID][1] for nodeID in
                                               range(0, len(train_nodes_lookup))}

                    print("train_nodes_lookup_dict", train_nodes_lookup_dict)


                    all_correct_classified_nodes_as1 = []
                    all_correct_classified_nodes_as0 = []

                    all_incorrect_classified_nodes_as1 = []
                    all_incorrect_classified_nodes_as0 = []
                    correct_edge_index = []
                    incorrect_edge_index = []


                    # this stores the fractions already
                    global_true_homophily = []
                    global_pred_homophily = []
                    correct_incorrect_homophily_prediction = []  # 1=correct, 0 = incorrect!

                    for nodeID in range(0, data.num_nodes):  # true_predicted_nodeIDs_and_class.keys():
                        if nodeID in true_predicted_nodeIDs_and_class or nodeID in false_predicted_nodeIDs_and_class:

                            # get value not index for all_nodes_lookup cos its now a list?
                            if nodeID in true_predicted_nodeIDs_and_class:
                                print("Node", nodeID, "==>", all_nodes_lookup_dict[nodeID], "True Predicted class ==>",
                                      true_predicted_nodeIDs_and_class[nodeID])  # TODO C

                            if nodeID in false_predicted_nodeIDs_and_class:
                                print("Node", nodeID, "==>", all_nodes_lookup_dict[nodeID], "False Predicted class ==>",
                                      false_predicted_nodeIDs_and_class[nodeID])  # TODO C

                            true_homophily = []
                            pred_homophily = []

                            selected_node = all_nodes_lookup_dict[nodeID]

                            if nodeID in true_predicted_nodeIDs_and_class:
                                true_class_of_selected_node = true_predicted_nodeIDs_and_class[nodeID]
                                predicted_class_of_selected_node = true_predicted_nodeIDs_and_class[nodeID]
                            else:
                                true_class_of_selected_node = 1 - false_predicted_nodeIDs_and_class[nodeID]  # inverse
                                predicted_class_of_selected_node = false_predicted_nodeIDs_and_class[nodeID]

                            print("true_class_of_selected_node", true_class_of_selected_node)
                            print("predicted_class_of_selected_node", predicted_class_of_selected_node)

                            # Homiphily begins
                            # deal with true predicted first
                            if selected_node:  # if neghbors of selected node are not empty

                                # get correct n incorrect pred info
                                if nodeID in true_predicted_nodeIDs_and_class:
                                    correct_incorrect_homophily_prediction.append(1)  # correct pred homophily
                                else:
                                    correct_incorrect_homophily_prediction.append(0)  # incorrect pred homophily

                                # print("baba")
                                for i in range(0, len(selected_node)):  # get all neighbors



                                    neigh = selected_node[i]  # get the neighbor(s) of the node
                                    print("neigh", neigh)
                                    membership = 100  # intial value

                                    # This gives true homophily
                                    if neigh in true_predicted_nodeIDs_and_class:
                                        membership = true_predicted_nodeIDs_and_class[neigh]
                                    elif neigh in false_predicted_nodeIDs_and_class:  # if in the false pred # inverse
                                        membership = 1 - false_predicted_nodeIDs_and_class[neigh]
                                    else:
                                        membership = true_class_of_selected_node  # true_predicted_nodeIDs_and_class[nodeID] # becomes membership of the initial node if key not in true_predicted_nodeIDs_and_class

                                    true_homophily.append(membership)

                                    print("membership true", membership)
                                    print("End =========")

                                    # Predicted homophily: Whatever the data predicts is true
                                    if neigh in true_predicted_nodeIDs_and_class:
                                        membership = true_predicted_nodeIDs_and_class[neigh]
                                    elif neigh in false_predicted_nodeIDs_and_class:
                                        membership = false_predicted_nodeIDs_and_class[
                                            neigh]  # no need for inverse since this is the predicted
                                    else:
                                        membership = predicted_class_of_selected_node  # true_predicted_nodeIDs_and_class[nodeID] # becomes membership of the initial node if key not in true_predicted_nodeIDs_and_class

                                    pred_homophily.append(membership)

                                    print("membership pred", membership)

                                print("true_homophily", true_homophily)
                                print("pred_homophily", pred_homophily)

                                # look thru each true and pred homophily, compare each element with their true n pred membership. Sum them and get average.
                                # Append to global_true_homophily and global_pred_homophily

                                frac_true_homophily = 0
                                frac_pred_homophily = 0

                                for i in range(0, len(true_homophily)):  # cos len of true_homophily n pred_homophily are the same
                                    if true_class_of_selected_node == true_homophily[i]:
                                        frac_true_homophily += 1
                                    if predicted_class_of_selected_node == pred_homophily[i]:
                                        frac_pred_homophily += 1

                                frac_true_homophily = frac_true_homophily / len(true_homophily)
                                frac_pred_homophily = frac_pred_homophily / len(pred_homophily)

                                print("frac_true_homophily = ", frac_true_homophily)
                                print("frac_pred_homophily = ", frac_pred_homophily)

                                global_true_homophily.append(frac_true_homophily)
                                global_pred_homophily.append(frac_pred_homophily)

                            # Homophily Ends

                    print("global_true_homophily", global_true_homophily)
                    print("global_pred_homophily", global_pred_homophily)
                    print("correct_incorrect_homophily_prediction", correct_incorrect_homophily_prediction)

                    np.savetxt(save_global_pred_homophily, global_pred_homophily)
                    np.savetxt(save_global_true_homophily, global_true_homophily)
                    np.savetxt(save_correct_incorrect_homophily_prediction, correct_incorrect_homophily_prediction)

                    print("lenghts", len(correct_incorrect_homophily_prediction), len(global_true_homophily),
                          len(global_pred_homophily))

                    # # Plot graph!
                    # sns.displot(global_true_homophily)
                    # plt.xlabel('Values', size=10)
                    # plt.ylabel('Counts', size=10)
                    # plt.show()
                    #
                    # sns.displot(global_pred_homophily)
                    # plt.xlabel('Values', size=10)
                    # plt.ylabel('Counts', size=10)
                    # plt.show()

                    # sys.exit()

                    # # this stores the fractions already
                    # global_true_homophily = []
                    # global_pred_homophily = []

                    for nodeID in true_predicted_nodeIDs_and_class.keys():
                        # get value not index for all_nodes_lookup cos its now a list?
                        print("Node", nodeID, "==>", all_nodes_lookup_dict[nodeID], "True Predicted class ==>",
                              true_predicted_nodeIDs_and_class[nodeID])  # TODO C

                        # true_homophily = []
                        # pred_homophily = []
                        #
                        # selected_node = all_nodes_lookup_dict[nodeID]

                        # if nodeID in true_predicted_nodeIDs_and_class:
                        #     true_class_of_selected_node = true_predicted_nodeIDs_and_class[nodeID]
                        #     predicted_class_of_selected_node = true_predicted_nodeIDs_and_class[nodeID]
                        # else:
                        #     true_class_of_selected_node = 1- false_predicted_nodeIDs_and_class[nodeID] #inverse
                        #     predicted_class_of_selected_node = false_predicted_nodeIDs_and_class[nodeID]
                        #
                        # print("true_class_of_selected_node", true_class_of_selected_node)
                        # print("predicted_class_of_selected_node", predicted_class_of_selected_node)

                        # # Homiphily begins
                        # # deal with true predicted first
                        # if selected_node: # if neghbors of selected node are not empty
                        #     # print("baba")
                        #     for i in range(0,len(selected_node)): #get all neighbors
                        #
                        #         # if i in true_predicted_nodeIDs_and_class: #get true class of the node
                        #         #     true_membership = true_predicted_nodeIDs_and_class[nodeID]
                        #         # elif i in false_predicted_nodeIDs_and_class:
                        #         #     true_membership = 1- false_predicted_nodeIDs_and_class[nodeID]
                        #         # else:
                        #         #     print("No membership. Stopping program now")
                        #         #     sys.exit() #exit the program!
                        #
                        #         neigh = selected_node[i] #get the neighbor(s) of the node
                        #         print("neigh", neigh)
                        #         membership = 100 # intial value
                        #
                        #         # This gives true homophily
                        #         if neigh in true_predicted_nodeIDs_and_class:
                        #             membership = true_predicted_nodeIDs_and_class[neigh]
                        #         elif neigh in false_predicted_nodeIDs_and_class: # if in the false pred # inverse
                        #             membership = 1-false_predicted_nodeIDs_and_class[neigh]
                        #         else:
                        #             membership = true_class_of_selected_node #true_predicted_nodeIDs_and_class[nodeID] # becomes membership of the initial node if key not in true_predicted_nodeIDs_and_class
                        #
                        #         true_homophily.append(membership)
                        #
                        #         print("membership true", membership)
                        #         print("End =========")
                        #
                        #         # Predicted homophily: Whatever the data predicts is true
                        #         if neigh in true_predicted_nodeIDs_and_class:
                        #             membership = true_predicted_nodeIDs_and_class[neigh]
                        #         elif neigh in false_predicted_nodeIDs_and_class:
                        #             membership = false_predicted_nodeIDs_and_class[neigh] # no need for inverse since this is the predicted
                        #         else:
                        #             membership = predicted_class_of_selected_node #true_predicted_nodeIDs_and_class[nodeID] # becomes membership of the initial node if key not in true_predicted_nodeIDs_and_class
                        #
                        #         pred_homophily.append(membership)
                        #
                        #         print("membership pred", membership)
                        #
                        #     print("true_homophily", true_homophily)
                        #     print("pred_homophily", pred_homophily)
                        #
                        #     # look thru each true and pred homophily, compare each element with their true n pred membership. Sum them and get average.
                        #     # Append to global_true_homophily and global_pred_homophily
                        #
                        #     frac_true_homophily = 0
                        #     frac_pred_homophily = 0
                        #
                        #     for i in range(0, len(true_homophily)):  # cos len of true_homophily n pred_homophily are the same
                        #         if true_class_of_selected_node == true_homophily[i]:
                        #             frac_true_homophily += 1
                        #         if predicted_class_of_selected_node == pred_homophily[i]:
                        #             frac_pred_homophily += 1
                        #
                        #     frac_true_homophily = frac_true_homophily / len(true_homophily)
                        #     frac_pred_homophily = frac_pred_homophily / len(pred_homophily)
                        #
                        #     print("frac_true_homophily = ", frac_true_homophily)
                        #     print("frac_pred_homophily = ", frac_pred_homophily)
                        #
                        #     global_true_homophily.append(frac_true_homophily)
                        #     global_pred_homophily.append(frac_pred_homophily)
                        #
                        #
                        # # Homophily Ends

                        # all_correct_classified_nodes.append(nodeID)

                        # cater for 1 and 0
                        if true_predicted_nodeIDs_and_class[nodeID] == 1:
                            all_correct_classified_nodes_as1.append(nodeID)
                        else:
                            all_correct_classified_nodes_as0.append(nodeID)

                        for i in range(0, len(all_nodes_lookup_dict[nodeID])):
                            edge_ind = (nodeID, all_nodes_lookup_dict[nodeID][i])
                            # print("edge_ind", edge_ind)
                            correct_edge_index.append((edge_ind))

                    # print("global_true_homophily", global_true_homophily)
                    # print("global_pred_homophily", global_pred_homophily)
                    #
                    # # Plot graph!
                    # sns.displot(global_true_homophily)
                    # plt.xlabel('Values', size=10)
                    # plt.ylabel('Counts', size=10)
                    # plt.show()
                    #
                    # sns.displot(global_pred_homophily)
                    # plt.xlabel('Values', size=10)
                    # plt.ylabel('Counts', size=10)
                    # plt.show()

                    print("======================End True predicted=============================")

                    for nodeID in false_predicted_nodeIDs_and_class.keys():
                        # get value not index for all_nodes_lookup cos its now a list?
                        print("Node", nodeID, "==>", all_nodes_lookup_dict[nodeID], "False Predicted class ==>",
                              false_predicted_nodeIDs_and_class[nodeID])  # TODO C

                        # cater for different colors for wrongly predicted as 1 or 0
                        if false_predicted_nodeIDs_and_class[nodeID] == 1:
                            all_incorrect_classified_nodes_as1.append(nodeID)
                        else:
                            all_incorrect_classified_nodes_as0.append(nodeID)

                        for i in range(0, len(all_nodes_lookup_dict[nodeID])):
                            edge_ind = (nodeID, all_nodes_lookup_dict[nodeID][i])
                            # print("edge_ind", edge_ind)
                            incorrect_edge_index.append((edge_ind))


                    def plot_graph_result(edges_correct, nodes0, nodes1, edges_incorrect, nodes2, nodes3):
                        print("edges", edges_correct)
                        # labels = labels.numpy() #dataset.data.y.numpy()

                        # correct 0
                        G0 = nx.Graph()
                        G0.add_nodes_from(nodes0)
                        G0.add_edges_from(edges_correct)

                        # correct 1
                        G1 = nx.Graph()
                        G1.add_nodes_from(nodes1)
                        G1.add_edges_from(edges_correct)

                        # wrongly pred as 0
                        G2 = nx.Graph()
                        G2.add_nodes_from(nodes2)
                        G2.add_edges_from(edges_incorrect)

                        # wrongly pred as 1
                        G3 = nx.Graph()
                        G3.add_nodes_from(nodes3)  # only change nodes. The edges is same
                        G3.add_edges_from(edges_incorrect)

                        # plt.subplot(111)
                        options = {
                            'node_size': 30,
                            'width': 0.2,
                        }

                        nx.draw(G0, with_labels=False, node_color="#8E44AD", cmap=plt.cm.tab10, font_weight='bold', **options)
                        nx.draw(G1, with_labels=False, node_color="#D2B4DE", cmap=plt.cm.tab10, font_weight='bold', **options)
                        nx.draw(G2, with_labels=False, node_color="#D35400", cmap=plt.cm.tab10, font_weight='bold', **options)
                        nx.draw(G3, with_labels=False, node_color="#EDBB99", cmap=plt.cm.tab10, font_weight='bold', **options)
                        plt.legend(["Correctly Predicted as 0", "", "Correctly Predicted as 1", "", "Wrongly predicted as 0", "",
                                    "Wrongly predicted as 1"])  # Empty cos of the link?
                        # plt.savefig(save_pics_filename)
                        plt.show()


                    # plot_graph_result(correct_edge_index, all_correct_classified_nodes_as0, all_correct_classified_nodes_as1, incorrect_edge_index, all_incorrect_classified_nodes_as0, all_incorrect_classified_nodes_as1)

                    # # plot_graph_result(incorrect_edge_index, all_incorrect_classified_nodes, color="pink")

                    print("==================== End Quick analysis===============================")

                    # from sklearn.svm import SVC
                    # from sklearn.ensemble import VotingClassifier
                    # from sklearn.linear_model import LogisticRegression
                    # from sklearn.tree import DecisionTreeClassifier
                    #
                    # # Ensemble method using Logistic regression & Decision trees
                    # lr_clf = LogisticRegression(random_state=0)
                    #
                    # dec_clf = DecisionTreeClassifier()
                    #
                    # voting_clf2 = VotingClassifier(
                    #     estimators=[('lr', lr_clf), ('decision', dec_clf)],
                    #     voting='hard')
                    # voting_clf2.fit(X_attack_InOut, y_attack_InOut) # for shadow
                    #
                    # # performance
                    # print("Ensemble accuracy performance! Interested", voting_clf2.score(X_InOutTrain, y_InOutTrain)) # for target


print("target_train_loss_acc_per_run", target_train_loss_acc_per_run)
print("target_approx_train_acc_per_run", target_approx_train_acc_per_run)
print("target_train_acc_per_run", target_train_acc_per_run)
print("target_test_acc_per_run", target_test_acc_per_run)
print("target_macro_acc_per_run", target_macro_acc_per_run)
print("target_micro_acc_per_run", target_micro_acc_per_run)

print("shadow_train_loss_acc_per_run", shadow_train_loss_acc_per_run)
print("shadow_approx_train_acc_per_run", shadow_approx_train_acc_per_run)
print("shadow_train_acc_per_run", shadow_train_acc_per_run)
print("shadow_test_acc_per_run", shadow_test_acc_per_run)
print("shadow_macro_acc_per_run", shadow_macro_acc_per_run)
print("shadow_micro_acc_per_run", shadow_micro_acc_per_run)

print("precision_per_run", precision_per_run)
print("auroc_per_run", auroc_per_run)
print("recall_per_run", recall_per_run)
print("f1_score_per_run", f1_score_per_run)
print("total_time_per_run", total_time_per_run)



target_train_loss_acc_per_run_mean = statistics.mean(target_train_loss_acc_per_run)
target_approx_train_acc_per_run_mean = statistics.mean(target_approx_train_acc_per_run)
target_train_acc_per_run_mean = statistics.mean(target_train_acc_per_run)
target_test_acc_per_run_mean = statistics.mean(target_test_acc_per_run)
target_macro_acc_per_run_mean = statistics.mean(target_macro_acc_per_run)
target_micro_acc_per_run_mean = statistics.mean(target_micro_acc_per_run)


shadow_train_loss_acc_per_run_mean = statistics.mean(shadow_train_loss_acc_per_run)
shadow_approx_train_acc_per_run_mean = statistics.mean(shadow_approx_train_acc_per_run)
shadow_train_acc_per_run_mean = statistics.mean(shadow_train_acc_per_run)
shadow_test_acc_per_run_mean = statistics.mean(shadow_test_acc_per_run)
shadow_macro_acc_per_run_mean = statistics.mean(shadow_macro_acc_per_run)
shadow_micro_acc_per_run_mean = statistics.mean(shadow_micro_acc_per_run)

precision_per_run_mean = statistics.mean(precision_per_run)
auroc_per_run_mean = statistics.mean(auroc_per_run)
recall_per_run_mean = statistics.mean(recall_per_run)
f1_score_per_run_mean = statistics.mean(f1_score_per_run)

total_time_per_run_mean = statistics.mean(total_time_per_run)




target_train_loss_acc_per_run_stdev = statistics.stdev(target_train_loss_acc_per_run)
target_approx_train_acc_per_run_stdev = statistics.stdev(target_approx_train_acc_per_run)
target_train_acc_per_run_stdev = statistics.stdev(target_train_acc_per_run)
target_test_acc_per_run_stdev = statistics.stdev(target_test_acc_per_run)
target_macro_acc_per_run_stdev = statistics.stdev(target_macro_acc_per_run)
target_micro_acc_per_run_stdev = statistics.stdev(target_micro_acc_per_run)


shadow_train_loss_acc_per_run_stdev = statistics.stdev(shadow_train_loss_acc_per_run)
shadow_approx_train_acc_per_run_stdev = statistics.stdev(shadow_approx_train_acc_per_run)
shadow_train_acc_per_run_stdev = statistics.stdev(shadow_train_acc_per_run)
shadow_test_acc_per_run_stdev = statistics.stdev(shadow_test_acc_per_run)
shadow_macro_acc_per_run_stdev = statistics.stdev(shadow_macro_acc_per_run)
shadow_micro_acc_per_run_stdev = statistics.stdev(shadow_micro_acc_per_run)

precision_per_run_stdev = statistics.stdev(precision_per_run)
auroc_per_run_stdev = statistics.stdev(auroc_per_run)
recall_per_run_stdev = statistics.stdev(recall_per_run)
f1_score_per_run_stdev = statistics.stdev(f1_score_per_run)

total_time_per_run_stdev = statistics.stdev(total_time_per_run)


result_file_average.write("Data: "+relax_target_data_type+ " TargetModel: "+model_type+"-"+ str(target_num_neurons)+
                          " Epochs: " + str(model_training_epoch-1) +" Loss: "+str(round(target_train_loss_acc_per_run_mean, 4))+
                          " Approx Train: "+ str(target_approx_train_acc_per_run_mean)+ " std: "+str(round(target_approx_train_acc_per_run_stdev, 4))+
                          "\n|| Train Acc: "+str(target_train_acc_per_run_mean)+ " std: "+ str(round(target_train_acc_per_run_stdev, 4))+
                          " Test Acc: "+str(target_test_acc_per_run_mean) + " std: "+str(round(target_test_acc_per_run_mean,4))+
                          " Macro: "+str(target_macro_acc_per_run_mean)+ " Micro: "+str(target_micro_acc_per_run_mean)  + "\n")


result_file_average.write("Data: "+data_type+ " ShadowModel: "+shadow_model_type+"-"+ str(shadow_num_neurons)+
                          " Epochs: " + str(shadow_model_training_epoch-1) +" Loss: "+str(round(shadow_train_loss_acc_per_run_mean, 4))+
                          " Approx Train: "+ str(shadow_approx_train_acc_per_run_mean)+ " std: "+str(round(shadow_approx_train_acc_per_run_stdev, 4))+
                          "\n|| Train Acc: "+str(shadow_train_acc_per_run_mean)+ " std: "+ str(round(shadow_train_acc_per_run_stdev, 4))+
                          " Test Acc :"+str(shadow_test_acc_per_run_mean) + " std: "+str(round(shadow_test_acc_per_run_mean, 4))+
                          " Macro"+str(shadow_macro_acc_per_run_mean)+ " Micro: "+str(shadow_micro_acc_per_run_mean)  + "\n")


result_file_average.write("|| Precision: "+ str(precision_per_run_mean)+" std: " +str(round(precision_per_run_stdev, 4)) +
                          " || Recall: "+ str(recall_per_run_mean) + " std: "+str(round(recall_per_run_stdev, 4))+
                          " || AUROC: "+ str(auroc_per_run_mean) + " std: "+ str(round(auroc_per_run_stdev, 4))+
                          " || F1 score: "+ str(f1_score_per_run_mean)+" std: "+str(round(f1_score_per_run_stdev, 4))+
                          "\n========== Model: " + model_type + " || Time: " + str(round(total_time_per_run_mean, 4)) + " =========== \n\n\n")


sys.stdout = old_stdout

log_file.close()
result_file_average.close()
