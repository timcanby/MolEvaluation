from rdkit import rdBase, Chem
import scipy
print(rdBase.rdkitVersion)
import pandas as pd
import os
import sys
import hyperparameters
from rdkit.Chem import AllChem, Descriptors
import torch
import torch.nn.functional as F
import torch.utils.data
from metrics import metrics as  moses_metrics
import argparse
import json
import collections
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit.Chem import QED
import deepchem as dc
import ast
from sklearn.metrics import silhouette_score
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.metrics.pairwise import pairwise_distances
import warnings
class DistanceMethods(object):
    def __init__(self,method):
        # loss_record_list for record result
        self.record_list = []
        #self.method=method

        if method== "Tanimoto":
            self.metric_selected = DataStructs.TanimotoSimilarity
        elif method== "Dice":
            self.metric_selected = DataStructs.DiceSimilarity
        elif method== "Cosine":
            self.metric_selected = DataStructs.CosineSimilarity
        elif method== "Sokal":
            self.metric_selected = DataStructs.SokalSimilarity
        if method== "Russel":
            self.metric_selected = DataStructs.RusselSimilarity
        if method== "Kulczynski":
            self.metric_selected = DataStructs.KulczynskiSimilarity
        # If an exact match is not confirmed, this last case will be used if provided
        else:
            self.metric_selected = DataStructs.TanimotoSimilarity

    def mol_similarity_calc(self,smi1, smi2):
        mol1 = Chem.MolFromSmiles(smi1[0])
        mol2 = Chem.MolFromSmiles(smi2[0])
        fp1 = AllChem.RDKFingerprint(mol1)
        fp2 = AllChem.RDKFingerprint(mol2)
        similarity = DataStructs.FingerprintSimilarity(fp1, fp2, metric=self.metric_selected )

        return similarity

    def sample_and_calc_distance(self,gen,test, n_sample):
        new_list_gen = np.array(random.sample(gen, n_sample))
        new_list_test= np.array(random.sample(test, n_sample))
        sample_list = np.append(np.zeros_like(new_list_gen), np.ones_like(new_list_test))
        data_test_gen = np.append(new_list_gen, new_list_test).reshape(-1, 1)
        data_gen_gen = np.append(new_list_gen, new_list_gen).reshape(-1, 1)
        distance_matrix_gen_test= scipy.spatial.distance.cdist(new_list_gen.reshape(-1, 1), new_list_test.reshape(-1, 1),
                                                        metric=self.mol_similarity_calc)
        distance_matrix1_gen_gen = scipy.spatial.distance.cdist(new_list_gen.reshape(-1, 1), new_list_gen.reshape(-1, 1),
                                                        metric=self.mol_similarity_calc)
        distance_matrix_test_test = scipy.spatial.distance.cdist(new_list_test.reshape(-1, 1), new_list_test.reshape(-1, 1),
                                                        metric=self.mol_similarity_calc)
        distance_matrix_gen_test_pdist = scipy.spatial.distance.cdist(data_test_gen.reshape(-1, 1), data_test_gen.reshape(-1, 1),
                                                        metric=self.mol_similarity_calc)
        distance_matrix_gen_gen_pdist = scipy.spatial.distance.cdist(data_gen_gen.reshape(-1, 1), data_gen_gen.reshape(-1, 1),
                                                        metric=self.mol_similarity_calc)

        np.fill_diagonal(distance_matrix_gen_test_pdist, 0)
        np.fill_diagonal(distance_matrix_gen_gen_pdist , 0)
        from sklearn.metrics import silhouette_score
        silhouette_score_gen_test=silhouette_score(distance_matrix_gen_test_pdist, sample_list, metric="precomputed")
        silhouette_score_gen_gen = silhouette_score(distance_matrix_gen_gen_pdist, sample_list, metric="precomputed")

        return distance_matrix_gen_test,distance_matrix1_gen_gen,distance_matrix_test_test,silhouette_score_gen_test,silhouette_score_gen_gen

if __name__ == "__main__":
    warnings.simplefilter('ignore')
    test = ['Oc1ccccc1-c1cccc2cnccc12', 'COc1cccc(NC(=O)Cc2coc3ccc(OC)cc23)c1']
    test_sf = ['COCc1nnc(NC(=O)COc2ccc(C(C)(C)C)cc2)s1',
               'O=C(C1CC2C=CC1C2)N1CCOc2ccccc21',
               'Nc1c(Br)cccc1C(=O)Nc1ccncn1']
    gen = ['CNC', 'Oc1ccccc1-c1cccc2cnccc12',
           'CCCP',
           'Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1',
           'Cc1nc(NCc2ccccc2)no1-c1ccccc1', 'Oc1ccccc1-c1cccc2cnccc12']

    distance_method=DistanceMethods(method='Dice')
    Dice_result=distance_method.sample_and_calc_distance(gen=gen,test=test, n_sample=2)
    print(f'Distance(gen*test): {np.mean(Dice_result[0])}')
    print(f'Distance(gen*gen): {np.mean(Dice_result[1])}')
    print(f'Distance(test*test): {np.mean(Dice_result[2])}')
    print(f'Silhouette Score(n=2)(gen*gen): {Dice_result[4]}')
    print(f'Silhouette Score(n=2)(gen*test): {Dice_result[3]}')
    dic=moses_metrics.get_all_metrics(gen=gen,test=test,k=100)
    for k, v in dic.items():
        print(k, v)
    #for each_item in moses_metrics.get_all_metrics(gen=gen,test=test):
        #print(each_item)
    #print(distance_method.sample_and_calc_distance(gen=gen,test=test, n_sample=2))
    # print(metrics.get_all_metrics(train_smile, generate,k=2088))
    #print(moses_metrics.get_all_metrics(gen=gen,test=test))