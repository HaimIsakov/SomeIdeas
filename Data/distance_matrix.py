import os

import cyclic_vae_best_params_weight_cdr3 as ae
import numpy as np
import pickle
import torch
from tqdm import tqdm
import csv


def embed_for_encoding(x, emb_dict, device):
    tensors = []
    for tcr_idx in x.view(-1).tolist():
        tensors.append(emb_dict[tcr_idx].to(device).long())
    return torch.stack(tensors)


def embed(tcr, v_gene, amino_pos_to_num, max_length, v_dict, device):
    padding = torch.zeros(1, max_length)
    for i in range(len(tcr)):
        amino = tcr[i]
        pair = (amino, i)
        padding[0][i] = amino_pos_to_num[pair]
    # print("v_gene in v_dict",  v_gene in v_dict)
    # print("v_gene", v_gene)
    # print("padding shape", padding.shape)
    combined = torch.cat((padding.to(device), v_dict[v_gene].to(device)), dim=1)
    return combined


def create_distance_matrix(device, outliers_file="outliers", adj_mat="dist_mat"):
    # if os.path.isfile(f"dist_mat_{run_number}.csv"):
    #     print(f"dist_mat_{run_number}.csv is found")
    #     return
    outliers = pickle.load(open(f"{outliers_file}.pkl", 'rb'))
    print("start loading model")
    ae_dict = torch.load('vae_vgene_ae.pt', map_location=device)
    print("finish loading model")

    model = ae.Model2(ae_dict['max_len'], 10, ae_dict['vgene_dim'], ae_dict['v_dict'], encoding_dim=ae_dict['enc_dim'])
    model.load_state_dict(ae_dict['model_state_dict'])
    embeddings = []
    for tcr, _ in outliers.keys():
        # print("tcr", tcr)
        cdr3 = tcr.split('_')[0]
        vgene = tcr.split('_')[1]
        # print(vgene)
        # print(type(vgene))
        if vgene is "unknown" or vgene not in ae_dict['v_dict']:
            print("vgene is ", vgene)
            continue
        else:
            embeddings.append(embed(cdr3, vgene, ae_dict['amino_pos_to_num'], ae_dict['max_len'], ae_dict['v_dict'], device))
    encodings = []
    model = model.to(device)
    for emb in embeddings:
        with torch.no_grad():
            emb = emb.to(device)
            _, mu, _ = model(emb)
            encodings.append(mu)
    tcrs = [tcr for tcr, _ in outliers.keys() if tcr.split('_')[1] != "unknown" and tcr.split('_')[1] in ae_dict['v_dict']]
    # print(tcrs)
    header = [''] + tcrs
    matrix = [header]
    for i, enc1 in tqdm(enumerate(encodings)):
        line = []
        for enc2 in encodings:
            dist = torch.cdist(enc1, enc2)
            line.append(dist.item())
        line = [tcrs[i]] + line
        matrix.append(line)
    with open(f"{adj_mat}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)


# if __name__ == '__main__':
#     device = "cpu"
#     create_distance_matrix(device)
