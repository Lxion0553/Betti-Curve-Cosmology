import numpy as np
import os

def cut_data(bcs, cuts=[None, None, None]):

    bcs_out = []
    for cut, bc in zip(cuts, bcs):
        if cut is not None:
            bc = bc[:,cut]
        bcs_out.append(bc)

    return bcs_out

def read_params(file):
    d = np.loadtxt(file,skiprows=1)
    return d

def read_bcs(files, cuts = [[0,7], [1,15], [6,20]], conc=False, params='/hscratch/ljy/ProcessedData/nwLH/latin_hypercube_params_nwLH.txt', num_density=False, normalize_boxsize=None):
    out = {}
    paras = read_params(params)
    for file in files:
        bcs = np.loadtxt(file, comments='#')
        index = file.split('/')[6]
        if num_density:
            with open(file, 'r') as f:
                for line in f:
                    scale_lenth = np.float32(line.split(':')[1].split(' ')[1])
                    break
            key = tuple(paras[int(index)].tolist() + [scale_lenth])
        else:
            key = tuple(paras[int(index)].tolist())
        
        if normalize_boxsize is not None:
            normalize_factor = scale_lenth**6/normalize_boxsize**3
            bcs = [bc*normalize_factor for bc in bcs]

        if cuts is not None:
            if not conc:
                out[key] = [bcs[0][cuts[0]],bcs[1][cuts[1]],bcs[2][cuts[2]]]
            else:
                out[key] = np.concatenate([bcs[0][cuts[0]],bcs[1][cuts[1]],bcs[2][cuts[2]]])
        else:
            if not conc:
                out[key] = [bcs[0],bcs[1],bcs[2]]
            else:
                out[key] = np.concatenate([bcs[0],bcs[1],bcs[2]])
    return out

def write_bcs(f, data):
    """
    format: "label" : "bc0" : "bc1" : "bc2"
    separator: ","
    """
    with open(f, "w") as file:
        for key, value in data.items():
            for i in key[:-1]:
                file.write(f"{i},")
            file.write(f"{key[-1]}:")
            for v in value:
                for i in v[:-1]:
                    file.write(f"{i},")
                file.write(f"{v[-1]}:")
            file.write("\n")
    return 0
