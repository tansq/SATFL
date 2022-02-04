def load_dataset_files(dataset='nist'):

    if dataset == 'nist':

        base_path = ''  
        nist = open(base_path+'nist.txt','r').read().splitlines()
        train_files = nist[160:]#
        valid_files = nist[:160]

    return train_files,valid_files