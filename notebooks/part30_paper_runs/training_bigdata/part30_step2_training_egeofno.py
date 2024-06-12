import sys
from architecture.session_bigdata import Session, egeofno_ver1, egeofno_ver2, \
    egeofno_ver3, egeofno_ver4, svdfno_ver1, svdfno_ver2, svdfno_ver3, svdfno_ver4

######### FIX MATRIX PLOTTING TAMALE #####
N_train = 1
#data_dir = "/home/emastr/deep-micro-slip-model/data/micro_geometries_boundcurv_repar_256_torch/data_big_clean.torch"
data_dir = "/home/emastr/deep-micro-slip-model/data/micro_geometries_boundcurv_repar_256_torch_high_variance/"
#save_dir = "/mnt/data0/emastr/article_training_nodecay/"
save_dir = "/mnt/data0/emastr/article_training_hugedata/"
dash_dir = "/home/emastr/deep-micro-slip-model/data/dashboard/egeofno_huge/"

device = "cuda:1"
for seed in [0, 1, 2, 3, 4, 5]:
    save_name = f"fno_ver1_seed{seed}"
    net = egeofno_ver1(device=device)
    session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir, device=device, path_data=data_dir)
    session.train_nsteps(N_train)
    

    save_name = f"fno_ver2_seed{seed}"
    net = egeofno_ver2(device=device)
    session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir, device=device, path_data=data_dir)
    session.train_nsteps(N_train)
    
    
    save_name = f"fno_ver3_seed{seed}"
    net = egeofno_ver3(device=device)
    session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir, device=device, path_data=data_dir)
    session.train_nsteps(N_train)
    
    
    save_name = f"fno_ver4_seed{seed}"
    net = egeofno_ver4(device=device)
    session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir, device=device, path_data=data_dir)
    session.train_nsteps(N_train)