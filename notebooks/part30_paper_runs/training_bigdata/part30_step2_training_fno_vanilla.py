import sys
from architecture.session_bigdata import Session, fno_ver1, fno_ver2, fno_ver3, fno_ver4, fno_ver5

######### FIX MATRIX PLOTTING TAMALE #####
N_train = 1
#data_dir = "/home/emastr/deep-micro-slip-model/data/micro_geometries_boundcurv_repar_256_torch/data_big_clean.torch"
#data_dir = "/home/emastr/deep-micro-slip-model/data/micro_geometries_boundcurv_unirepar_256_torch_high_variance/"
#data_dir = "/home/emastr/deep-micro-slip-model/data/micro_geometries_boundcurv_unirepar_256_torch_high_variance/"
data_dir = "/mnt/data0/emastr/geometries_torch/no_variance_256/"
#save_dir = "/mnt/data0/emastr/article_training_nodecay/"
#save_dir = "/mnt/data0/emastr/article_training_hugedata/"
save_dir = "/mnt/data0/emastr/article_training_hugedata_novar/"
dash_dir = "/home/emastr/deep-micro-slip-model/data/dashboard/fno_vanilla_huge_novar/"

device="cuda:0"
for seed in [1, 2, 3, 4, 5]:
    
    print(f"Seed {seed}, model ver2")
    save_name = f"fno_vanilla_ver2_seed{seed}"
    net = fno_ver2(device=device) 
    session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir + "v2", device=device, path_data=data_dir)
    session.train_nsteps(N_train)
    
    print(f"Seed {seed}, model ver5")
    save_name = f"fno_vanilla_ver5_seed{seed}"
    net = fno_ver5(device=device) 
    session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir + "v5", device=device, path_data=data_dir)
    session.train_nsteps(N_train)
    
    if False:
        print(f"Seed {seed}, model ver4")
        save_name = f"fno_vanilla_ver4_seed{seed}"
        net = fno_ver4(device=device) 
        session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir + "v4", device=device, path_data=data_dir)
        session.train_nsteps(N_train)
        
        print(f"Seed {seed}, model ver2")
        save_name = f"fno_vanilla_ver2_seed{seed}"
        net = fno_ver2(device=device) 
        session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir + "v2", device=device, path_data=data_dir)
        session.train_nsteps(N_train)
        
        print(f"Seed {seed}, model ver3")    
        save_name = f"fno_vanilla_ver3_seed{seed}"
        net = fno_ver3(device=device) 
        session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir + "v3", device=device, path_data=data_dir)
        session.train_nsteps(N_train)
        
        print(f"Seed {seed}, model ver1")
        save_name = f"fno_vanilla_ver1_seed{seed}"
        net = fno_ver1(device=device) 
        session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir + "v1", device=device, path_data=data_dir)
        session.train_nsteps(N_train)
