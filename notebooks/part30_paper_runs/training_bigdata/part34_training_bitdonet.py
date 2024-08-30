import sys
from architecture.session_bigdata import Session, fno_ver1, fno_ver2, fno_ver3, fno_ver4
from architecture.bitdonet import BITDOnet

######### FIX MATRIX PLOTTING TAMALE #####
N_train = 1000
#data_dir = "/home/emastr/deep-micro-slip-model/data/micro_geometries_boundcurv_repar_256_torch/data_big_clean.torch"
data_dir = "/home/emastr/deep-micro-slip-model/data/micro_geometries_boundcurv_repar_256_torch_high_variance/"
#data_dir = "/mnt/data0/emastr/geometries_torch/variance_norepar_512/"
#save_dir = "/mnt/data0/emastr/article_training_nodecay/"
save_dir = "/mnt/data0/emastr/training/article_training_norepar/"
dash_dir = "/home/emastr/deep-micro-slip-model/data/dashboard/bitdonet/"


#device="cuda:1"
for seed in [0, 1, 2, 3, 4, 5]:
    
    save_name = f"bitdonet_ver2_bigbatch_epochlr"
    device = "cuda:0"
    net = BITDOnet(modes = 41, dtype="double", device=device)
    print(net.layer_widths)
    session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir, device=device, path_data=data_dir, lr_schedule=True)
    session.train_nsteps(N_train)
    #print(f"Seed {seed}, model ver4")
    #net = bitdonet(device=device) 
    #session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir + "v4", device=device, path_data=data_dir)
    #session.train_nsteps(N_train)