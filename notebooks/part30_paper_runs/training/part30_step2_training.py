import sys
from architecture.session import Session, egeofno_ver1, egeofno_ver2, \
    egeofno_ver3, egeofno_ver4, svdfno_ver1, svdfno_ver2, svdfno_ver3, svdfno_ver4

######### FIX MATRIX PLOTTING TAMALE #####
N_train = 20001

save_dir = "/cephyr/users/stromem/Alvis/deep-micro-slip-model/data/article_training/"
dash_dir = "/cephyr/users/stromem/Alvis/deep-micro-slip-model/data/dashboard/"

save_name = "svdfno_ver4"
net = svdfno_ver4()
session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir)
session.train_nsteps(N_train)


save_name = "svdfno_ver3"
net = svdfno_ver3()
session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir)
session.train_nsteps(N_train)


save_name = "svdfno_ver1"
net = svdfno_ver1()
session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir)
session.train_nsteps(N_train)

save_name = "svdfno_ver2"
net = svdfno_ver2()
session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir)
session.train_nsteps(N_train)


######### FNO

save_name = "fno_ver1"
net = egeofno_ver1()
session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir)
session.train_nsteps(N_train)

save_name = "fno_ver2"
net = egeofno_ver2()
session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir)
session.train_nsteps(N_train)

save_name = "fno_ver3"
net = egeofno_ver3()
session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir)
session.train_nsteps(N_train)

save_name = "fno_ver4"
net = egeofno_ver4()
session = Session(net, save_name=save_name, save_dir=save_dir, dash_dir=dash_dir)
session.train_nsteps(N_train)


