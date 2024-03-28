import os

root = "/cephyr/users/stromem/Alvis/phd/"

#os.mkdir(f'{root}data')
os.makedirs(f'{root}data/article_training/', exist_ok=False)
os.makedirs(f'{root}data/', exist_ok=False)