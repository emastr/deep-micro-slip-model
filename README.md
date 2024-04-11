## Installation

To install the package, we use
  ```pip install -e . build```
This installs the packages in editable mode (and builds it).

To connect with jupyter to remote, run
```jupyter --no-browser --port=8888```
on the server, and then
```ssh -p2225 -f -N -L localhost:8888:localhost:8888 emastr@domain.math.kth.se```

To copy over to Alvis, we use
```rsync -r -v --exclude="data/" /home/emastr/github/deep-micro-slip-model /cephyr/users/stromem/Alvis```.
This will copy over everything except for data. 



