#!/bin/bash
# PWD=pwd
echo $PWD

nvidia-docker run -it --name instance-seg -v $PWD:/workspace/ -p 5000:8888 -p 5001:6006 mypytorch /bin/bash
#nvidia-docker run --rm -it --name instance-seg -v $PWD:/workspace/ -p 5000:8888 -p 5001:6006 mypytorch /bin/bash
# pip install graphviz
# pip install torchviz
# pip install jupyter
