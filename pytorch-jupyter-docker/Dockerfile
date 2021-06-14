FROM pytorch/pytorch

# PyKDL
# RUN dpkg -i python3-pykdl_1.4.0-7ubuntu1_amd64.deb

# visualization packages for pytorch
RUN pip install graphviz
RUN pip install torchviz
# visualization support library 
RUN apt-get update && apt-get -y install python-pydot python-pydot-ng graphviz

# jupyter installation 
RUN pip install jupyter
RUN pip install lxml

# pytorch vision repository 
RUN pip install torchvision

# lxml for parsing urdf file
RUN pip install lxml
