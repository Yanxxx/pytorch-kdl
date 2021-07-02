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

# tensor board and opencv python
RUN pip install tensorboard
RUN pip install opencv-python

RUN apt-get update && apt-get install -y python3-opencv
RUN apt install -y libgl1-mesa-glx

# matplotlib module
RUN python -m pip install -U matplotlib
