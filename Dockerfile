FROM jupyter/minimal-notebook

LABEL maintainer="prmottajr@gmail.com"
LABEL version="1.1.1"
LABEL description="This is custom Docker Image for the pyneblina library."

USER root

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN apt install -y git gcc g++ libgtest-dev cmake vim libunwind8 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

# RUN cd /tmp && \
#     git clone https://github.com/paulomotta/neblina-core && \
#     cd neblina-core && \
#     cmake . && \
#     make && \
#     make install && \
#     ldconfig 

COPY neblina-core/src /tmp/neblina-core/src
COPY neblina-core/include /tmp/neblina-core/include
COPY neblina-core/test /tmp/neblina-core/test
COPY neblina-core/CMakeLists.txt /tmp/neblina-core/CMakeLists.txt

RUN cd /tmp && \
    cd neblina-core && \
    cmake . && \
    make && \
    make install && \
    ldconfig 

RUN ln -s /usr/local/lib /usr/local/lib64

# RUN apt install -y python3-pip && \
#     rm -rf /var/lib/apt/lists/* && \
#     apt clean

RUN pip3 install pytest

RUN cd /tmp && \
    git clone https://github.com/paulomotta/pyneblina && \
    cd pyneblina && \
    python3 setup.py install

COPY pyneblina/setup.py /tmp/pyneblina/setup.py
COPY pyneblina/test.py /tmp/pyneblina/test.py
COPY pyneblina/neblina_wrapper.c /tmp/pyneblina/neblina_wrapper.c

RUN cd /tmp/pyneblina && \
    python3 setup.py install
 
COPY pyneblina/walk.py /tmp/pyneblina/walk.py

# RUN pip3 install numpy
# RUN pip3 install scipy
# RUN pip3 install networkx
# RUN pip3 install matplotlib
RUN pip3 install hiperwalk

# RUN cd /tmp && \
#     git clone https://github.com/hiperwalk/hiperwalk.git

RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER ${NB_UID}