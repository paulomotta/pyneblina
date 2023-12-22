# Use the jupyter/minimal-notebook base image
FROM jupyter/minimal-notebook

LABEL maintainer="prmottajr@gmail.com"
LABEL version="1.1.1"
LABEL description="This is custom Docker Image for the pyneblina library."

USER root

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies for CUDA Toolkit 12 and OpenCL headers
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    dirmngr \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

# Add the CUDA Toolkit 12 repository
RUN echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list

# Import the NVIDIA public key using gpg
RUN gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC && \
    gpg --export --armor A4B469963BF863CC | apt-key add -

# Update package lists and install CUDA Toolkit 12
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cuda-12-0 \
    libcudnn8 \
    libcudnn8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure CUDA environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

RUN apt update

RUN apt install -y git gcc g++ libgtest-dev cmake vim libunwind8 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

COPY hiperblas-core/src /tmp/hiperblas-core/src
COPY hiperblas-core/include /tmp/hiperblas-core/include
COPY hiperblas-core/test /tmp/hiperblas-core/test
COPY hiperblas-core/CMakeLists.txt /tmp/hiperblas-core/CMakeLists.txt

RUN cd /tmp && \
    cd hiperblas-core && \
    cmake . && \
    make && \
    make install && \
    ldconfig 

RUN ln -s /usr/local/lib /usr/local/lib64

RUN pip3 install pytest

COPY pyhiperblas/setup.py /tmp/pyhiperblas/setup.py
COPY pyhiperblas/test.py /tmp/pyhiperblas/test.py
COPY pyhiperblas/neblina_wrapper.c /tmp/pyhiperblas/neblina_wrapper.c

RUN cd /tmp/pyhiperblas && \
    python3 setup.py install
 
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install networkx
RUN pip3 install matplotlib
RUN pip3 install hiperwalk

RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER ${NB_UID}