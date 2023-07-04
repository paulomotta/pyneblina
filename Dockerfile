FROM neblina-core:1.1.1a

LABEL maintainer="prmottajr@gmail.com"
LABEL version="1.1.1"
LABEL description="This is custom Docker Image for the pyneblina library."

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN apt install -y python3 python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean

RUN pip3 install pytest

RUN cd /tmp && \
    git clone https://github.com/paulomotta/pyneblina && \
    cd pyneblina && \
    python3 setup.py install

COPY setup.py /tmp/pyneblina/setup.py
COPY test.py /tmp/pyneblina/test.py
COPY neblina_wrapper.c /tmp/pyneblina/neblina_wrapper.c

RUN cd /tmp/pyneblina && \
    python3 setup.py install
 
# COPY walk.py /tmp/pyneblina/walk.py

# RUN pip3 install hiperwalk

VOLUME ["/src"]