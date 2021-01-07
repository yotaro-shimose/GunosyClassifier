# Base Image
FROM ubuntu:20.04


# Update Package Manager
RUN set -x && \
    apt update && \
    apt upgrade -y

# Install WGet, Git
RUN set -x && \
    apt install -y wget && \
    apt install -y git

# Install Anaconda
RUN set -x && \
    wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh && \
    bash Anaconda3-2020.11-Linux-x86_64.sh -b && \
    rm Anaconda3-2020.11-Linux-x86_64.sh

# Set anaconda path
ENV PATH $PATH:/root/anaconda3/bin

# Create virtual env and initialize conda
RUN conda create -n gunosy python=3.6 anaconda -y

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "gunosy", "/bin/bash", "-c"]

# Clone repository
RUN git clone https://github.com/yotaro-shimose/gunosy.git

# Install dependencies
RUN cd gunosy && \
    pip install -r requirement.txt

