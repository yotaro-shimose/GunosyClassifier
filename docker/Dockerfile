# Base Image
FROM python:3.6-buster

# Update Package Manager
RUN set -x && \
    apt update && \
    apt upgrade -y

# Install WGet, Git
RUN set -x && \
    apt install -y git


RUN mkdir /root/.ssh
COPY ./ssh/id_rsa /root/.ssh
COPY ./ssh/id_rsa.pub /root/.ssh
COPY ./ssh/config /root/.ssh
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
# Clone repository
RUN git clone git@github.com:yotaro-shimose/GunosyClassifier.git

# RUN conda activate gunosy && \
RUN cd GunosyClassifier && \
    pip install -r requirement.txt && \
    pip install transformers["ja"]