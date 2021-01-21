# Base Image
FROM python:3.6-buster

# Update Package Manager
RUN set -x && \
    apt update && \
    apt upgrade -y

# Install WGet, Git
RUN set -x && \
    apt install -y git

COPY ./ GunosyClassifier/

RUN cd GunosyClassifier && \
    pip install -r ./docker/requirement.txt && \
    pip install transformers["ja"]