FROM rust:1.42

# FROM debian:buster-slim
# RUN apt-get update && apt-get install -y extra-runtime-dependencies
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y python3-pip

WORKDIR /usr/src/whitenoise-core

COPY ./validator-rust ./validator-rust
COPY ./prototypes ./prototypes

WORKDIR /usr/src/whitenoise-core/c

#RUN cargo install --path .

RUN "cargo build"

ENTRYPOINT ["tail", "-f", "/dev/null"]
