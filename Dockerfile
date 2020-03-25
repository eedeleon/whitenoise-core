FROM rust:1.42

# FROM debian:buster-slim
# RUN apt-get update && apt-get install -y extra-runtime-dependencies
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y python3-pip

WORKDIR /usr/src/whitenoise-core

COPY ./prototypes ./prototypes
COPY ./validator-rust ./validator-rust
COPY ./runtime-rust ./runtime-rust
COPY ./bindings-python ./bindings-python


#WORKDIR /usr/src/whitenoise-core/validator-rust
#RUN "cargo build"

WORKDIR /usr/src/whitenoise-core/runtime-rust
RUN cargo build

#RUN cargo install --path .

RUN "cargo build"

ENTRYPOINT ["tail", "-f", "/dev/null"]
