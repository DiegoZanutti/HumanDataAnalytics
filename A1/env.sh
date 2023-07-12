#!/bin/bash

export DOCKER_IMAGE_NAME=HDA-integration
export DOCKER_CONTAINER_NAME=container-$(DOCKER_IMAGE_NAME)
export HDA_INTEGRATION_PATH_IN_CONTAINER=/usr/HDA
export ENTRYPOINT_PATH=$(HDA_INTEGRATION_PATH_IN_CONTAINER)/src/main.py