#!/bin/bash

# Check if two arguments were provided
if [ $# -ne 2 ]
  then
    echo "Two arguments required: <image type> (e.g. train, inference) and <server type> (e.g. cpu, gpu, triton))"
    exit 1
fi

image_type=$1
server_type=$2

# Validate the arguments
if ! [[ $image_type == "train" || $image_type == "inference" ]]
then
    echo "Invalid image type. Should be either 'train' or 'inference'"
    exit 1
fi

if ! [[ $server_type == "cpu" || $server_type == "gpu" || $server_type == "triton" ]]
then
    echo "Invalid server type. Should be 'cpu', 'gpu', or 'triton'"
    exit 1
fi

# Build the path to the docker-compose file
path="./docker/${image_type}/${server_type}/docker-compose.yml"

# Run the appropriate docker-compose command based on the arguments
docker-compose -f $path up -d --build && docker attach ag-release-tools && docker-compose -f $path down 
