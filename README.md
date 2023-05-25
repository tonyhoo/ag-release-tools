# ag-release-tools

This repository contains Docker configurations and source code for testing AutoGluon Docker image releases. It allows for easy setup and testing of different AutoGluon images across various environments, including CPU, GPU, and Triton servers.

## Repository Structure

The repository is structured as follows:
```
src/  # where all testing codes hosted and will be run in the docker container
docker/ # definition of docker images based on testing needs
│ 
│
├── train/
│ ├── cpu/
│ │ ├── Dockerfile
│ │ └── docker-compose.yml
│ ├── gpu/
│ │ ├── Dockerfile
│ │ └── docker-compose.yml
│ └── triton/
│ ├── Dockerfile
│ └── docker-compose.yml
│
└── inference/
├── cpu/
│ ├── Dockerfile
│ └── docker-compose.yml
├── gpu/
│ ├── Dockerfile
│ └── docker-compose.yml
└── triton/
├── Dockerfile
└── docker-compose.yml
```



The `src/` directory contains the source code and test scripts for the project.

Each subdirectory under `docker` which has 'train' and 'inference' corresponds to a specific server type (CPU, GPU, or Triton), and contains a Dockerfile for creating the Docker image and a docker-compose file for running the image.

## How to Use

To run a specific Docker image, use the provided bash script `run_docker.sh`:

```bash
./run_docker.sh <image_type> <server_type>
```

Where:

<image_type> should be either 'train' or 'inference'.
<server_type> should be either 'cpu', 'gpu', or 'triton'.
For example, to test the training image on a CPU server, you would run:

```
./run_docker.sh infernece triton
```


This command will look for the corresponding docker-compose.yml file in the 'train/cpu' directory and run docker-compose up.

Make sure the script is executable:

```
chmod +x run_docker.sh
```

## Requirements
You need Docker and Docker Compose installed on your machine to use this repository. For installation instructions, see:

* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)