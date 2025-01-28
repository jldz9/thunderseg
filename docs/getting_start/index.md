# Installation

Setting up a software environment is a painful experience. Thunderseg provides several methods to get the environment ready.

The installation process will take roughly 5-10 minutes and may vary depends on the network condition and computer resource.

## Via Conda
<details>
  <summary>Don't know how to install Conda?</summary>
  <p> Check <a href="https://docs.anaconda.com/miniconda/install/">Miniconda install guide</a></p>
    <p>OR </p>
  Simply do 
  ```py
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
  ```
</details>

```bash
conda env create -f https://raw.githubusercontent.com/jldz9/thunderseg/refs/heads/master/environment.yml
pip install thunderseg
```

## Via Container
<details>
  <summary>Don't know how to install Docker?</summary>
  <p> Check <a href="https://docs.docker.com/engine/install/">Docker install guide</a></p>
</details>

??? Warning
     If you want to activate<b> GPU</b> processing while using docker, using NVIDIA GPUs with [CUDA](https://developer.nvidia.com/cuda-gpus) support is necessary. 

     Additionally, you will need to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 
### > Use dockerfile

```bash
wget https://raw.githubusercontent.com/jldz9/thunderseg/refs/heads/master/.devcontainer/Dockerfile 
docker build -t thunderseg:1.0.0.dev25 .
```
After running above code block, you should be able to find image in docker desktop image tab 
![docker image](img/docker_img_example_dark.png#only-dark)
![docker image](img/docker_img_example_light.png#only-light)
or in command line: 
```bash
user:~$ docker images
REPOSITORY   TAG       IMAGE ID       CREATED             SIZE
thunderseg   1.0.0.dev25     ba968b128eda   About an hour ago   20.5GB
```

### > Use image from Docker Hub

```bash
docker pull jldz9/thunderseg:1.0.0.dev25
```

### > Use apptainer
<details>
  <summary>Don't know how to install Apptainer?</summary>
  <p> Check <a href="https://apptainer.org/docs/user/latest/quick_start.html#installation">Apptainer install guide</a></p>
</details>
Apptainer is a docker alternative in order to run images on HPC environment, normally you don't need to install it locally.

You can load apptainer by using `module load apptainer` under HPC environment that runs SLURM workload manager

*[HPC]: High-performance computing
*[SLURM]: Simple Linux Utility for Resource Management

To pull image from docker hub using apptainer: 

```bash
apptainer pull thunderseg_100dev.sif docker://jldz9/thunderseg:1.0.0.dev25
```

## Via Source code
If you would like to contribute to this project, thanks in advance! :smile:

You can pull the source code from GitHub by:
```bash
git clone https://github.com/jldz9/thunderseg.git
```
and then set up the environment by: 
```bash
conda env create -f thunderseg/environment.yml
```
we also provide devcontainer.json for vscode under .devcontainer directory

