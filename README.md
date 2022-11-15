# CSC591 Virus Propagation Model
> This project implementation is done towards fulfillment of Project 5 of CSC 591 Graph Data Mining

## Research Paper
The implementation is inspired by the paper given in the `research_paper` folder. The paper is [Got the Flu? (or Mumps) Check the Eigenvalue!](https://arxiv.org/pdf/1004.0060v1.pdf)

## Goal
To implement the given problem and objectives stated [here](https://github.com/tusharkini/Virus_propogation_model/blob/main/objective/P4VirusPropagation.pdf). Analyze the propagation of a virus in a network and prevent a network wide epidemic. In order to do that, your team will need to:
- Analyze and understand a virus propagation model.
- Calculate the effective strength of a virus.
- Simulate the propagation of a virus in a network.
- Implement immunization policies to prevent a virus from spreading across a network. 


---
## Data  

You will be provided with the following materials in advance:
- Supplementary material on virus propagation.
- Parameter values for experiments:
    - Transmission probabilities β 1 = 0.20 and β 2 = 0.01.
    - Healing probabilities δ 1 =0.70 and δ 2 = 0.60.
    - Number of available vaccines k 1 = 200.
- Static contact network (i.e., one undirected unweighted graph) for Option 1:
    - static.network

---


## Getting Started

### Installation

- Install Python3 from [here](https://www.python.org/downloads/) and finish the required setup in the executable file.
- Install pip package manager for future downloads-
    ```bash
    $ python -m ensurepip --upgrade
    ```
- Upgrade the version of pip-
    ```
    $ python -m pip install --upgrade pip
    ```
- Install NetworkX for graph processing-
    ```bash
    $ pip install networkx
    ```
- Upgrade the version of networkx-
    ```
    $ pip install --upgrade networkx
    ```

- Create working directory named `Virus_propagation_P5` and go inside it
    ```bash
    $ mkdir Virus_propagation_P5
    $ cd Virus_propagation_P5
    ```
- Clone this repository from [here](https://github.com/tusharkini/Virus_propogation_model) or use the following in GitBash
    ```bash
    $ git clone https://github.com/tusharkini/Virus_propogation_model
    ```
### Running the Algorithm Code
- Run the algorithm code using- 
    ```bash
    $ python main.py 
    ```
    This will create a series of images in the `results` folder. These plot names are self explanatory in nature.

## Authors

- Tushar Kini [Github](https://github.com/tusharkini)

---

## Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)


-----
