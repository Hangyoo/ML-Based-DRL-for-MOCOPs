# An Effective Cross-Task Unified Multi-Objective Neural Combinatorial Optimization Meta-Framework for Solving Diverse Combinatorial Problems

This repository contains the training and testing data for five Multi-Objective Combinatorial Optimization (MOCO) problems, along with the experimental results and corresponding true Pareto fronts (where available).

- Multi-Objective Travelling Salesman Problem (MOTSP)
- Multi-Objective Vehicle Routing Problem with Time Windows (MOVRPTW)
- Multi-Objective Capacitated Vehicle Routing Problem (MOCVRP)
- Multi-Objective Knapsack Problem (MOKP)
- Multi-Objective Open Vehicle Routing Problem (MOOVRP)

# Directory Structure
### Training Data
Due to the large size of the training data (>1GB) and GitHub's file size limitations, we have uploaded the training data to BaiduNetdisk. The data is divided into four compressed files: `MOTSP_training_data`, `MOCVRP_training_data`, `MOVRPTW_training_data`, and `MOKP_training_data`. Note: Our model is trained only on the `MOTSP`, `MOCVRP`, `MOVRPTW`, and `MOKP` datasets, and it generalizes to solve the `MOOVRP` using a zero-shot approach. Therefore, the `MOOVRP` is not included in the training data but is part of the test set. You can download the training data from the following link:

[Download from BaiduNetdisk](https://pan.baidu.com/s/1NXqK6oSfnDpNCpzBLHDlGA?pwd=rudd)
Access Code: `rudd`

To make it easier for users to understand the format of the training data, we have also uploaded example files, each containing 20 training instances (Note: All instance data is normalized to the range [0, 1] to facilitate training):
-   **MOTSPP_training_data_example.txt**:  
    > Contains 4 columns: the first two are used to compute the first objective, and the last two are for the second objective.
    
-   **MOCVRP_training_data_example.txt**:  
    > Contains 3 columns: the first two represent city coordinates, and the third represents demand. The vehicle capacity is set to 1.
    
-   **MOVRPTWP_training_data_example.txt**:  
    > Contains 6 columns: the first two represent city coordinates, and the other four represent demand, earliest service time, latest service time, and service duration, respectively. The vehicle capacity is set to 1.0.
    
-   **MOKP_training_data_example.txt**:  
    > Contains 3 columns: the first column represents item weight, and the second and third columns correspond to the two values of the item. The vehicle capacity is set to 1, and the knapsack capacity is set to 1.0.
    
### Training Data
We provide two types of test data:
1.  Benchmark Test Set
2.  Random Test Set

|                |Dataset                          |Link                         |
|----------------|-------------------------------|-----------------------------|
|MOTSP           |`TSPLib`            |https://eden.dei.uc.pt/~paquete/mtsp/          |
|MOCVRP          |`CVRPLib`            |http://vrp.galgos.inf.puc-rio.br/index.php/en/            |
|MOOVRTW         |`Solomon`|https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/|
|MOKP            |`Benchmark instances used in [1]`            |https://www.lamsade.dauphine.fr/~vdp/tv19           |
|MOOVRP          |`C1-C8, F10-F12`|-- is en-dash, --- is em-dash|
>[1] _Bergman, D., Bodur, M., Cardonha, C., Cire, A.A. (2022) Network Models for Multiobjective Discrete Optimization. INFORMS Journal on Computing 34(2):990-1005_

# Experimental Results

We report the results obtained by our algorithm on all test instances of each MOCO problem.

# Contacts

If you have any questions, feel free to contact me at:  
[hangyulou1994@gmail.com](mailto:hangyulou1994@gmail.com) or [louhangyu@stumail.neu.edu.cn](mailto:louhangyu@stumail.neu.edu.cn)
