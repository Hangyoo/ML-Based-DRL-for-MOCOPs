# An Effective Cross-Task Unified Multi-Objective Neural Meta-Framework for Solving Diverse Combinatorial Problems

## Code to be uploaded... The related code will be provided later.

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
    
### Test Data
We provide two types of test data:
1.  Benchmark Test Set
2.  Random Test Set

|                |Dataset                          |Link                         |
|----------------|-------------------------------|-----------------------------|
|MOTSP           |`TSPLib`            |https://eden.dei.uc.pt/~paquete/mtsp/ ; https://eden.dei.uc.pt/~paquete/tsp/          |
|MOCVRP          |`CVRPLib`            |http://vrp.galgos.inf.puc-rio.br/index.php/en/            |
|MOOVRTW         |`Solomon`|https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/|
|MOKP            |`Benchmark instances used in [1]`            |https://www.lamsade.dauphine.fr/~vdp/tv19           |
|MOOVRP          |`C1-C8, F10-F12`|[here](https://github.com/Hangyoo/ML-Based-DRL-for-MOCOPs/tree/main/Test%20Data/Benchmark%20Test%20Set/MOOVRP)|
|Feature Selection  |`UCI`|[here](https://archive.ics.uci.edu/)|
>[1] _Bergman, D., Bodur, M., Cardonha, C., Cire, A.A. (2022) Network Models for Multiobjective Discrete Optimization. INFORMS Journal on Computing 34(2):990-1005_

# Experimental Results

We report the results obtained by our algorithm on all test instances of each MOCO problem.

## Comparison Algorithms
###  Learning-based comparison algorithms
We have selected the five best learning-based algorithms for solving MOCOPs to date as comparison algorithms, as detailed below:
|    Index            |Dataset            |Year              |Link                         |
|----------------|-------------------------------|-----------------------------|------------|
|1           |`DRL-MOA` |     2021      |[here](https://ieeexplore.ieee.org/document/9040280)          |
|2           |`PMOCO`   |     2022      |[here](https://openreview.net/forum?id=QuObT9BTWo)            |
|3           |`AM-MOCO`|      2019      |[here]()|
|4           |`ML-AM` |      2022      |[here](https://ieeexplore.ieee.org/document/9714721)        |
|5           |`EMNH`|    2023 | [here](https://openreview.net/forum?id=593fc38lhN&referrer=%5Bthe%20profile%20of%20Zizhen%20Zhang%5D(%2Fprofile%3Fid%3D~Zizhen_Zhang1))       |

### Exact and heuristic comparison algorithms
For each MOCOP, we have selected the best 2-3 algorithms (including iteration-based heuristics and exact algorithms) that have addressed the corresponding MOCOP to date as comparison algorithms, as detailed below:
|    Index     |MOTSP  |MOCVRP  |MOVRPTW    |MOKP   |
|--------|----------|------|------|--|
|1       |[2] `PDA(2017)`    |[4] `CoEA-DAE(2023)` |[6] `CCMO(2020)`  |[9] `MOFPA(2018)`|
|2       |[3] `HLS-EA(2021)` |[5] `IMOLEM(2021)` |[7] `INSGAII(2021)`  |[10] `Direct(2021)`|
|3       |——		   | —— |[8] `M-MOEA/D(2016)`  |[11] `FHCo(2022)`|

> MOSTP
>>[2] _Cornu, M., Cazenave, T., & Vanderpooten, D. (2017). Perturbed decomposition algorithm applied to the multi-objective traveling salesman problem. _Computers & Operations Research_, _79_, 314-330._

>>[3] _Agrawal, A., Ghune, N., Prakash, S., & Ramteke, M. (2021). Evolutionary algorithm hybridized with local search and intelligent seeding for solving multi-objective Euclidian TSP. _Expert Systems with Applications_, _181_, 115192._

>MOCVRP
>>[4] _Xiao, J., Zhang, T., Du, J., & Zhang, X. (2019). An evolutionary multiobjective route grouping-based heuristic algorithm for large-scale capacitated vehicle routing problems. _IEEE transactions on cybernetics_, _51_(8), 4173-4186._

>>[5] _Niu, Y., Kong, D., Wen, R., Cao, Z., & Xiao, J. (2021). An improved learnable evolution model for solving multi-objective vehicle routing problem with stochastic demand. _Knowledge-Based Systems_, _230_, 107378_.

>MOVRPTW
>>[6] _Tian, Y., Zhang, T., Xiao, J., Zhang, X., & Jin, Y. (2020). A coevolutionary framework for constrained multiobjective optimization problems. _IEEE Transactions on Evolutionary Computation_, _25_(1), 102-116._

>>[7] _Srivastava, G., Singh, A., & Mallipeddi, R. (2021). NSGA-II with objective-specific variation operators for multiobjective vehicle routing problem with time windows. _Expert Systems with Applications_, _176_, 114779._

>>[8] _Qi, Y., Hou, Z., Li, H., Huang, J., & Li, X. (2015). A decomposition based memetic algorithm for multi-objective vehicle routing problem with time windows. _Computers & Operations Research_, _62_, _61-77_.

>MOKP
>>[9] _Zouache, D., Moussaoui, A., & Abdelaziz, F. B. (2018). A cooperative swarm intelligence algorithm for multi-objective discrete optimization with application to the knapsack problem. _European Journal of Operational Research_, _264(1)_, 74-88_.

>>[10] _Tamby, S., & Vanderpooten, D. (2021). Enumeration of the nondominated set of multiobjective discrete optimization problems. _INFORMS Journal on Computing_, _33_(1), 72-85._

>>[11] _Sahinkoc, H. M., & Bilge, Ü. (2022). A reference set based many-objective co-evolutionary algorithm with an application to the knapsack problem. _European Journal of Operational Research_, _300_(2), 405-417._


# Contacts
If you have any questions, feel free to contact me at:  
[hangyulou1994@gmail.com](mailto:hangyulou1994@gmail.com) 
