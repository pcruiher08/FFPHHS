# The Firefighter Problem Hyper-heuristic Solver (FFPHHS)

## The representation

In FFPHHS, the FFP instance is represented as an n $\times$ n matrix that corresponds to an undirected graph of $n$ nodes, where a value of zero in a cell $(i, j$) indicates no connection between nodes $i$ and $j$. Conversely, a value of one for the same cell indicates an edge connecting both nodes. Since the graph is undirected, if an edge exists between node $i$ and $j$, it also exists between $j$ and $i$.

Currently, the system can read only a limited number of FFP instances, which are also distributed within this repository. The system reads one instance at a time, but you may want to implement a way to read batches of instances.

## The solver

The solver takes an FFP instance and solves iteratively using a system of turns. Every turn, the solver:

- Saves as many nodes as possible (given the available firefighters per turn). Once a node is protected, it will remain protected until the solver stops. Please note that when a firefighter protects a node, the solver "removes" it from the FFP instance (it removes the edges connecting such a node with the rest of the nodes). This allows for reducing the number of nodes and edges within the instance.
- It propagates the fire to every unprotected node directly connected to a burning node.

The process stops when there are no more nodes to propagate the fire.

The solver can handle more than one firefighter and also heuristics and hyper-heuristics.

## The features

There are four features already defined within FFPHHS:

- **EDGE_DENSITY**. It represents the percentage of edges that remain in the FFP instance. It is calculated as the sum of the remaining edges divided by the maximum number of edges within the instance.
- **AVG_DEGREE**. It represents the normalized average degree of the FFP instance. The degree of a node is the number of edges that connect such a node with other nodes in the graph. This feature calculates the average degree among all the nodes and then divides it by the maximum degree within the instance (n - 1).
- **BURNING_NODES**. It represents the percentage of unprotected nodes that the fire has reached. It is calculated as the number of burning nodes over the total number of nodes. Please note that this feature is also used as a performance metric for the solver. In this case, the smaller the value, the better.
- **BURNING_EDGES**. It is somehow related to the previous feature. However, instead of measuring the nodes, it represents the percentage of edges that connect burning nodes with unprotected ones. The larger this value, the more difficult it will be to stop the fire from spreading.
- **NODES_IN_DANGER**. It represents the percentage of nodes that are currently in danger. In other words, the number of nodes at risk since they are directly connected to one or more burning nodes.

Please note that all the features described above are dynamic. This means that they change as the FFP instance is being solved.

## The heuristics

There are two heuristics already defined within FFPHHS:

- **LDEG**. This is the implementation of the local degree heuristic. This heuristic will protect first the unprotected node at risk (connected directly to a burning node) with the largest degree.
- **GDEG**. This is a generalization of the previous heuristic. The global degree heuristic first protects the unprotected node with the largest degree, even if it is not at risk when making the decision.

## About the hyper-heuristics

Feel free to modify this code in any way that best fits your needs and implement your hyper-heuristics from scratch (please consider potential changes to the solver if doing so). However, if you want to use our basic skeleton for implementing hyper-heuristics, you will have to extend the **HyperHeuristic** class provided along with the system. In order to do so, you are requested to implement at least three methods (that will override the ones in the superclass):

- **__init__**. The constructor of your hyper-heuristic. It is mandatory that you provide a valid list of features and heuristics and call the super constructor to initialize these variables properly.
- **__str__**. It returns the string representation of the hyper-heuristic. It is needed for visualizing the hyper-heuristics on screen.
- **nextHeuristic**- The core of the hyper-heuristic since it indicates the next heuristic to apply given the current problem state.

## Disclaimer

FFPHHS is provided "as is" and "with all faults." We make no representations or warranties of any kind concerning the safety, suitability, lack of viruses, inaccuracies, typographical errors, or other harmful components of this FFPHHS. There are inherent dangers in the use of any software, and you are solely responsible for determining whether FFPHHS is compatible with your equipment and other software installed on your equipment. You are also solely responsible for the protection of your equipment and backup of your data, and we will not be liable for any damages you may suffer in connection with using, modifying, or distributing FFPHHS.