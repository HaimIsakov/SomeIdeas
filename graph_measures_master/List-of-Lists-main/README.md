# List-of-Lists Format

Lior Shifman and Haim Isakov

Our format is aimed at increasing the efficiency of our code by leveraging
the computer's internal cache mechanism. We call this structure a List Of Lists Graph.
The LOL Graph is composed of two arrays:
1. An array which is composed of all the neighborhood lists placed back to
back (Neighbors).
2. An array which holds the starting index of each node's neighbor list (In-
dices).

There follows an example of the conversion routine for a simple graph. The given
graph is the one composed of these edges: (0->1, 0->2,0->3,2->0,3->1,3->2).
The behavior of the conversion now depends on whether the graph is directed
or undirected.

• If the graph is directed, the result is as follows: Indices: [0, 3, 3, 4, 6],
Neighbors: [1, 2, 3, 0, 1, 2]

• Else, the results for the undirected graph are: Indices: [0, 3, 5, 7, 10],
Neighbors: [1, 2, 3, 0, 3, 0, 3, 0, 1, 2]

The LOL Graph object is designed around the principle of cache-awareness.
The most important thing to remember when accessing the graph is that we are aiming to accelerate the computations by loading sections
of the graph into the cache ahead of time for quick access. When using the
LOL Graph, this comes into effect when we iterate over the ofset vector first
and then access the blocks of neighbor nodes in the graph vector. By doing
this, we are pulling the entire list of a certain node's neighbors into the cache,
allowing us to iterate over them extremely quickly.

# How to use?

For undirected graph:
```

undirected_graph = LolGraph(directed=False, weighted=True)

undirected_graph.convert(edgeList) //For example: undirected_graph.convert([[1,2,3], [4,6,0.1]])
```

For directed graph:
```
directed_graph= DLGW(weighted=True)

directed_graph.convert(edgeList) //For example: directed_graph.convert([[1,2,3], [4,6,0.1]])
```

Basically, you can use LolGraph class to create a directed graph, but some functions (in_degrees, predecessors) are not implemented in the most effecient way. 
Therefore, use LolGraph for undirected graph, and DLGW for directed graph.

The name of the implemented functions are the same as if you were using Networkx library.
