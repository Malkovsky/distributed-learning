from collections import defaultdict
from math import isclose
from utils.fast_averaging import find_optimal_weights

ABC_3 = {'Alice':   {'Alice': 0.34, 'Bob': 0.33, 'Charlie': 0.33},
         'Bob':     {'Alice': 0.33, 'Bob': 0.34, 'Charlie': 0.33},
         'Charlie': {'Alice': 0.33, 'Bob': 0.33, 'Charlie': 0.34}}

ABC_5 = {'Alice':   {'Alice': 0.5,  'Bob': 0.25, 'Charlie': 0.25},
         'Bob':     {'Alice': 0.25, 'Bob': 0.5,  'Charlie': 0.25},
         'Charlie': {'Alice': 0.25, 'Bob': 0.25, 'Charlie': 0.5}}

ABC_9 = {'Alice':   {'Alice': .9,   'Bob': 0.05, 'Charlie': 0.05},
         'Bob':     {'Alice': 0.05, 'Bob': .9,   'Charlie': 0.05},
         'Charlie': {'Alice': 0.05, 'Bob': 0.05, 'Charlie': .9}}

CYCLE4_5 = {'North': {'North': 0.5, 'East':  0.25, 'West':  0.25},
            'East':  {'East':  0.5, 'North': 0.25, 'South': 0.25},
            'South': {'South': 0.5, 'East':  0.25, 'West':  0.25},
            'West':  {'West':  0.5, 'South': 0.25, 'North': 0.25}}

FULL4_52 = {'North': {'North': 0.52, 'East':  0.16, 'West':  0.16, 'South': 0.16},
            'East':  {'East':  0.52, 'North': 0.16, 'South': 0.16, 'West':  0.16},
            'South': {'South': 0.52, 'East':  0.16, 'West':  0.16, 'North': 0.16},
            'West':  {'West':  0.52, 'South': 0.16, 'North': 0.16, 'East':  0.16}}

TOP_5 = {
    0: {0: 0.85, 1: 0.05, 2: 0.05, 4: 0.05},
    1: {0: 0.05, 1: 0.85, 2: 0.05, 3: 0.05},
    2: {0: 0.05, 1: 0.05, 2: 0.8, 3: 0.05, 4: 0.05},
    3: {1: 0.05, 2: 0.05, 3: 0.85, 4: 0.05},
    4: {0: 0.05, 2: 0.05, 3: 0.05, 4: 0.85},
}

LONELY = {'Model': {'Model': 1.0}}


def adj2edges(graph: dict) -> set:
    """
    Converts adjacency list of graph to a list of edges.
    :param graph: adjacency list
    :return: set of edges where (v, u) and (u, v) is the same edges.
    """
    edges = set()

    for v, neighbors in graph.items():
        for u in neighbors:
            if v != u\
                    and (v, u) not in edges\
                    and (u, v) not in edges:
                edges |= {(v, u)}
    return edges


def edges2topology(edges: list, weights: list = None) -> dict:
    """
    Converts the edges of a graph to an adjacency list with edge weights.
    If no weights are given then it takes them from utils.fast_averaging.find_optimal_weights
    Normalizes all edge sums to 1.0 for each vertices.
    The graph must be undirected.
    :param edges: list of edges (v, u)
    :param weights: list of edge weights
    :return: adjacency list, where graph[v] - adjacency list for vertex v, graph[v][u] - weight for edge (v, u)
    """
    if not weights:
        weights, _ = find_optimal_weights(edges)
        weights = list(weights)

    assert (len(edges) == len(weights)),\
        f"The number of edges= {len(edges)}, the number of weights= {len(weights)}, but must be equal."

    graph = defaultdict(dict)
    sums = defaultdict(float)
    for i, (v, u) in enumerate(edges):
        w = weights[i] if weights else 1.
        if u not in graph[v]:
            graph[v][u] = w
        else:
            graph[v][u] += w
        sums[v] += w

        if v not in graph[u]:
            graph[u][v] = w
        else:
            graph[u][v] += w
        sums[u] += w

    for v in sums:
        if v in graph[v] and isclose(sums[v], 1.):
            continue
        assert (sums[v] >= 0.), f"The sum of the weights incident to the vertex {v} edges is negative ({sums[v]})."
        if sums[v] < 1.:
            w = 1. - sums[v]
            if v not in graph[v]:
                graph[v][v] = w
            else:
                graph[v][v] += w
        else:
            if v not in graph[v]:
                graph[v][v] = 1.
                sums[v] += 1.
            for u in graph[v]:
                graph[v][u] /= sums[v]

    return graph
