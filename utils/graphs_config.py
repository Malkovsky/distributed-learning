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

LONELY = {'Model': {'Model': 1.0}}


def adj2edges(graph: dict) -> set:
    edges = set()

    for v, neighbors in graph.items():
        for u in neighbors:
            if v != u\
                    and (v, u) not in edges\
                    and (u, v) not in edges:
                edges.update({(v, u)})
    return edges
