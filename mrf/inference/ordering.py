from collections import abc, deque
from itertools import combinations


def find_neighbours(node: int, edges: set[tuple[int, ...]]) -> set[int]:
    """Find neighbours of a node in the graph.

    Args:
        node (int): The node for which to find neighbours.
        edges (set[tuple[int, int]]): The set of edges in the graph.

    Returns:
        set[int]: A set of neighbouring nodes.
    """
    return {
        n for n1, n2 in edges if n1 == node or n2 == node for n in (n1, n2) if n != node
    }


def greedy_ordering(
    # graph: gum.MarkovRandomField,
    nodes: set[int],
    edges: set[tuple[int, ...]],
    metric: abc.Callable[[int, set[tuple[int, ...]] | None], int],
) -> list[int]:
    """Perform a greedy ordering of the nodes in a Markov Random Field or Bayes Net.

    Args:
        # graph (gum.MarkovRandomField): The graph to order.
        nodes (set[int]): A set of nodes in the graph.
        edges (set[tuple[int, ...]]): A set of edges in the graph, where each edge is a tuple of two nodes.
        metric (abc.Callable[[gum.MarkovRandomField, int], int]): A function that computes a metric for each node.

    Returns:
        list[int]: A list of nodes in the order determined by the greedy algorithm.
    Raises:
        TypeError: If the graph is not a MarkovRandomField or BayesNet.
    Example:
        >>> def metric(graph, node):
        ...     return len(graph.neighbours(node))
        >>> mrf = gum.MarkovRandomField()
        >>> order = greedy_ordering(mrf, metric)
    """
    # if isinstance(graph, gum.MarkovRandomField):
    #     # undigraph = gum.MarkovRandomField(graph)
    #     undigraph = deepcopy(graph)
    # else:
    #     raise TypeError("Graph must be a MarkovRandomField.")

    # nodes: set[int] = graph.nodes()  # type: ignore
    # edges: set[tuple[int, ...]] = graph.edges()  # type: ignore
    edges = {tuple(sorted(edge)) for edge in edges}

    # Initialize the ordering and unmarked nodes
    unmarked = set(nodes)
    order = deque()

    for k in range(len(nodes)):
        # Select an unmarked node that minimizes the metric
        min_node = min(unmarked, key=lambda x: metric(x, edges))
        order.append(min_node)

        neighbours: set[int] = find_neighbours(min_node, edges)
        if len(neighbours) > 1:  # To avoid creating singleton cliques
            # Introduce edges between the neighbours of the selected node
            candidate_edges = combinations(neighbours, 2)

            for e in candidate_edges:
                e = tuple(sorted(e))
                if e not in edges:
                    edges.add(e)

        unmarked.remove(min_node)

    return list(order)


def minneighbours_metric(node: int, edges: set[tuple[int, ...]]) -> int:
    """**MIN-NEIGHBOURS** cost function: the number of the neighbours of a vertex in the
    current graph (the cost of the clique creation)"""
    return len(find_neighbours(node, edges))  # type: ignore


def minfill_metric(node: int, edges: set[tuple[int, ...]]) -> int:
    """**MIN-FILL** cost function: the number of the edges that needs to be added to the graph
    to its elimination (the "edge" cost of the clique creation)"""
    neighbours: set[int] = find_neighbours(node, edges)

    # Generate all possible edges between neighbors
    candidate_edges = {tuple(sorted((u, v))) for u, v in combinations(neighbours, r=2)}

    # Count edges not already in the graph
    existing_edges = {tuple(sorted(edge)) for edge in edges}
    return len(candidate_edges - existing_edges)
