import collections.abc as abc
import typing as ty
from itertools import combinations

import pyagrum as gum


def names(net: gum.BayesNet | gum.MarkovRandomField, ids: abc.Iterable[str | int]):
    """
    Get the names of the variables in the Markov random field.

    Args:
        *ids (str): The IDs of the variables.

    Returns:
        list[str]: A list of variable names.
    """
    _names = []
    for _id in ids:
        if isinstance(_id, str):
            try:
                name = net.variable(_id).name()
            except gum.NotFound:
                raise ValueError(f"Variable '{_id}' not found in the network.")
            _names.append(name)
        elif isinstance(_id, int):
            if _id not in set(net.nodes()):
                raise ValueError(f"Variable ID '{_id}' not found in the network.")

            _names.append(net.variable(_id).name())

    return _names


def names_dict(
    net: gum.BayesNet | gum.MarkovRandomField,
    ids_dict: abc.Mapping[str | int, ty.Any],
) -> dict[str, ty.Any]:
    """
    Convert the keys and the values of a dictionary to their corresponding string names.

    It is assumed that any key in the dictionary is a valid variable ID or name in the network.
    Also values are assumed to be integer labels of the variables or strings.

    Args:
        net (gum.BayesNet | gum.MarkovRandomField): The network from which to get the names.
        ids_dict (abc.Mapping[str | int, ty.Any]): A dictionary with variable IDs or names as keys.

    Returns:
        dict[str, ty.Any]: A dictionary with variable names as keys and their corresponding values.
    """
    ids = ids_dict.keys()
    _names = names(net, ids)

    # Create initial mapping from names to values
    _names_to_val = dict(zip(_names, ids_dict.values()))

    # Process the dictionary to convert integer values to their labels
    result = {}
    for k, v in _names_to_val.items():
        if isinstance(v, int):
            result[k] = net.variable(k).label(v)
        else:
            # Keep string values as they are
            result[k] = v

    return result


def ids(
    net: gum.BayesNet | gum.MarkovRandomField, names: abc.Iterable[str | int]
) -> list[int]:
    """
    Get the IDs of the variables in the Markov random field.

    Args:
        *names (str): The names of the variables.

    Returns:
        list[int]: A list of variable IDs.
    """
    _ids = []
    for name in names:
        if isinstance(name, str):
            try:
                _id = net.idFromName(name)
            except gum.NotFound:
                raise ValueError(f"Variable '{name}' not found in the network.")
            _ids.append(_id)
        elif isinstance(name, int):
            if name not in set(net.nodes()):
                raise ValueError(f"Variable ID '{name}' not found in the network.")

            _ids.append(name)

    return _ids


def ids_dict(
    net: gum.BayesNet | gum.MarkovRandomField,
    names_dict: abc.Mapping[str | int, ty.Any],
) -> dict[int, ty.Any]:
    """Convert the keys of a dictionary to their corresponding IDs.

    Args:
        net (gum.BayesNet | gum.MarkovRandomField): The network from which to
            get the IDs.
        names_dict (abc.Mapping[str | int, ty.Any]): A dictionary with variable
            names or IDs as keys.
    Returns:
        dict[int, ty.Any]: A dictionary with variable IDs as keys and their
    """
    _ids_to_val = {}
    names = names_dict.keys()
    _ids = ids(net, names)
    _ids_to_val = dict(zip(_ids, names_dict.values()))
    return _ids_to_val


def to_str_dict(
    net: gum.BayesNet | gum.MarkovRandomField,
    str_dict: abc.Mapping[str | int, str | int],
) -> dict[str, ty.Any]:
    """Convert the keys and the values of a dictionary to their corresponding string names."""
    _str_dict = {}
    for key, value in str_dict.items():
        if isinstance(key, int):
            str_key = net.variable(key).name()
        elif isinstance(key, str):
            str_key = key
        else:
            raise ValueError(f"Key '{key}' is neither an int nor a str.")

        if isinstance(value, int):
            str_value = net.variable(str_key).label(value)
        elif isinstance(value, str):
            str_value = value
        else:
            raise ValueError(f"Value '{value}' is neither an int nor a str.")

        _str_dict[str_key] = str_value
    return _str_dict


#################################################################
# Pyagrum MRF conversion utility functions
#################################################################


def convert_bn_to_moral_graph(bn: gum.BayesNet) -> gum.MixedGraph:
    """
    Moralize a Bayesian network by adding edges between parents of the same child node
    and removing the direction of the edges.

    Args:
        bn (gum.BayesNet): The Bayesian network to be moralized.

    Returns:
        gum.MixedGraph: The moral graph of the Bayesian network.
    """
    moral_graph = gum.MixedGraph()
    for node in bn.nodes():
        moral_graph.addNodeWithId(node)

    for node in bn.nodes():  # type: ignore
        parents = list(bn.parents(node))  # type: ignore

        # Connect all pairs of parents
        edges = list(combinations(parents, 2))
        for e in edges:
            if e not in bn.arcs():
                moral_graph.addEdge(*e)

        # Add original edges (undirected)
        for parent in parents:
            moral_graph.addArc(parent, node)

    return moral_graph


def extract_clique_families(graph: gum.MixedGraph) -> list[set[int]]:
    """
    Extract clique families from a moral graph.

    This by any means is not a complete clique tree extraction algorithm.

    Args:
        graph (gum.MixedGraph): The moral graph.

    Returns:
        list[set[int]]: A list of sets representing the cliques.
    """
    cliques = []
    for node in graph.nodes():
        family = list(graph.parents(node))  # type: ignore
        cliques.append(set(family) | set([node]))
    return cliques


#################################################################
# Networkx MRF conversion utility functions
#################################################################


def convert_bn_to_moral_graph_nx(bn: gum.BayesNet):
    """Convert a pyAgrum BN to a NetworkX undirected moral graph.

    Args:
        bn (gum.BayesNet): The Bayesian network to be converted.

    Returns:
        nx.Graph: The undirected moral graph.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required for this function. Please install it with 'pip install networkx'."
        )

    G = nx.Graph()
    for node in bn.nodes():
        G.add_node(node)
        parents = list(bn.parents(node))
        # Connect all pairs of parents (moralization)
        for u, v in combinations(parents, 2):
            G.add_edge(u, v)
        # Add original edges (undirected)
        for parent in parents:
            G.add_edge(parent, node)
    return G


def extract_maximal_cliques(bn: gum.BayesNet) -> list[set[int]]:
    """Extract maximal cliques using NetworkX."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required for this function. Please install it with 'pip install networkx'."
        )

    G = convert_bn_to_moral_graph_nx(bn)
    return [set(clique) for clique in nx.find_cliques(G)]
