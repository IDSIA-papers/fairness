import typing as ty
from pathlib import Path

import pyagrum as gum
from loguru import logger


def extract_markov_blanket(
    bn: gum.BayesNet, target: str, save_path: ty.Optional[str | Path]
) -> gum.BayesNet:
    """
    Get the Markov blanket of a target variable in a Bayesian network, if a node in the Markov blanket
    has a parent being removed from the network, the node gets a uniform distribution

    Parameters:
        bn: Bayesian network
        target: Target variable
        save_path: Path to save the Markov blanket, if None, it won't be saved

    Returns:
        bn_new: Bayesian network with the Markov blanket of the target variable
    """
    logger.info(
        "--------------------------------------------------------------------------"
    )
    logger.info(f"Getting Markov blanket for {target}...")
    blanket = gum.MarkovBlanket(bn, target)
    nodes_to_remove = set(bn.nodes()) - set(blanket.nodes())  # type: ignore
    nodes_new_cpt = []  # nodes to add uniform distribution
    for node in blanket.nodes():
        if blanket.parents(node) == set():  # checks if node doesn't have parents
            parents_node_og_bn = bn.parents(node)
            if (
                parents_node_og_bn == set() or parents_node_og_bn in blanket.nodes()
            ):  # if node already didn't have parents or parents are present in the blanket
                continue
            else:
                nodes_new_cpt.append(node)

    relevant_arcs = set()

    # Add arcs from parents to target node
    for parent_id in bn.parents(target):
        relevant_arcs.add((parent_id, bn.idFromName(target)))

    # Add arcs from target to children
    children_ids = list(bn.children(target))
    for child_id in children_ids:
        relevant_arcs.add((target, child_id))

    # Add arcs between other parents of the children
    for child_id in children_ids:
        child_parents = list(bn.parents(child_id))
        for parent_id in child_parents:
            # Add arc from any parent to the child
            relevant_arcs.add((parent_id, child_id))

    arcs_to_remove = bn.arcs() - relevant_arcs

    bn_new = gum.BayesNet(bn)

    for node in nodes_to_remove:
        bn_new.erase(node)

    for arc in arcs_to_remove:
        bn_new.eraseArc(arc[0], arc[1])

    for node in nodes_new_cpt:
        logger.debug(
            f"Node {node} has parents {bn.parents(node)}, but they are not in the Markov blanket. Setting uniform distribution."
        )
        bn_new.cpt(node).fillWith(
            1 / bn_new.cpt(node).domainSize()
        )  # fill with uniform distribution

    for node_id in bn_new.nodes():
        if len(bn_new.parents(node_id)) == len(bn.parents(node_id)):
            # Only copy CPTs for nodes whose parent structure hasn't changed
            bn_new.cpt(node_id)[:] = bn.cpt(node_id)[:]
        else:
            # For nodes with modified parent structure, recalculate CPTs or set to uniform
            # This is a simplified approach - in practice you might need more sophisticated CPT adjustment
            bn_new.cpt(node_id).fillWith(1.0 / len(bn.variable(node_id).domain()))

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        gum.saveBN(bn_new, (save_path / f"{target}_markov_blanket.pkl").as_posix())

    return bn_new


def simplification_1(
    bn: gum.BayesNet,
    name: str,
    target: str,
    save_path: ty.Optional[str] = None,
    name_prefix: str = "",
) -> gum.BayesNet:
    """Simple simplification of the network by removing independent nodes"""
    nodes_toremove = []
    logger.info(f"Checking for independent nodes in {name}...")
    for j in bn.names() - {target}:  # type: ignore
        if bn.isIndependent(target, j, ()):
            logger.debug(f"{target} INDEP {j}")
            nodes_toremove.append(j)

    simplified_bn = gum.BayesNet(bn)
    for node in set(nodes_toremove):
        simplified_bn.erase(node)

    if save_path:
        logger.info(f"Saving simplified network to {save_path} ...")
        gum.saveBN(simplified_bn, (Path(save_path) / f"{name_prefix}.pkl").as_posix())
        logger.success(f"Saved simplified network to {save_path} as {name_prefix}.pkl")

    return simplified_bn


def simplification_2(
    bn: gum.BayesNet, name: str, target: str, sensible_columns: list[str]
) -> gum.BayesNet:
    """Second simplification of the network by removing independent nodes given sensible columns,
    if sesnsible columns are not in the network, they are removed from the network

    Parameters:
        bn: Bayesian network
        name: Name of the network
        target: Target variable
        sensible_columns: List of sensible columns

    Returns:
        simplified_bn: Simplified Bayesian network
    """

    nodes_toremove = []
    print(f"Checking for independent nodes in {name} given sensible features ...")
    if not bool(set(sensible_columns) & set(bn.names())):
        print(f"Sensible columns removed from the network in previous steps")
        return bn
    else:
        sensible_columns_f = set(sensible_columns) & set(bn.names())
    for col in sensible_columns_f:
        for k in bn.names() - {target, col}:
            if bn.isIndependent(target, k, col):
                print(f"{target} INDEP {k} given {col}")
                nodes_toremove.append(k)

    simplified_bn = gum.BayesNet(bn)
    for node in set(nodes_toremove):
        simplified_bn.erase(node)
    print()

    return simplified_bn


def simplification_3(bn: gum.BayesNet, name: str, target: str, sensible_columns: str):
    """Third simplification of the network by removing sensible nodes that are independent given public nodes,
    if sensible columns are not in the network, they are removed from the network
    """

    nodes_toremove = []
    print(f"Checking for independent nodes in {name} given public features ...")
    if not bool(set(sensible_columns) & set(bn.names())):
        print(f"Sensible columns removed from the network in previous steps")
        return bn
    else:
        sensible_columns_f = set(sensible_columns) & set(bn.names())
    a = bn.names() - sensible_columns_f - {target}
    for col in sensible_columns_f:
        if bn.isIndependent(target, col, a):
            print(f"{target} INDEP {col} given {a}")
            nodes_toremove.append(col)

    simplified_bn = gum.BayesNet(bn)
    for node in set(nodes_toremove):
        simplified_bn.erase(node)
    print()

    return simplified_bn


def add_sensible_to_target_arcs(
    learning_params: dict, sensible_features: list[str], target: str
) -> dict[str, list[tuple[str, str]]]:
    """
    Add arcs from sensible features to the target variable in the learning parameters.

    Parameters:
        learning_params: Dictionary of learning parameters for Bayesian network learning
        sensible_features: List of sensitive/protected attributes
        target: Target variable names

    Returns:
        Updated learning parameters with mandatory arcs added
    """

    # Make a copy to avoid modifying the original
    params = learning_params.copy() if learning_params else {}

    # Initialize the must_have_arcs list if it doesn't exist
    if "must_have_arcs" not in params:
        params["must_have_arcs"] = []

    # Add arcs between sensible features and target
    for feature in sensible_features:
        arc = (feature, target)

        # Add the arc if it's not already in the list
        if arc not in params["must_have_arcs"]:
            params["must_have_arcs"].append(arc)

    logger.debug(
        f"Added {len(sensible_features)} mandatory arcs between sensible features and target"
    )
    return params


def filter_relevant_features(
    markov_blanket: gum.BayesNet, sensible_features: list[str]
) -> list[str]:
    """
    Get the sensible features that are relevant for the target variable in the Markov blanket.
    This function checks which sensible features are present in the Markov blanket of the target variable.

    Parameters:
        markov_blanket: The Markov blanket of the target variable.
        sensible_features: List of sensible features to check.
    Returns:
        List of sensible features that are present in the Markov blanket.
    """

    relevant_sensible_features = []
    for feature in sensible_features:
        if feature in list(markov_blanket.names()):
            relevant_sensible_features.append(feature)
    return relevant_sensible_features
