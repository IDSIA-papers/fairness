import pyagrum as gum

from .utils import convert_bn_to_moral_graph, extract_clique_families


def BNtoMRF(bn: gum.BayesNet) -> gum.MarkovRandomField:
    """
    Build a Markov Random Field (MRF) from a Bayesian Network (BN).

    Args:
        bn (gum.BayesNet): The Bayesian network to be converted.
    Returns:
        gum.MarkovRandomField: The resulting Markov Random Field.
    """

    def assignment_by_copy(bn: gum.BayesNet, clique: set[int]) -> gum.Tensor:
        """
        Assign a potential to a clique by copying the CPT of the child node.

        Args:
            bn (gum.BayesNet): The Bayesian network.
            clique (set[int]): The clique of nodes.

        Returns:
            gum.Tensor: The potential for the clique.
        """
        # Case 1: clique is a single node
        if len(clique) == 1:
            node = next(iter(clique))
            if not bn.parents(node):
                return bn.cpt(node)
            else:
                return gum.Tensor()

        # Case 2: clique is a set of nodes
        else:
            # Find the "child" node in the clique
            for node in clique:
                parents = set(bn.parents(node))  # type: ignore
                if parents == clique - {node}:  # TODO: check if this is correct
                    return bn.cpt(node)
            return gum.Tensor()

    mrf = gum.MarkovRandomField()
    graph = convert_bn_to_moral_graph(bn)
    cliques = extract_clique_families(graph)
    # graph: nx.Graph = convert_bn_to_moral_graph(bn)
    # cliques = extract_maximal_cliques(bn)

    # print(f"Cliques found: {cliques}")

    for node in graph.nodes():  # type: ignore
        variable = bn.variable(node)
        mrf.add(variable)

    for clique in cliques:
        # print(f"Processing clique: {clique}")

        potential = assignment_by_copy(bn, clique)
        if not potential.empty():
            # print(f"Adding potential for clique: {clique}")
            mrf.addFactor(potential)

    return mrf


def BNtoRatioMRF(
    bn: gum.BayesNet, target: int | str, normalize: bool = False
) -> gum.MarkovRandomField:
    """
    Build a Markov Random Field (MRF) from a Bayesian Network (BN) using the robust ratio method.

    This method is used to convert a Bayesian network into a Markov random field by
    assigning potentials to cliques based on the conditional probability tables (CPTs) of the
    child nodes. The target variable is used to determine the cliques and their potentials.

    Args:
        bn (gum.BayesNet): The Bayesian network to be converted.
        target (int): The target variable.
        normalize (bool): Whether to normalize the potentials. Defaults to False.

    Returns:
        gum.MarkovRandomField: The resulting Markov Random Field.
    """
    if isinstance(target, str):
        target = int(bn.idFromName(target))

    if target not in set(bn.nodes()):  # type: ignore
        raise ValueError(
            f"Target variable {target} not found in the Bayesian network. Please check the target variable."
        )

    if len(bn.variable(target).labels()) > 2:
        raise NotImplementedError(
            "Only target binary variables are supported. Please check the target variable."
        )

    mrf = gum.MarkovRandomField()
    children_of_target = bn.children(target)

    # add all the variables to the MRF but the target variable
    for node in bn.nodes():  # type: ignore
        if node != target:
            mrf.add(bn.variable(node))

    for node in bn.nodes():  # type: ignore
        if node == target or node in children_of_target:
            target_varname = bn.variable(target).name()
            target_domain = bn.variable(target).labels()
            if len(target_domain) != 2:
                raise ValueError(
                    f"Target variable {target} must be binary. Found {len(target_domain)} labels."
                )

            cpt = bn.cpt(bn.variable(node).name())
            phi = gum.Tensor()

            # Following eq (16) in the paper, we compute the ratio

            if len(cpt.names) == 1:
                continue  # Skip if the node has no parents

            phi = cpt.extract({target_varname: target_domain[1]}) / cpt.extract(
                {target_varname: target_domain[0]}
            )
            # phi = cpt.extract({target_varname: 0}) / cpt.extract({target_varname: 1})

            phi = phi.normalize() if normalize else phi

            mrf.addFactor(phi)

    return mrf
