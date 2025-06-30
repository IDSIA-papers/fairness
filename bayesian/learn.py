import os
import pickle
import typing as ty

import pandas as pd
import pyagrum as gum
import pyagrum.lib.image as gumimage
import pyagrum.lib.notebook as gnb
from loguru import logger


def learn_bayesian_network(
    dataset: pd.DataFrame,
    df_name: str,
    target: ty.Optional[str] = None,
    exclude_columns: ty.Optional[list[str]] = None,
    disconnect_columns: ty.Optional[list[str]] = None,
    show: bool = True,
    learning_method: str = "tabu",
    learning_params: ty.Optional[dict] = None,
    save_path: ty.Optional[str] = None,
) -> gum.BayesNet:
    """
    Build a Bayesian network from a dataframe using pyAgrum.

    Args:
        dataframe (pandas.DataFrame): The dataframe to learn the Bayesian network from.
        df_name (str): Name of the dataframe, used for saving the network.
        target (str, optional): The target variable to predict. If None, no specific target is set.
        exclude_cols (list, optional): Columns to exclude from the network.
        disconnect_cols (list, optional): Columns to disconnect from the rest of the network.
        show (bool): Whether to display the learned Bayesian network. Defaults to True.
        learning_method (str): Method to use ('tabu', 'greedy', or 'miic'). Defaults to 'tabu'.
        learning_params (dict, optional): Parameters for the learning method. Defaults to None.
        save_path (str, optional): Path to save the network. If None, it isn’t saved.

    Returns:
        pyAgrum.BayesNet: The learned Bayesian network.
    """
    learning_flag: bool = False

    logger.info(
        "--------------------------------------------------------------------------"
    )
    if save_path is None or f"{df_name}_{learning_method}.pkl" not in os.listdir(
        f"{save_path}/"
    ):
        # Make a copy of the dataframe to avoid modifying the original
        data = dataset.copy()

        if exclude_columns:
            logger.debug(f"Excluding columns: {exclude_columns}")
            data = data.drop(columns=exclude_columns, errors="ignore")

        # Learn the structure of the Bayesian network
        logger.info("Learning the structure of the Bayesian network...")
        learner = gum.BNLearner(data)

        if learning_params is None:
            learning_params = {}

        # Add structural constraints if provided
        if "must_have_arcs" in learning_params:
            for arc in learning_params["must_have_arcs"]:
                dest, src = arc  # from target to sensible feature
                logger.debug(f"Adding mandatory arc: {src} → {dest}")
                learner.addMandatoryArc(src, dest)

        if "forbidden_arcs" in learning_params:
            for arc in learning_params["forbidden_arcs"]:
                src, dest = arc
                logger.debug(f"Adding forbidden arc: {src} → {dest}")
                learner.addForbiddenArc(src, dest)

        # Set learning method
        if learning_method == "tabu":
            tabu_size = learning_params.get("tabu_list_size", 10)
            max_iter = learning_params.get("max_iter", 100)
            learner.useLocalSearchWithTabuList(tabu_size, max_iter)

        elif learning_method == "greedy":
            logger.info("Using Greedy Hill Climbing with default score")
            learner.useGreedyHillClimbing()

        elif learning_method == "miic":
            learner.useMIIC()

        elif learning_method == "3off2":
            # print("Using 3off2 structure learning algorithm")
            learner.use3off2()

        elif learning_method == "k2":
            if target is not None:
                # Create a node ordering with target as the last node (effect)
                node_names = list(data.columns)
                if target in node_names:
                    node_names.remove(target)
                    # Put other nodes first, followed by target
                    node_ordering = learning_params.get(
                        "node_ordering", node_names + [target]
                    )
                    logger.debug(
                        f"Using K2 with custom node ordering, target '{target}' last"
                    )
                    learner.useK2(node_ordering)
                else:
                    logger.debug(
                        f"Warning: Target variable '{target}' not found in data columns. Using default K2 ordering."
                    )
                    learner.useK2(list(data.columns))
            else:
                logger.info("Using K2 with default ordering")
                learner.useK2(list(data.columns))
        else:
            logger.info(
                f"Warning: Unknown learning method '{learning_method}'. Falling back to Tabu Search."
            )
            learner.useLocalSearchWithTabuList()

        # Handle disconnected columns
        if disconnect_columns:
            for column in disconnect_columns:
                logger.debug(f"Disconnecting column: {column}")
                for other_column in data.columns:
                    if other_column != column:
                        # Forbid arcs to and from the disconnected column
                        learner.addForbiddenArc(column, other_column)
                        learner.addForbiddenArc(other_column, column)

        # Learn the Bayesian network
        bn = learner.learnBN()

    else:
        logger.info("Loading from saved Bayesian network for dataset: ", df_name)
        with open(f"{save_path}/{df_name}_{learning_method}.pkl", "rb") as f:
            bn = pickle.load(f)

    # Display the learned Bayesian network
    if show:
        logger.success(
            f"Learned Bayesian network with {bn.size()} nodes and {bn.sizeArcs()} arcs"
        )

        # Create dictionaries for node colors
        colors = {}
        for node in bn.nodes():
            node_name = bn.variable(node).name()
            if node_name.startswith("T_"):
                colors[node_name] = 0.5
            elif (
                node_name.startswith("S")
                and len(node_name) > 1
                and node_name[1:].split("_")[0].isdigit()
            ):
                colors[node_name] = 0.2
            else:
                colors[node_name] = 0.7

        # Show the Bayesian network with custom node colors
        gnb.showBN(bn, nodeColor=colors)

        # save as png image
        if True:
            gumimage.export(
                bn, f"{save_path}/{df_name}_{learning_method}.pdf", nodeColor=colors
            )

    # Save the Bayesian network if a save path is provided
    if save_path:
        logger.info(f"Saving Bayesian network to {save_path}")
        try:
            gum.saveBN(bn, f"{save_path}/{df_name}_{learning_method}.pkl")
            # gum.saveBN(bn, f"{save_path}/{df_name}_{learning_method}.net")
            logger.success(f"Bayesian network successfully saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving Bayesian network: {e}")

    return bn
