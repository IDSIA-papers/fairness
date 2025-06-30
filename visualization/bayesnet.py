import typing as ty

import pyagrum as gum
import pyagrum.lib.image as gumimage
import pyagrum.lib.notebook as gnb
from loguru import logger


def visualize_bn(
    bn: gum.BayesNet,
    name: str,
    learning_method: str,
    save_path: ty.Optional[str] = None,
    simple: ty.Optional[str] = None,
):
    """
    Visualize the Bayesian network

    Args:
        bn (gum.BayesNet): The Bayesian network to visualize.
        name (str): Name of the dataset.
        learning_method (str): Learning method used for the Bayesian network.
        save_path (ty.Optional[str]): Path to save the Bayesian network image. Defaults to None.
        simple (ty.Optional[str]): Simplification type, if any. Defaults to None.
    """

    # Create dictionaries for node colors
    logger.info(f"Visualizing Bayesian network for dataset: {name}")
    logger.info(
        f"Learned Bayesian network with {bn.size()} nodes and {bn.sizeArcs()} arcs"
    )
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
    # if True:
    #     gumimage.export(bn, f"BNs_images/{name}_simplified.png", nodeColor=colors)

    if save_path:
        logger.info(f"Saving Bayesian network to {save_path}")
        try:
            gum.saveBN(bn, f"{save_path}/{name}_{learning_method}_{simple}.pkl")
            # gum.saveBN(bn, f"{save_path}/{name}_{learning_method}_{simple}.net")

            logger.success(f"Bayesian network successfully saved to {save_path}")
        except Exception as e:
            logger.success(f"Error saving Bayesian network: {e}")

        # Export the Bayesian network as a PDF
        gumimage.export(
            bn, f"{save_path}/{name}_{learning_method}_{simple}.pdf", nodeColor=colors
        )
