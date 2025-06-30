import itertools
import typing as ty

import numpy as np
import pyagrum as gum

from datasets.utils import powerset


def build_inference_engine(
    bn: gum.BayesNet, engine=gum.LazyPropagation
) -> gum.LazyPropagation:
    """
    Analyze the learned Bayesian network for a specific target variable.

    Parameters:
        bn (gum.BayesNet): The learned Bayesian network.

    Returns:
        The inference engine for the Bayesian network.
    """
    ie = engine(bn)
    ie.makeInference()
    return ie


def compute_posteriors_given_sensible_features_combinations(
    ie: gum.LazyPropagation,
    sensible_feat_to_states: dict[str, list[ty.Any]],
    target: str,
) -> dict[str, dict[str, np.typing.NDArray[np.float64]]]:
    """
    Get the posterior probabilities for all combinations of sensible features.

    This function iterates through all combinations of sensible features (its powerset),
    sets the evidence for the configuration of the sensible features we are iterating,
    and computes the posterior probabilities for the target variable.

    Parameters:
        ie: Inference engine
        sensible_states: Dictionary of the states of the sensible features
        target: Target variable

    Returns:
        posteriors: Dictionary of the posterior probabilities for all combinations of sensible features
    """

    posteriors = {}

    all_combinations = list(powerset(sensible_feat_to_states.keys()))[1:]

    for combination in all_combinations:
        columns_key = " and ".join(combination)
        posteriors[columns_key] = {}

        for values in itertools.product(
            *[sensible_feat_to_states[col] for col in combination]
        ):
            evidence = dict(zip(combination, map(str, values)))
            ie.setEvidence(evidence)
            ie.makeInference()
            posteriors[columns_key][" and ".join(map(str, values))] = ie.posterior(
                target
            ).toarray()
            ie.eraseAllEvidence()

    return posteriors


####################################################################
# Compute the mid-point assignment based on the posterior distributions (eq (12) in the paper)
####################################################################


def select_optimal_sensitive_feature_perturbation(
    baseline_posterior: np.ndarray,
    posterior_lower: np.typing.NDArray,
    assignment_lower: dict[str, ty.Any],
    posterior_upper: np.typing.NDArray,
    assignment_upper: dict[str, ty.Any],
    target_idx_label: int = 0,
) -> tuple[dict[str, ty.Any], np.typing.NDArray | float]:
    """
    Compute the private feature perturbation that maximizes the absolute difference in posterior probabilities
    for Y = 0.

    This function compares the baseline posterior with the upper and lower perturbations of the sensitive feature.
    If the baseline posterior is less than the average of the upper and lower perturbations, it returns the upper perturbation.
    Otherwise, it returns the lower perturbation.

    Parameters:
        baseline_posterior (np.ndarray): The posterior probabilities of the target variable without perturbation.
        posterior_lower (np.typing.NDArray): The posterior probabilities of the target variable with lower perturbation.
        assignment_lower (dict[str, ty.Any]): The assignment of the sensitive feature for the lower perturbation.
        posterior_upper (np.typing.NDArray): The posterior probabilities of the target variable with upper perturbation.
        assignment_upper (dict[str, ty.Any]): The assignment of the sensitive feature for the upper perturbation.
        target_idx_label (int): The index of the target label in the posterior probabilities.
    Returns:
        tuple: A tuple containing the assignment of the sensitive feature and the posterior probabilities for the selected perturbation.
    """

    if baseline_posterior[target_idx_label] < (
        0.5 * (posterior_upper[target_idx_label] + posterior_lower[target_idx_label])
    ):
        return assignment_upper, posterior_upper
    else:
        return assignment_lower, posterior_lower
