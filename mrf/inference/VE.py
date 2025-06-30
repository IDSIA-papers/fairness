import enum
import typing as tp
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from textwrap import dedent

import numpy as np
import pandas as pd
import pyagrum as gum
from loguru import logger

from mrf.inference.ordering import greedy_ordering, minfill_metric
from mrf.network.utils import ids, ids_dict, names, names_dict


class ProjectionOperation(enum.Enum):
    """
    Enumeration for projection operations used in Variable Elimination.
    This enum defines the operations that can be used to project factors during inference.
    """

    SUM = gum.Tensor.sumOut
    MAX = gum.Tensor.maxOut
    MIN = gum.Tensor.minOut


class AssignmentOperation(enum.Enum):
    """
    Enumeration for assignment operations used in Variable Elimination.
    This enum defines the operations that can be used to find the best assignment
    during inference.
    """

    ARGMAX = gum.Tensor.argmax
    ARGMIN = gum.Tensor.argmin


@dataclass
class VariableElimination:
    """
    Variable Elimination (VE) algorithm for probabilistic inference.
    This class implements the VE algorithm to compute marginal probabilities
    over a set of variables in a probabilistic graphical model.
    """

    PGM: gum.MarkovRandomField = field(repr=False)
    potentials_map: dict[frozenset[int], gum.Tensor] = field(
        init=False, default_factory=dict, repr=False
    )
    targets: list[int] = field(init=False, default_factory=list, repr=True)
    evidence: dict[int, tp.Any] = field(init=False, default_factory=dict, repr=True)
    elimination_ordering: list[int] = field(init=False, default_factory=list, repr=True)

    def __post_init__(self):
        if not isinstance(self.PGM, (gum.MarkovRandomField, gum.BayesNet)):
            raise NotImplementedError(
                f"MRF must be an instance of {gum.MarkovRandomField} or {gum.BayesNet}"
            )
        else:
            # self.PGM = deepcopy(self.PGM)
            if isinstance(self.PGM, gum.BayesNet):
                self.PGM = gum.BayesNet(self.PGM)  # type: ignore
            else:
                self.PGM = gum.MarkovRandomField(self.PGM)
        # logger.success(f"Variable Elimination initialized with MRF: {self.MRF}")

    ############################################################
    ## Marginal Inference/Posterior Inference
    ############################################################

    def posterior(
        self,
        targets: tp.Optional[tp.Iterable[int | str]] = None,
        evidence: tp.Optional[tp.Mapping[int | str, tp.Any]] = None,
    ) -> "pd.Series[float] | pd.DataFrame":
        """
        Compute the posterior distribution over a set of target variables
        given observed evidence in the Markov Random Field (MRF).

        Args:
            targets[Y] (Iterable[int | str], optional): The target variables for which to compute the posterior.
                If None, no targets are considered.
            evidence[E=e] (Mapping[int | str, tp.Any], optional): The observed evidence in the MRF.
                If None, no evidence is considered.
        Returns:
            pd.Series[float] | pd.DataFrame: The posterior distribution over the target variables.
                If multiple targets are provided, a DataFrame is returned; otherwise, a Series is returned.
        Example:
            >>> mrf = gum.MarkovRandomField()
            >>> ve = VariableElimination(mrf)
            >>> posterior = ve.posterior(targets=['A', 'B'], evidence={'C': 1})
        """
        self.init_active_factors_data()

        # TODO: this can be embededd in a validation method
        if targets is not None:
            self.targets: list[int] = ids(self.PGM, targets)
        else:
            raise ValueError(
                "Targets must be provided for posterior inference. "
                "Use the 'targets' parameter to specify them."
            )

        if evidence is None:
            self.evidence = {}
        else:
            self.evidence: dict[int, tp.Any] = ids_dict(self.PGM, evidence)
            self.potentials_map = self.update_factors_with_evidence(
                factors=self.potentials_map,
                evidence=self.evidence,
            )

        # self.elimination_ordering = greedy_ordering(self.MRF, minfill_metric)
        nodes: set[int] = self.PGM.nodes()  # type: ignore
        edges: set[tuple[int, int]] = (
            self.PGM.edges()
            if isinstance(self.PGM, gum.MarkovRandomField)
            else self.PGM.arcs()
        )  # type: ignore
        elimination_ordering = greedy_ordering(
            nodes=nodes,
            edges=edges,
            metric=minfill_metric,
        )
        self.elimination_ordering = [
            node
            for node in elimination_ordering
            if node not in self.evidence and node not in self.targets
        ]

        logger.info(
            dedent(f"""
        Performing posterior inference with the following parameters:
        - Targets: {names(self.PGM, self.targets)}
        - Evidence: {names_dict(self.PGM, self.evidence)}
        - Elimination Ordering: {names(self.PGM, self.elimination_ordering)}""")
        )

        return self.sum_product(
            elimination_ordering=self.elimination_ordering,
        ).topandas()

    def sum_product(
        self,
        elimination_ordering: list[int],
    ) -> gum.Tensor:
        """
        Perform the SUM-PRODUCT inference algorithm on the MRF.

        Args:
            elimination_ordering (list[int]): The order in which to eliminate variables.

        Returns:
            gum.Tensor: The resulting factor after marginalization. A distribution over the remaining variables.
        """
        active_factors = self.potentials_map.copy()
        for node in elimination_ordering:
            active_factors, _ = self.projection_product_elimination_step(
                node,
                factors=active_factors,
                projection_operation=ProjectionOperation.SUM,
            )

        # Final combination and normalization
        final_factor = reduce(
            mul,
            active_factors.values(),
            gum.Tensor(),
        )

        # Normalize the final factor to ensure it is a valid probability distribution
        final_factor = final_factor / final_factor.sum()

        self.potentials_map = active_factors

        return final_factor

    ###########################################################
    ## MAP Inference
    ###########################################################

    def MAP(
        self,
        evidence: tp.Optional[tp.Mapping[int | str, tp.Any]] = None,
        projection_operation: ProjectionOperation = ProjectionOperation.MAX,
        assignment_operation: AssignmentOperation = AssignmentOperation.ARGMAX,
        verbose: bool = True,
    ) -> tuple[dict[str, tp.Any], np.typing.NDArray | float]:
        """
        Perform MAP inference on the MRF.

        It computes the most likely assignment to ALL of the non-evidence variables; then

        Args:
            evidence (dict[int | str, tp.Any], optional): The observed evidence in the MRF.
                If None, no evidence is considered.
            projection_operation (callable, optional): The operation to use for marginalization.
                Defaults to gum.Tensor.maxOut.
            assignment_operation (callable, optional): The operation to use for finding the MAP assignment.
                Defaults to gum.Tensor.argmax.
            verbose (bool, optional): If True, print detailed information about the inference process.

        Returns:
            tuple[dict[str, tp.Any], gum.Tensor]: A tuple containing the MAP assignment as a dictionary
                and the correspoding probability of the MAP assignment.

        Example:
        Maximum A Posteriori (MAP) inference on a Markov Random Field (MRF).
        >>> mrf = gum.MarkovRandomField()
        >>> ve = VariableElimination(mrf)
        >>> map_assignment, map_value = ve.MAP(evidence={'A': 1, 'B': 0})

        Minimum A Posteriori (MAP) inference on a Markov Random Field (MRF).
        >>> mrf = gum.MarkovRandomField()
        >>> ve = VariableElimination(mrf)
        >>> map_assignment, map_value = ve.MAP(
        ...     evidence={'A': 1, 'B': 0},
        ...     projection_operation=gum.Tensor.minOut,
        ...     assignment_operation=gum.Tensor.argmin
        ... )
        """
        self.init_active_factors_data()
        self.targets = []

        if evidence is None:
            self.evidence = {}
        else:
            self.evidence: dict[int, tp.Any] = ids_dict(self.PGM, evidence)
            self.potentials_map = self.update_factors_with_evidence(
                factors=self.potentials_map, evidence=self.evidence
            )

        # self.elimination_ordering = greedy_ordering(self.MRF, minfill_metric)
        nodes = self.PGM.nodes()  # type: ignore
        edges = (
            self.PGM.edges()
            if isinstance(self.PGM, gum.MarkovRandomField)
            else self.PGM.arcs()
        )
        elimination_ordering = greedy_ordering(
            nodes=nodes,
            edges=edges,
            metric=minfill_metric,
        )
        self.elimination_ordering = [
            node for node in elimination_ordering if node not in self.evidence
        ]

        logger.info(
            dedent(f"""
        Performing MAP inference query with the following parameters:
        - Targets: {names(self.PGM, self.targets)}
        - Evidence: {names_dict(self.PGM, self.evidence)}
        - Elimination Ordering: {names(self.PGM, self.elimination_ordering)}
        - Projection Operation: {projection_operation.__name__}
        - Assignment Operation: {assignment_operation.__name__}""")
        ) if verbose else None

        return self.max_product(
            elimination_ordering=self.elimination_ordering,
            projection_operation=projection_operation,
            assignment_operation=assignment_operation,
        )

    def max_product(
        self,
        elimination_ordering: list[int],
        projection_operation: tp.Callable,
        assignment_operation: tp.Callable,
    ) -> tuple[dict[str, tp.Any], np.typing.NDArray | float]:
        """
        Perform the MAX-PRODUCT inference algorithm on the MRF.
        Args:
            elimination_ordering (list[int]): The order in which to eliminate variables.
            projection_operation (callable): The operation to use for marginalization.
                Defaults to gum.Tensor.maxOut. But it allows also to search for the minimum.
            assignment_operation (callable): The operation to use for finding the MAP assignment.
                Defaults to gum.Tensor.argmax. It allows also to search for the minimum.
        Returns:
            tuple[dict[str, tp.Any], gum.Tensor]: A tuple containing the MAP assignment as a dictionary
                and the corresponding tensor representing the assignment value.
        """
        active_factors = self.potentials_map.copy()

        elimination_trace: list[tuple[int, gum.Tensor]] = []
        for node in elimination_ordering:
            active_factors, combined_factor = self.projection_product_elimination_step(
                node,
                active_factors,
                projection_operation=projection_operation,
            )

            # Store the ELIMINATED node and the combined factor for traceback
            if len(combined_factor.names) > 0:
                elimination_trace.append((node, combined_factor))

        self.potentials_map = active_factors

        assignment = self.tracebackMAP(
            elimination_trace, operation=assignment_operation
        )

        if len(self.potentials_map) > 1:
            raise ValueError("Multiple active factors remain after elimination. ")
        else:
            last_factor = list(self.potentials_map.values())[0]
            map_value = last_factor.toarray()

        return assignment, map_value

    def tracebackMAP(
        self,
        elimination_trace: list[tuple[int, gum.Tensor]],
        /,
        operation: tp.Callable = gum.Tensor.argmax,
    ) -> dict[str, tp.Any]:
        """
        Traceback the elimination process to find the MAP assignment.

        Args:
            elimination_trace (list[tuple[int, gum.Tensor]]): The trace of the elimination process,
                where each tuple contains a node and the corresponding factor after elimination.
            operation (callable): The operation to use for finding the MAP assignment.
            If not provided, defaults to gum.Tensor.argmax.
        Returns:
            dict: A dictionary representing the MAP assignment for the variables.
        """
        assignment: dict = {}

        for eliminated_node, factor in reversed(elimination_trace):
            eliminated_variable = self.PGM.variable(eliminated_node).name()

            if not assignment:
                # First step, just take the argmax of the factor
                assignment_list, value = operation(factor)

                # By default we take the first assignment
                assignment_eliminated_variable: dict[str, tp.Any] = assignment_list[0]
                assignment_eliminated_variable = (
                    assignment_eliminated_variable.fromkeys(
                        [eliminated_variable],
                        assignment_eliminated_variable[eliminated_variable],
                    )
                )

                assignment.update(assignment_eliminated_variable)
            else:
                # Extract the slice of the factor consistent with current assignments
                reduced_factor = factor.extract(assignment)

                # Find best assignment for current variable given previous assignments
                if not reduced_factor.empty():
                    assignment_list, value = operation(reduced_factor)

                    assignment_eliminated_variable = assignment_list[0]
                    assignment_eliminated_variable = (
                        assignment_eliminated_variable.fromkeys(
                            [eliminated_variable],
                            assignment_eliminated_variable[eliminated_variable],
                        )
                    )

                    assignment.update(assignment_eliminated_variable)

        return assignment

    ###########################################################
    ## Marginal MAP Inference
    ###########################################################

    def M_MAP(
        self,
        targets: tp.Optional[tp.Iterable[int | str]] = None,
        evidence: tp.Optional[tp.Mapping[int | str, tp.Any]] = None,
        projection_operation: ProjectionOperation = ProjectionOperation.MAX,
        assignment_operation: AssignmentOperation = AssignmentOperation.ARGMAX,
        verbose: bool = True,
    ) -> tuple[dict[str, tp.Any], np.typing.NDArray | float]:
        """
        Perform Marginal MAP inference on the MRF.

        It aims to find the most likely assignment o a subset of variables, marginalizing over the rest.

        Args:
            targets[Y] (list[int | str], optional): The target variables for which to compute the marginal MAP.
                If None, all variables are considered.
            evidence[E=e] (dict[int | str, tp.Any], optional): The observed evidence in the MRF.
                If None, no evidence is considered.
            projection_operation (callable, optional): The operation to use for marginalization
                over the target variables. Defaults to gum.Tensor.maxOut.
            assignment_operation (callable, optional): The operation to use for finding the MAP assignment
                over the target variables. Defaults to gum.Tensor.argmax.
            verbose (bool, optional): If True, print detailed information about the inference process.
        Returns:
            tuple[dict[str, tp.Any], np.typing.NDArray | float]: A tuple containing the MAP assignment as a dictionary
                and the corresponding unnormalized probability of the marginal MAP assignment.

        Example:
        Maximum Marginal A Posteriori (MAP) inference on a Markov Random Field (MRF).
        >>> mrf = gum.MarkovRandomField()
        >>> ve = VariableElimination(mrf)
        >>> marginal_map_assignment, marginal_map_value = ve.M_MAP(
        ...     targets=['A', 'B'],
        ...     evidence={'C': 1},
        ... )

        Minimum Marginal A Posteriori (MAP) inference on a Markov Random Field (MRF).
        >>> mrf = gum.MarkovRandomField()
        >>> ve = VariableElimination(mrf)
        >>> marginal_map_assignment, marginal_map_value = ve.M_MAP(
        ...     targets=['A', 'B'],
        ...     evidence={'C': 1},
        ...     target_projection_operation=gum.Tensor.minOut,
        ...     target_assignment_operation=gum.Tensor.argmin
        ... )
        """
        self.init_active_factors_data()
        if targets is not None:
            self.targets: list[int] = ids(self.PGM, targets)
        else:
            raise ValueError(
                "Targets must be provided for Marginal MAP inference. "
                "Use the 'targets' parameter to specify them."
            )

        if evidence is None:
            self.evidence = {}
        else:
            self.evidence: dict[int, tp.Any] = ids_dict(self.PGM, evidence)
            self.potentials_map = self.update_factors_with_evidence(
                factors=self.potentials_map, evidence=self.evidence
            )

        # Given some targets and evidence, we need to compute the set of remaining variables
        # that needs to be eliminated by sum-product
        selexcluded = [
            node
            for node in set(self.PGM.nodes())  # type: ignore
            if node not in self.evidence and node not in self.targets
        ]

        # elimination_ordering = greedy_ordering(self.MRF, minfill_metric)  segmentation fault due to some memory leak in pyagrum
        nodes = self.PGM.nodes()  # type: ignore
        edges = (
            self.PGM.edges()
            if isinstance(self.PGM, gum.MarkovRandomField)
            else self.PGM.arcs()
        )
        elimination_ordering = greedy_ordering(
            nodes=nodes,
            edges=edges,
            metric=minfill_metric,
        )
        sum_product_ordering, max_product_ordering = (
            self.process_elimination_ordering_M_MAP(
                elimination_ordering,
                evidence=self.evidence,
                targets=self.targets,
                excluded=selexcluded,
            )
        )

        logger.info(
            dedent(f"""
        Performing Marginal MAP inference query with the following parameters:
        - Targets: {names(self.PGM, self.targets)}
        - Evidence: {names_dict(self.PGM, self.evidence)}
        - To eliminate: {names(self.PGM, selexcluded)}
        - Sum-Product Elimination Ordering: {names(self.PGM, sum_product_ordering)}
        - Max-Product Elimination Ordering: {names(self.PGM, max_product_ordering)}
        - Target Projection Operation: {projection_operation.__name__}
        - Target Assignment Operation: {assignment_operation.__name__}""")
        ) if verbose else None

        return self.max_sum_product(
            sum_product_ordering=sum_product_ordering,
            max_product_ordering=max_product_ordering,
            projection_operation=projection_operation,
            assignment_operation=assignment_operation,
        )

    def max_sum_product(
        self,
        sum_product_ordering: list[int],
        max_product_ordering: list[int],
        projection_operation: tp.Callable,
        assignment_operation: tp.Callable,
    ):
        """
        Perform the MAX-SUM-PRODUCT inference algorithm on the MRF.

        Args:
            sum_product_ordering (list[int]): The order in which to eliminate variables for the sum-product.
            max_product_ordering (list[int]): The order in which to eliminate variables for the max-product.
            projection_operation (callable): The operation to use for marginalization.
                Defaults to gum.Tensor.maxOut.
            assignment_operation (callable): The operation to use for finding the MAP assignment.
                Defaults to gum.Tensor.argmax.
        Returns:
            tuple[dict[str, tp.Any], np.typing.NDArray | float]: A tuple containing the MAP assignment as a dictionary
                and the corresponding tensor representing the assignment value.
        """
        # Compute the marginal factor by using the sum-product elimination
        marginal_factor_after_marginalization = self.sum_product(sum_product_ordering)

        # Now we need to compute the MAP over the target variables
        # Set the factors to the marginal factor obtained from the sum-product
        domain_ids = frozenset(
            ids(self.PGM, marginal_factor_after_marginalization.names)
        )
        self.potentials_map = {domain_ids: marginal_factor_after_marginalization}

        return self.max_product(
            elimination_ordering=max_product_ordering,
            projection_operation=projection_operation,
            assignment_operation=assignment_operation,
        )

    ############################################################
    ## VE-specific methods, additional Methods
    #############################################################

    def process_elimination_ordering_M_MAP(
        self,
        ordering: list[int],
        targets: list[int],
        evidence: dict[int, tp.Any],
        excluded: list[int],
    ) -> tuple[list[int], list[int]]:
        """
        Process the elimination ordering for Marginal MAP inference.
        This method modifies the initial elimination ordering to ensure that:
        - Target variables are eliminated using the max-product elimination.
        - Variables in the excluded list are eliminated using the sum-product elimination.
        - Evidence variables are removed from the ordering.

        Args:
            ordering (list[int]): The initial elimination ordering.
            evidence (dict[int | str, tp.Any]): The observed evidence in the MRF.
            targets (list[int | str]): The target variables for which to compute the MAP.

        Returns:
            list[int]: The processed elimination ordering.
        """
        if evidence is None:
            evidence = {}
        if targets is None:
            targets = []
        if excluded is None:
            excluded = []

        ordering = ordering.copy()

        # Remove evidence variables from the ordering
        ordering = [node for node in ordering if node not in evidence]

        # Target variables should follow the max-product elimination
        target_set = set(targets)
        max_product_ordering = [node for node in ordering if node in target_set]

        # Variables that are in the excluded list should follow the sum-product elimination
        excluded_set = set(excluded)
        sum_product_ordering = [node for node in ordering if node in excluded_set]

        return sum_product_ordering, max_product_ordering

    def projection_product_elimination_step(
        self,
        variable: int,
        factors: dict[frozenset[int], gum.Tensor],
        projection_operation: tp.Callable[
            [gum.Tensor, tp.Any],
            gum.Tensor,
        ],
    ) -> tuple[dict[frozenset[int], gum.Tensor], gum.Tensor]:
        """
        Perform the OP-product elimination for a given variable.

        This method performs the following steps:
        1. Identify and gather all factors that contain the variable to be eliminated.
        2. Remove these factors from the active factors.
        3. Combine the gathered factors using the specified combination operation.
        4. Project the combined factor to eliminate the variable using the specified projection operation.
        5. Add the resulting factor back to the active factors.
        6. Return the updated factors and the combined factor before projection.

        Parameters:
        variable (int): The variable to eliminate.
        factors (dict[frozenset[int], gum.Tensor]): The factors to consider for elimination.
        projection_operation (callable): The operation to use for marginalization. Default is gum.Tensor.sumOut.

        Returns:
        tuple[dict[frozenset[int], gum.Tensor], gum.Tensor]
            A tuple containing the updated factors and the resulting tensor after combination but before the projection.
        """
        factors = factors.copy()
        var_name = self.PGM.variable(variable).name()
        relevant_factors = {
            dom: factor for dom, factor in factors.items() if variable in dom
        }

        for dom in relevant_factors.keys():
            del factors[dom]

        if not relevant_factors:
            logger.info("No factors to combine")
            return factors, gum.Tensor()

        intermediate_phi = reduce(
            gum.Tensor.__mul__,
            relevant_factors.values(),
            gum.Tensor(),
        )
        projected_phi = projection_operation(intermediate_phi, var_name)

        # Do we have to normalize the projected factor?
        # NOTE: ask Alessandro
        # projected_phi = (
        #     projected_phi.normalize() if len(projected_phi.names) > 0 else projected_phi
        # )

        # We add the factor after the marginalization, note that if a factor has the same
        # domain as an already existing factor, it will be multiplied with the existing one.
        # This is still correct since anyway we will multiply when producing the intermediate factor.
        factors = self.add_factor(projected_phi, factors)

        return factors, intermediate_phi

    def add_factor(
        self, factor: gum.Tensor, factors: dict[frozenset[int], gum.Tensor]
    ) -> dict[frozenset[int], gum.Tensor]:
        """
        Add a factor to the active factors.

        If a factor with the same domain already exists, it will be multiplied with the existing one.
        If not, it will be added as a new factor.

        Args:
            factor (gum.Tensor): The factor to be added.
            factors (dict[frozenset[int], gum.Tensor]): The current active factors.
        Returns:

        """
        domain = frozenset(factor.names)
        domain_ids = frozenset(ids(self.PGM, domain))
        if domain_ids in factors:
            factors[domain_ids] *= factor
        else:
            factors[domain_ids] = factor

        return factors

    def init_active_factors_data(self):
        """
        Initialize the active factors data for the PGM.
        """
        if isinstance(self.PGM, gum.BayesNet):
            self.potentials_map = {}

            for node in self.PGM.nodes():
                variable = self.PGM.variable(node)
                cpt = self.PGM.cpt(node)

                factor_copy = gum.Tensor()
                for var in cpt.names:
                    factor_copy.add(self.PGM.variable(var))

                factor_copy[:] = cpt.toarray()

                self.potentials_map[frozenset(ids(self.PGM, factor_copy.names))] = (
                    factor_copy
                )
        elif isinstance(self.PGM, gum.MarkovRandomField):
            self.potentials_map = {}
            factors_domain = self.PGM.factors()

            # Do not ask me why, but this is useful to avoid a segmentation fault
            # we avoid sharing references with pyagrum and consequently with its C++ backend
            for domain in factors_domain:
                og_factor = self.PGM.factor(domain)
                factor_copy = gum.Tensor()
                for var in og_factor.names:
                    factor_copy.add(self.PGM.variable(var))

                factor_copy[:] = og_factor.toarray()
                self.potentials_map[frozenset(domain)] = factor_copy

    def reset(self):
        """
        Reset the Variable Elimination instance to its initial state.
        This method clears the factors, targets, evidence, and elimination ordering,
        allowing for a fresh start with the same MRF.
        """
        self.potentials_map = {}
        self.targets = []
        self.evidence = {}
        self.elimination_ordering = []

    def update_factors_with_evidence(
        self, factors: dict[frozenset[int], gum.Tensor], evidence: dict[int, tp.Any]
    ) -> dict[frozenset[int], gum.Tensor]:
        """
        Process the evidence by slicing the factors to remove the evidence variables.

        This method updates the active factors by removing the variables that are observed
        in the evidence. It slices the factors to keep only the variables that are not
        observed in the evidence, effectively reducing the domain of the factors.

        Args:
            factors (dict[frozenset[int], gum.Tensor]): The current active factors.
            evidence (dict[int, tp.Any]): The observed evidence in the MRF.

        Returns:
            dict[frozenset[int], gum.Tensor]: The updated factors after processing the evidence.
        """
        updated_factors = factors.copy()

        for evs_id, evs_value in evidence.items():
            evs_varname = self.PGM.variable(evs_id).name()

            factors_to_update = {
                domain: factor
                for domain, factor in updated_factors.items()
                if evs_id in domain
            }

            for domain in factors_to_update:
                factor = updated_factors.pop(domain)

                if len(domain) == 1:
                    # If the factor is a single variable, we can ignore it
                    # as it is now observed and does not contribute to the inference.
                    continue

                sliced = factor.extract({evs_varname: evs_value})

                new_domain = frozenset(
                    self.PGM.idFromName(name) for name in sliced.names
                )

                if new_domain:
                    updated_factors = self.add_factor(sliced, updated_factors)
        return updated_factors
