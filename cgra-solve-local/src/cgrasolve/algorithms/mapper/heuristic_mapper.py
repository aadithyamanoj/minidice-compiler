"""
Heuristic Mapper: ASAP Scheduling + Iterative Place-and-Route

This mapper implements a heuristic flow inspired by ClusteredMapper:
1. ASAP Scheduling - assigns operations to time slots
2. Iterative Place-and-Route Loop:
   - AnnealPlacer: spatial placement using simulated annealing
   - PathFinder: routing connections through CGRA interconnect
   - Adaptive temperature adjustment based on acceptance rate
   - Early stopping on convergence
"""

import random
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Set, Tuple, Optional, Any
from collections import Counter, deque

from cgrasolve.graph.dfg import DFG, DFGNode
from cgrasolve.graph.mrrg import MRRG, MRRGNode
from cgrasolve.schedules.latency_spec import LatencySpecification
from cgrasolve.algorithms.scheduler.asap_scheduler import ASAPScheduler
from cgrasolve.algorithms.scheduler.modulo_scheduler import ModuloScheduler
from cgrasolve.algorithms.scheduler.types import (
    ModuloSchedule,
    SchedulingFailureReason,
    SchedulingResult,
    get_node_schedule_time,
    stamp_schedule_on_dfg,
)
from cgrasolve.algorithms.placer.anneal_placer import AnnealPlacer
from cgrasolve.algorithms.router.pathfinder import PathFinder


class HeuristicMapper:
    """
    Heuristic mapper using ASAP scheduling followed by iterative place-and-route.

    The mapper follows this flow:
    1. ASAP Scheduling: Assigns each operation to a time slot
    2. Iterative Loop (until success or max iterations):
       a. Placement: Use simulated annealing to place operations on FUs
       b. Routing: Use PathFinder to route connections
       c. Temperature adjustment: Adapt based on acceptance rate
       d. Convergence check: Stop early if cost plateaus
    """

    def __init__(
        self,
        dfg: DFG,
        mrrg: MRRG,
        latency_spec: LatencySpecification,
        initial_temperature: float = 1000.0,
        max_iterations: int = 1000,
        max_time: Optional[float] = None,
        convergence_window: int = 10,
        convergence_threshold: float = 0.01,
        random_seed: int = 42,
        swap_factor: int = 10,
        precision_penalty_weight: float = 0.1,
        wire_weight: float = 0.5,
        latency_weight: float = 0.5,
        mrc_weight: float = 0.0,
        peu_weight: float = 0.0,
        normalize_costs: bool = True,
        router_max_iterations: int = 70,
        p_growth_rate: float = 1.5,
        h_growth_rate: float = 1.5,
        scheduler_mode: str = "auto",
        target_ii: Optional[int] = None,
        min_ii: Optional[int] = None,
        max_ii: Optional[int] = None,
        fixed_schedule: Optional[Dict[object, int]] = None,
        mrrg_loader: Optional[Callable[[int], Any]] = None,
        warm_start_decay: float = 0.5,
        congestion_historical_decay: float = 0.5,
        routing_convergence_factor: float = 1.0,
        schedule_perturbation: bool = True,
        max_schedule_perturbations: int = 3,
        temp_cool_factor: float = 0.85,
        temp_reheat_factor: float = 1.15,
        temp_gentle_factor: float = 0.95,
        max_temperature_ratio: float = 2.0,
        fixed_placement: Optional[List[Tuple[str, str]]] = None,
        schedule_debug: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the heuristic mapper.

        Args:
            dfg: Data flow graph to map
            mrrg: Modulo routing resource graph (target architecture)
            latency_spec: Operation and network latency specifications
            initial_temperature: Starting temperature for simulated annealing
            max_iterations: Maximum place-and-route iterations
            max_time: Maximum runtime in seconds (None = no limit)
            convergence_window: Number of iterations to check for convergence
            convergence_threshold: Relative cost change threshold for convergence
            random_seed: Random seed for reproducibility
            swap_factor: Number of swaps per temperature (num_ops * swap_factor)
            precision_penalty_weight: Weight for precision mismatch penalty
            wire_weight: Weight for wirelength cost component
            latency_weight: Weight for latency cost component
            mrc_weight: Weight for memory resource cost component
            peu_weight: Weight for PE utilization pressure component
            normalize_costs: Whether to normalize cost components
            router_max_iterations: Max iterations for PathFinder routing
            p_growth_rate: PathFinder present congestion growth rate
            h_growth_rate: PathFinder history congestion growth rate
            warm_start_decay: Decay factor (0.0-1.0) for historical congestion
                costs carried across placement iterations. Lower values forget
                faster; higher values trust prior routing intelligence more.
            congestion_historical_decay: Decay factor applied to historical
                congestion scores when merging into placement feedback.
            routing_convergence_factor: Max routing overuse fluctuation within
                the convergence window before declaring convergence. When > 0,
                convergence requires both placement cost AND routing overuse to
                plateau. Set to 0 for legacy behavior (cost-only convergence).
            schedule_perturbation: Whether to perturb non-fixed schedule times
                when convergence is detected with routing still failing.
            max_schedule_perturbations: Maximum number of schedule perturbation
                rounds before giving up.
            temp_cool_factor: Temperature multiplier when acceptance rate is
                above target (cool down). Must be < 1.0.
            temp_reheat_factor: Temperature multiplier when acceptance rate is
                below target (reheat). Must be > 1.0.
            temp_gentle_factor: Temperature multiplier when acceptance rate is
                near target (gentle cooling). Must be < 1.0.
            max_temperature_ratio: Maximum temperature as a ratio of
                initial_temperature. Prevents unbounded reheat compounding.
            debug: Enable debug output
        """
        self._dfg = dfg
        self._mrrg = mrrg
        self._base_mrrg = mrrg  # original MRRG for slicing (never overwritten)
        self._latency_spec = latency_spec
        self._initial_temperature = initial_temperature
        self._max_iterations = max_iterations
        self._max_time = max_time
        self._convergence_window = convergence_window
        self._convergence_threshold = convergence_threshold
        self._random_seed = random_seed
        self._swap_factor = swap_factor
        self._precision_penalty_weight = precision_penalty_weight
        self._wire_weight = wire_weight
        self._latency_weight = latency_weight
        self._mrc_weight = mrc_weight
        self._peu_weight = peu_weight
        self._normalize_costs = normalize_costs
        self._router_max_iterations = router_max_iterations
        self._p_growth_rate = p_growth_rate
        self._h_growth_rate = h_growth_rate
        self._scheduler_mode = scheduler_mode
        self._target_ii = target_ii
        self._min_ii = min_ii
        self._max_ii = max_ii
        self._mrrg_loader = mrrg_loader
        self._warm_start_decay = warm_start_decay
        self._congestion_historical_decay = congestion_historical_decay
        self._routing_convergence_factor = routing_convergence_factor
        self._schedule_perturbation = schedule_perturbation
        self._max_schedule_perturbations = max_schedule_perturbations
        self._temp_cool_factor = temp_cool_factor
        self._temp_reheat_factor = temp_reheat_factor
        self._temp_gentle_factor = temp_gentle_factor
        self._max_temperature = initial_temperature * max_temperature_ratio
        self._fixed_placement: List[Tuple[str, str]] = list(fixed_placement) if fixed_placement else []
        self._schedule_debug = schedule_debug
        self._debug = debug
        self._rng = random.Random(random_seed)

        # Results
        self._placement: Optional[Dict[DFGNode, MRRGNode]] = None
        self._routing_solution = None
        self._schedule: Optional[ModuloSchedule] = None
        self._last_routing_result = None
        self._scheduling_attempts: List[Dict[str, Any]] = []
        self._iterations_run = 0
        self._final_cost = 0.0
        self._timed_out = False
        self._mapping_start_time: Optional[float] = None
        self._mapping_deadline: Optional[float] = None
        self._user_fixed_schedule = self._normalize_fixed_schedule(fixed_schedule or {})
        self._explicit_fixed_schedule: Set[str] = set(self._user_fixed_schedule.keys())
        for node in self._dfg.get_nodes():
            if node.scheduled_time is not None:
                self._user_fixed_schedule.setdefault(node.id, node.scheduled_time)
            elif node.asap_time is not None:
                self._user_fixed_schedule.setdefault(node.id, node.asap_time)

    def _deadline_exceeded(self) -> bool:
        """Return True when the overall mapping deadline has expired."""
        return self._mapping_deadline is not None and time.time() >= self._mapping_deadline

    def _elapsed_mapping_time(self) -> float:
        """Return the elapsed wall-clock time for the current mapping attempt."""
        if self._mapping_start_time is None:
            return 0.0
        return time.time() - self._mapping_start_time

    def _build_failure_result(
        self,
        *,
        runtime: float,
        error_message: str,
        placement: Optional[Dict[str, str]] = None,
        routes: Optional[Dict[Tuple[str, int], List[List[str]]]] = None,
        route_metadata: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
        routing_error_message: Optional[str] = None,
        routing_failure_summary: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        ii: Optional[int] = None,
        schedule: Optional[Dict[str, Any]] = None,
        attempt_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build a stable failure result for all mapper failure paths."""
        normalized_errors = list(validation_errors or [])
        return {
            'status': 'failed',
            'placement': placement,
            'routes': routes,
            'route_metadata': route_metadata,
            'runtime': runtime,
            'iterations': self._iterations_run,
            'final_cost': self._final_cost,
            'ii': ii,
            'schedule': schedule,
            'attempt_history': list(attempt_history or []),
            'error_message': error_message,
            'routing_error_message': routing_error_message,
            'routing_failure_summary': routing_failure_summary,
            'routing_stats': self._routing_stats_to_dict(),
            'random_seed': self._random_seed,
            'validation': {
                'is_valid': False,
                'errors': normalized_errors,
            },
        }

    def map(self) -> Dict[str, Any]:
        """
        Run the heuristic mapping flow.

        Returns:
            Dictionary with keys:
                - status: 'success' or 'failed'
                - placement: Dict[str, str] mapping DFG node IDs to MRRG node IDs
                - routes: Dict with routing paths
                - runtime: Total mapping time in seconds
                - iterations: Number of place-and-route iterations performed
                - final_cost: Final placement cost
                - error_message: Error message if failed (optional)
        """
        start_time = time.time()
        self._mapping_start_time = start_time
        self._mapping_deadline = (
            start_time + self._max_time if self._max_time is not None else None
        )
        self._scheduling_attempts = []
        self._schedule = None

        # Reset public/internal result state so repeated map() calls on the
        # same mapper instance do not leak stale failures or routes.
        self._placement = None
        self._routing_solution = None
        self._last_routing_result = None
        self._iterations_run = 0
        self._final_cost = 0.0
        self._timed_out = False

        if self._debug:
            print(f"Random seed: {self._random_seed}")

        try:
            ii_candidates = self._get_ii_candidates()
            last_schedule_error: Optional[str] = None
            if len(ii_candidates) == 1 and self._resolve_scheduler_mode(ii_candidates[0]) == "asap":
                ii = ii_candidates[0]
                self._dfg.initiation_interval = ii
                self._reset_mapping_state()

                if self._debug:
                    print("\n=== Phase 1: ASAP Scheduling ===")

                success = self._run_asap_scheduling()
                if not success:
                    runtime = time.time() - start_time
                    return self._build_failure_result(
                        runtime=runtime,
                        error_message='ASAP scheduling failed',
                        validation_errors=['ASAP scheduling failed'],
                        ii=self._schedule.ii if self._schedule else ii,
                        schedule=self._convert_schedule_to_dict(),
                        attempt_history=list(self._scheduling_attempts),
                    )

                if self._debug:
                    print("\n=== Phase 2: Iterative Place-and-Route ===")

                success = self._run_iterative_place_and_route(ii)
                runtime = time.time() - start_time

                if success:
                    placement_dict = self._convert_placement_to_dict()
                    routing_dict = self._convert_routing_to_dict()
                    route_metadata = self._convert_route_metadata_to_dict()
                    is_valid, validation_errors = self._validate_solution()
                    validation = {
                        'is_valid': is_valid,
                        'errors': validation_errors,
                    }

                    if not is_valid:
                        return self._build_failure_result(
                            runtime=runtime,
                            error_message='Mapping produced invalid solution',
                            placement=placement_dict,
                            routes=routing_dict,
                            route_metadata=route_metadata,
                            routing_error_message=(
                                self._last_routing_result.message
                                if self._last_routing_result else None
                            ),
                            routing_failure_summary=(
                                self._last_routing_result.failure_summary
                                if self._last_routing_result else None
                            ),
                            validation_errors=validation_errors,
                            ii=self._schedule.ii if self._schedule else ii,
                            schedule=self._convert_schedule_to_dict(),
                            attempt_history=list(self._scheduling_attempts),
                        )

                    return {
                        'status': 'success',
                        'placement': placement_dict,
                        'routes': routing_dict,
                        'route_metadata': route_metadata,
                        'runtime': runtime,
                        'iterations': self._iterations_run,
                        'final_cost': self._final_cost,
                        'ii': self._schedule.ii if self._schedule else ii,
                        'schedule': self._convert_schedule_to_dict(),
                        'attempt_history': list(self._scheduling_attempts),
                        'routing_stats': self._routing_stats_to_dict(),
                        'random_seed': self._random_seed,
                        'validation': validation,
                    }
            else:
                for ii in ii_candidates:
                    if self._deadline_exceeded():
                        self._timed_out = True
                        break

                    self._mrrg = self._load_mrrg_for_ii(ii)
                    self._dfg.initiation_interval = ii
                    self._reset_mapping_state()

                    if self._debug:
                        print(f"\n=== Scheduling Attempt (II={ii}) ===")

                    scheduling_result = self._run_scheduler(ii)
                    self._record_scheduling_result(scheduling_result)
                    if not scheduling_result.success or scheduling_result.schedule is None:
                        last_schedule_error = scheduling_result.message or "Scheduling failed"
                        continue

                    self._schedule = scheduling_result.schedule
                    stamp_schedule_on_dfg(self._dfg, self._schedule, update_asap=True)

                    if self._debug:
                        print("\n=== Phase 2: Iterative Place-and-Route ===")

                    success = self._run_iterative_place_and_route(ii)
                    runtime = time.time() - start_time

                    if success:
                        placement_dict = self._convert_placement_to_dict()
                        routing_dict = self._convert_routing_to_dict()
                        route_metadata = self._convert_route_metadata_to_dict()
                        is_valid, validation_errors = self._validate_solution()
                        validation = {
                            'is_valid': is_valid,
                            'errors': validation_errors,
                        }

                        if not is_valid:
                            return self._build_failure_result(
                                runtime=runtime,
                                error_message='Mapping produced invalid solution',
                                placement=placement_dict,
                                routes=routing_dict,
                                route_metadata=route_metadata,
                                routing_error_message=(
                                    self._last_routing_result.message
                                    if self._last_routing_result else None
                                ),
                                routing_failure_summary=(
                                    self._last_routing_result.failure_summary
                                    if self._last_routing_result else None
                                ),
                                validation_errors=validation_errors,
                                ii=self._schedule.ii if self._schedule else None,
                                schedule=self._convert_schedule_to_dict(),
                                attempt_history=list(self._scheduling_attempts),
                            )

                        return {
                            'status': 'success',
                            'placement': placement_dict,
                            'routes': routing_dict,
                            'route_metadata': route_metadata,
                            'runtime': runtime,
                            'iterations': self._iterations_run,
                            'final_cost': self._final_cost,
                            'ii': self._schedule.ii,
                            'schedule': self._convert_schedule_to_dict(),
                            'attempt_history': list(self._scheduling_attempts),
                            'routing_stats': self._routing_stats_to_dict(),
                            'random_seed': self._random_seed,
                            'validation': validation,
                        }

            runtime = time.time() - start_time
            if self._timed_out:
                error_msg = f'Timeout after {runtime:.2f}s ({self._iterations_run} iterations)'
            elif self._last_routing_result is not None:
                error_msg = f'Failed to route after {self._iterations_run} iterations'
            else:
                error_msg = last_schedule_error or 'Scheduling failed'

            return self._build_failure_result(
                runtime=runtime,
                error_message=error_msg,
                routing_error_message=(
                    self._last_routing_result.message
                    if self._last_routing_result else None
                ),
                routing_failure_summary=(
                    self._last_routing_result.failure_summary
                    if self._last_routing_result else None
                ),
                validation_errors=(
                    list(self._last_routing_result.validation_errors)
                    if self._last_routing_result else []
                ),
                ii=self._schedule.ii if self._schedule else None,
                schedule=self._convert_schedule_to_dict(),
                attempt_history=list(self._scheduling_attempts),
            )

        except Exception as e:
            runtime = time.time() - start_time
            message = f'Mapping error: {str(e)}'
            return self._build_failure_result(
                runtime=runtime,
                error_message=message,
                validation_errors=[message],
                ii=self._schedule.ii if self._schedule else None,
                schedule=self._convert_schedule_to_dict(),
                attempt_history=list(self._scheduling_attempts),
            )

    def _normalize_fixed_schedule(self, fixed_schedule: Dict[object, int]) -> Dict[str, int]:
        """Normalize fixed schedule keys to DFG node IDs."""
        normalized: Dict[str, int] = {}
        for key, time in fixed_schedule.items():
            if isinstance(key, DFGNode):
                normalized[key.id] = time
            else:
                normalized[str(key)] = time
        return normalized

    def _resolve_scheduler_mode(self, ii: int) -> str:
        """Resolve the effective scheduler mode for one II attempt."""
        scheduler_mode = self._scheduler_mode
        if scheduler_mode == "auto":
            scheduler_mode = "modulo" if ii > 1 else "asap"
        return scheduler_mode

    def _run_asap_scheduling(self) -> bool:
        """Run ASAP scheduling using the legacy bool contract expected by runtime tests."""
        fixed_schedule = {
            self._dfg.get_node(node_id): time
            for node_id, time in self._user_fixed_schedule.items()
            if self._dfg.get_node(node_id) is not None
        }

        try:
            scheduler = ASAPScheduler(
                dfg=self._dfg,
                latency_spec=self._latency_spec,
                fixed_schedule=fixed_schedule,
                debug=self._schedule_debug or self._debug,
            )
            max_latency = scheduler.schedule()
            if self._debug:
                print(f"ASAP scheduling completed. Max latency: {max_latency}")

            last_result = getattr(scheduler, "last_result", None)
            if last_result is not None:
                self._schedule = last_result.schedule
                self._record_scheduling_result(last_result)
            return True
        except Exception as exc:
            if self._debug:
                print(f"ASAP scheduling failed: {exc}")
            return False

    def _get_ii_candidates(self) -> List[int]:
        """Return the II search range for the current mapping request.

        When neither ``min_ii`` nor ``max_ii`` is provided, the mapper
        auto-computes MII (from ResMII/RecMII) as the lower bound and
        uses ``mrrg.II`` as the upper bound, searching the full feasible
        range automatically.
        """
        if self._target_ii is not None:
            return [self._target_ii]

        # Auto-compute MII when min_ii not provided.
        if self._min_ii is not None:
            min_ii = self._min_ii
        else:
            from cgrasolve.algorithms.scheduler.ii_analysis import IIAnalyzer
            try:
                analyzer = IIAnalyzer(self._dfg, self._mrrg, self._latency_spec)
                res_mii, _ = analyzer.compute_res_mii()
                rec_mii, _ = analyzer.compute_rec_mii()
                min_ii = max(res_mii, rec_mii, 1)
            except Exception:
                min_ii = self._dfg.initiation_interval or 1

        # Default max_ii to base mrrg.II (full search range).
        base_ii = self._base_mrrg.II
        max_ii = self._max_ii if self._max_ii is not None else base_ii
        # Cap at base mrrg.II when no loader available and user didn't
        # explicitly request a higher max_ii.
        if self._mrrg_loader is None and self._max_ii is None and max_ii > base_ii:
            max_ii = base_ii
        if max_ii < min_ii:
            raise ValueError(f"max_ii ({max_ii}) must be >= min_ii ({min_ii})")
        return list(range(min_ii, max_ii + 1))

    def _load_mrrg_for_ii(self, ii: int) -> MRRG:
        """Resolve the MRRG to use for one II attempt.

        Priority:
        1. If an ``mrrg_loader`` is provided, use it (advanced override).
        2. Otherwise, slice the base MRRG to *ii* cycles via
           ``MRRG.slice_to_ii``.  This produces a right-sized graph
           without needing separate per-II architecture directories.
        """
        if self._mrrg_loader is not None:
            loaded = self._mrrg_loader(ii)
            if isinstance(loaded, MRRG):
                return loaded
            if hasattr(loaded, "mrrg"):
                return loaded.mrrg
            raise TypeError(
                f"Unsupported mrrg_loader return type: {type(loaded).__name__}"
            )
        if ii <= self._base_mrrg.II:
            return self._base_mrrg.slice_to_ii(ii)
        raise ValueError(
            f"Cannot search II={ii}; base MRRG II is {self._base_mrrg.II} "
            f"and no mrrg_loader was provided"
        )

    def _reset_mapping_state(self) -> None:
        """Clear state between II attempts while preserving user-fixed schedule points."""
        self._dfg.clear_scheduling_info()
        self._placement = None
        self._routing_solution = None
        self._last_routing_result = None
        self._schedule = None
        self._iterations_run = 0
        self._final_cost = 0.0
        self._timed_out = False
        for node in self._dfg.get_nodes():
            if node.id in self._user_fixed_schedule:
                node.scheduled_time = self._user_fixed_schedule[node.id]
                node.asap_time = self._user_fixed_schedule[node.id]

    def _run_scheduler(self, ii: int) -> SchedulingResult:
        """Run the configured scheduler for one II attempt."""
        scheduler_mode = self._resolve_scheduler_mode(ii)

        fixed_schedule = dict(self._user_fixed_schedule)
        if scheduler_mode == "asap":
            try:
                scheduler = ASAPScheduler(
                    dfg=self._dfg,
                    latency_spec=self._latency_spec,
                    fixed_schedule={
                        self._dfg.get_node(node_id): time
                        for node_id, time in fixed_schedule.items()
                        if self._dfg.get_node(node_id) is not None
                    },
                    debug=self._schedule_debug or self._debug,
                )
                scheduler.schedule()
                if scheduler.last_result is None:
                    raise RuntimeError("ASAP scheduler did not produce a structured result")
                return scheduler.last_result
            except Exception as exc:
                return SchedulingResult(
                    success=False,
                    failure_reason=SchedulingFailureReason.SEARCH_EXHAUSTED,
                    message=str(exc),
                )

        scheduler = ModuloScheduler(
            dfg=self._dfg,
            mrrg=self._mrrg,
            latency_spec=self._latency_spec,
            fixed_schedule=fixed_schedule,
            debug=self._schedule_debug or self._debug,
        )
        return scheduler.schedule(ii=ii)

    def _record_scheduling_result(self, scheduling_result: SchedulingResult) -> None:
        """Append the structured scheduling result to the mapper attempt log."""
        for attempt in scheduling_result.attempts:
            self._scheduling_attempts.append(
                {
                    "ii": attempt.ii,
                    "success": attempt.success,
                    "failure_reason": (
                        attempt.failure_reason.value
                        if attempt.failure_reason is not None
                        else None
                    ),
                    "message": attempt.message,
                    "diagnostics": dict(attempt.diagnostics),
                }
            )

    def _run_iterative_place_and_route(self, target_ii: int) -> bool:
        """
        Run iterative place-and-route loop with adaptive temperature.

        Features:
        - Incremental O(degree) placement cost via AnnealPlacer cache (B1)
        - Routing congestion feedback injected into placement cost (B2)
        - Router warm-starting with historical costs from previous iteration (B3)

        Args:
            target_ii: Target initiation interval for this mapping attempt.

        Returns:
            True if routing succeeded, False otherwise
        """
        # Initialize placer
        placer = AnnealPlacer(
            num_rows=self._mrrg.rows,
            num_cols=self._mrrg.cols,
            dfg=self._dfg,
            latency_spec=self._latency_spec,
            mrrg=self._mrrg,
            fixed_placement=self._fixed_placement,
            random_seed=self._random_seed,
            swap_factor=self._swap_factor,
            target_ii=target_ii,
            precision_penalty_weight=self._precision_penalty_weight,
            wire_weight=self._wire_weight,
            latency_weight=self._latency_weight,
            mrc_weight=self._mrc_weight,
            peu_weight=self._peu_weight,
            normalize_costs=self._normalize_costs,
        )

        # Set initial placement
        placer.set_initial_placement()

        # Diagnostic: report xbar channel filter results
        if self._debug:
            xbar_counts = placer._fu_xbar_channel_counts
            print(f"[diag] xbar channel counts computed for {len(xbar_counts)} FU nodes")
            constrained = placer._node_constrained_fus
            if constrained:
                for node_id, candidates in constrained.items():
                    print(f"[diag] constrained op '{node_id}': {len(candidates)} valid FUs")
            else:
                print("[diag] no constrained ops (no ops with >=2 fixed iopad inputs found)")

        # Initialize temperature and convergence tracking
        temperature = self._initial_temperature
        cost_history = deque(maxlen=self._convergence_window)
        overuse_history: deque = deque(maxlen=self._convergence_window)

        # Warm-start state for router (B3) — preserved across iterations
        warm_start_costs: Optional[Dict[str, float]] = None

        # Schedule perturbation tracking (B4)
        perturbed_node_ids: Set[str] = set()
        schedule_perturbation_count = 0

        # Iterative loop
        for iteration in range(self._max_iterations):
            self._iterations_run = iteration + 1

            # Check time limit
            if self._deadline_exceeded():
                if self._debug:
                    elapsed_time = self._elapsed_mapping_time()
                    print(f"\n✗ Time limit exceeded ({elapsed_time:.2f}s >= {self._max_time}s)")
                self._timed_out = True
                return False

            if self._debug:
                print(f"\n--- Iteration {self._iterations_run} (T={temperature:.2f}) ---")

            # Run annealing at current temperature
            anneal_start = time.time()
            placer.anneal(initial_temperature=temperature, deadline=self._mapping_deadline)
            anneal_time = time.time() - anneal_start
            self._placement = placer._op_node_placement_map.copy()
            # Fold reserved/fixed placements into the placement passed to the router.
            for dfg_id, mrrg_id in placer._reserved_dfg_node_id_to_fu_node_id.items():
                dfg_node = self._dfg.get_node(dfg_id)
                mrrg_node = self._mrrg.get_node(mrrg_id)
                if dfg_node is not None and mrrg_node is not None:
                    self._placement[dfg_node] = mrrg_node
            self._final_cost = placer.get_total_cost()
            acceptance_rate = getattr(placer, "_acceptance_rate", 0.35)

            if placer._timed_out:
                self._timed_out = True
                return False

            if self._debug:
                print(f"Placement cost: {self._final_cost:.2f}")
                print(
                    f"Placement stats: ops={len(self._placement)}, "
                    f"acceptance={acceptance_rate:.3f}, anneal_time={anneal_time:.2f}s"
                )
                for dfg_node, mrrg_node in sorted(self._placement.items(), key=lambda x: x[0].id):
                    print(f"  {dfg_node.id:<12} -> {mrrg_node.id}")

            if self._deadline_exceeded():
                self._timed_out = True
                return False

            # Attempt to route with current placement (B3: warm-start)
            router = PathFinder(
                dfg=self._dfg,
                mrrg=self._mrrg,
                placement=self._placement,
                p_growth_rate=self._p_growth_rate,
                h_growth_rate=self._h_growth_rate,
                max_iterations=self._router_max_iterations,
                debug=self._debug,
                deadline=self._mapping_deadline,
                target_ii=target_ii,
                warm_start_historical_costs=warm_start_costs,
                warm_start_decay=self._warm_start_decay,
            )

            route_start = time.time()
            routing_result = router.route()
            route_time = time.time() - route_start
            self._last_routing_result = routing_result

            # Preserve historical costs for next iteration (B3)
            warm_start_costs = router.get_historical_costs()

            if routing_result.timed_out:
                self._timed_out = True
                return False

            if routing_result.success:
                # Success! Save routing solution and exit
                self._routing_solution = routing_result.routing_solution

                if self._debug:
                    print(f"\n✓ Routing succeeded at iteration {self._iterations_run}")
                    print(f"Final placement cost: {self._final_cost:.2f}")
                    print(
                        f"Iteration timing: route_time={route_time:.2f}s, "
                        f"elapsed={self._elapsed_mapping_time():.2f}s"
                    )

                return True

            if self._debug:
                print("✗ Routing failed, adjusting temperature and retrying")
                print(
                    f"Iteration timing: route_time={route_time:.2f}s, "
                    f"elapsed={self._elapsed_mapping_time():.2f}s"
                )
                failure_summary = routing_result.failure_summary
                if failure_summary:
                    print(f"Last sink failure: {failure_summary}")
                if routing_result.message:
                    print(f"Routing result: {routing_result.message}")

            # Extract congestion feedback for next placement iteration (B2)
            congestion_map = router.get_congestion_map()
            historical_map = router.get_historical_congestion_map()

            # Merge current overuse + decayed historical congestion
            combined_congestion: Dict[Tuple[int, int], float] = {}
            for coord, score in congestion_map.items():
                combined_congestion[coord] = combined_congestion.get(coord, 0.0) + score
            for coord, score in historical_map.items():
                combined_congestion[coord] = combined_congestion.get(coord, 0.0) + self._congestion_historical_decay * score

            # Weight grows with iteration count (mild at first, stronger as routing keeps failing)
            congestion_weight = min(0.5, 0.05 * (iteration + 1))
            placer.set_congestion_feedback(combined_congestion, congestion_weight)

            if self._debug and combined_congestion:
                print(
                    f"  Congestion feedback: {len(combined_congestion)} locations "
                    f"({len(congestion_map)} current, {len(historical_map)} historical), "
                    f"weight={congestion_weight:.3f}"
                )

            # Track routing overuse for convergence detection (B4)
            overuse = max(
                0,
                routing_result.stats.total_occupancy - routing_result.stats.total_capacity,
            )
            overuse_history.append(overuse)

            # Adaptive temperature adjustment based on acceptance rate
            temperature = self._adjust_temperature(temperature, acceptance_rate)

            # Early stopping if both cost and routing have plateaued
            cost_history.append(self._final_cost)
            if self._check_convergence(cost_history, overuse_history):
                if self._debug:
                    print(f"\nConvergence detected at iteration {self._iterations_run}")
                    print("Placement cost and routing overuse have both plateaued")

                # Attempt schedule perturbation before giving up
                if (
                    self._schedule_perturbation
                    and schedule_perturbation_count < self._max_schedule_perturbations
                ):
                    perturbed = self._try_schedule_perturbation(
                        target_ii, placer, perturbed_node_ids,
                    )
                    if perturbed:
                        schedule_perturbation_count += 1
                        # Reset convergence tracking after perturbation
                        cost_history.clear()
                        overuse_history.clear()
                        if self._debug:
                            print(
                                f"  Schedule perturbation #{schedule_perturbation_count} applied, "
                                f"continuing ({self._max_schedule_perturbations - schedule_perturbation_count} left)"
                            )
                        continue
                break

        # Failed to route after max iterations
        if self._debug:
            print(f"\n✗ Failed to route after {self._iterations_run} iterations")

        return False

    def _adjust_temperature(self, current_temp: float, acceptance_rate: float) -> float:
        """
        Adjust temperature adaptively based on acceptance rate.

        Similar to ClusteredMapper's determineTemperature(10):
        - If acceptance rate is too high, decrease temperature (more selective)
        - If acceptance rate is too low, increase temperature (more exploratory)

        Args:
            current_temp: Current temperature
            acceptance_rate: Acceptance rate from last annealing run (0.0 to 1.0)

        Returns:
            Adjusted temperature
        """
        # Target acceptance rate around 0.3-0.4 (30-40%)
        target_rate = 0.35

        if acceptance_rate > target_rate + 0.1:
            # Too many acceptances - cool down faster
            result = current_temp * self._temp_cool_factor
        elif acceptance_rate < target_rate - 0.1:
            # Too few acceptances - heat up
            result = current_temp * self._temp_reheat_factor
        else:
            # Close to target - gentle cooling
            result = current_temp * self._temp_gentle_factor

        return min(result, self._max_temperature)

    def _check_convergence(
        self,
        cost_history: deque,
        overuse_history: Optional[deque] = None,
    ) -> bool:
        """
        Check if both placement cost and routing overuse have converged.

        Placement cost convergence uses relative variation within the window.
        Routing convergence (B4) checks that overuse spread (max - min) within
        the window is within ``routing_convergence_factor``.  If routing is
        still volatile, convergence is suppressed even when cost has plateaued.

        Args:
            cost_history: Recent placement cost values.
            overuse_history: Recent routing overuse values (optional, B4).

        Returns:
            True if converged, False otherwise.
        """
        if len(cost_history) < self._convergence_window:
            return False

        # 1. Check placement cost plateau (existing logic)
        costs = list(cost_history)
        min_cost = min(costs)
        max_cost = max(costs)

        baseline = max(abs(max_cost), abs(min_cost), 1.0)
        relative_variation = (max_cost - min_cost) / baseline

        cost_converged = relative_variation < self._convergence_threshold
        if not cost_converged:
            return False

        # 2. If routing-aware convergence is disabled, use legacy behavior
        if self._routing_convergence_factor <= 0 or overuse_history is None:
            return True

        # 3. Not enough routing data yet — don't converge prematurely
        if len(overuse_history) < self._convergence_window:
            return False

        # 4. Check if routing overuse has ALSO plateaued
        overuse_vals = list(overuse_history)
        overuse_spread = max(overuse_vals) - min(overuse_vals)

        if overuse_spread > self._routing_convergence_factor:
            if self._debug:
                print(
                    f"  Convergence suppressed: routing overuse spread "
                    f"{overuse_spread:.0f} > factor {self._routing_convergence_factor:.1f}"
                )
            return False

        return True

    def _try_schedule_perturbation(
        self,
        target_ii: int,
        placer: AnnealPlacer,
        perturbed_node_ids: Set[str],
    ) -> bool:
        """Perturb non-fixed schedule times to escape a routing plateau.

        Picks a random non-fixed DFG node and shifts its scheduled time by
        +/-1 (modulo II), updating the placer's internal schedule so the
        next placement iteration uses the new timing.

        Args:
            target_ii: Current initiation interval.
            placer: The active AnnealPlacer instance.
            perturbed_node_ids: Set of node IDs already perturbed (updated in-place).

        Returns:
            True if a perturbation was applied, False if no eligible node found.
        """
        candidates = [
            n for n in self._dfg.get_nodes()
            if n.id not in self._explicit_fixed_schedule
            and n.id not in perturbed_node_ids
        ]
        if not candidates:
            return False

        self._rng.shuffle(candidates)

        for node in candidates:
            current_time = get_node_schedule_time(node)
            if current_time is None:
                continue

            for delta in [1, -1]:
                new_time = current_time + delta
                if new_time < 0:
                    continue
                try:
                    valid = self._is_valid_schedule_shift(
                        node, new_time, target_ii, placer
                    )
                except Exception:
                    continue
                if valid:
                    node.scheduled_time = new_time
                    node.asap_time = new_time
                    self._user_fixed_schedule[node.id] = new_time
                    perturbed_node_ids.add(node.id)

                    # Relocate placement to valid FU at new cycle
                    try:
                        placer.clear_op_placement(node)
                        placer.update_op_placement(
                            node, placer.get_random_unoccupied_fu(node)
                        )
                    except Exception:
                        pass  # stub placers may not support this

                    if node.mobility is not None and node.mobility > 0:
                        node.mobility -= 1

                    if self._debug:
                        print(
                            f"  Perturbed {node.id}: t={current_time} -> t={new_time}"
                        )
                    return True

        return False

    def _is_valid_schedule_shift(
        self,
        node: DFGNode,
        new_time: int,
        ii: int,
        placer: AnnealPlacer,
    ) -> bool:
        """Check whether shifting *node* to *new_time* keeps the schedule legal.

        A shift is valid when none of *node*'s DFG predecessors or successors
        would violate latency ordering under the new time.

        Args:
            node: The DFG node to shift.
            new_time: Proposed new schedule time.
            ii: Current initiation interval.
            placer: Active placer (unused currently, reserved for future checks).

        Returns:
            True if the shift is safe, False otherwise.
        """
        for pred in self._dfg.get_predecessors(node.id):
            pred_time = get_node_schedule_time(pred)
            if pred_time is not None and new_time <= pred_time:
                return False

        for succ in self._dfg.get_successors(node.id):
            succ_time = get_node_schedule_time(succ)
            if succ_time is not None and new_time >= succ_time:
                return False

        # Check FU availability at the target cycle
        new_ctx = new_time % ii
        node_bw = node.get_attribute("bitwidth")
        req_out = node_bw if isinstance(node_bw, int) and node_bw > 0 else None
        # Derive input requirement from incoming edge bitwidths
        incoming = self._dfg.get_incoming_edges(node.id)
        in_bws = [
            e.get_attribute("bitwidth") for e in incoming
            if isinstance(e.get_attribute("bitwidth"), int)
            and e.get_attribute("bitwidth") > 0
        ]
        req_in = max(in_bws) if in_bws else None
        available_fus = placer._temporal_op_type_to_fu_nodes.get(
            (node.operation, new_ctx, req_out, req_in), []
        )
        if not available_fus:
            return False

        # Check resource saturation: count ops already at this (operation, cycle)
        ops_at_slot = sum(
            1 for n in self._dfg.get_nodes()
            if n.id != node.id
            and get_node_schedule_time(n) is not None
            and get_node_schedule_time(n) % ii == new_ctx
            and n.operation == node.operation
        )
        if ops_at_slot >= len(available_fus):
            return False

        return True

    def _convert_placement_to_dict(self) -> Dict[str, str]:
        """
        Convert placement from Dict[DFGNode, MRRGNode] to Dict[str, str].

        Returns:
            Dictionary mapping DFG node IDs to MRRG node IDs
        """
        if self._placement is None:
            return {}

        return {
            dfg_node.id: mrrg_node.id
            for dfg_node, mrrg_node in self._placement.items()
        }

    def _convert_routing_to_dict(self) -> Dict[Tuple[str, int], List[List[str]]]:
        """
        Convert routing solution to dictionary format.

        PathFinder returns: Dict[Tuple[HyperVal, int], RouteBundle]
        We convert to: Dict[Tuple[str, int], List[List[str]]]

        Unfractured: {("n_add", 0): [["node1", "node2", ...]]}
        Fractured:   {("n_add", 0): [["sub0_nodes..."], ["sub1_nodes..."], ...]}

        Returns:
            Dictionary mapping (source_id, dest_index) to list of sub-paths
            (each sub-path is a list of node IDs).
        """
        if self._routing_solution is None:
            return {}

        from cgrasolve.algorithms.router.types import RouteBundle

        routing_dict: Dict[Tuple[str, int], List[List[str]]] = {}

        for (hyperval, dest_idx), bundle in self._routing_solution.items():
            source_id = hyperval.source_id
            routing_dict[(source_id, dest_idx)] = bundle.to_node_id_paths()

        return routing_dict

    def _convert_schedule_to_dict(self) -> Optional[Dict[str, Any]]:
        """Convert the accepted schedule to a plain dictionary."""
        if self._schedule is None:
            return None
        return {
            "ii": self._schedule.ii,
            "node_times": dict(self._schedule.node_times),
            "asap_times": dict(self._schedule.asap_times),
            "alap_times": dict(self._schedule.alap_times),
            "mobility": dict(self._schedule.mobility),
            "diagnostics": dict(self._schedule.diagnostics),
        }

    def _routing_stats_to_dict(self) -> Optional[Dict[str, Any]]:
        """Return the most recent routing stats as a plain dict."""
        if self._last_routing_result is None:
            return None
        stats = asdict(self._last_routing_result.stats)
        stats["utilization_percent"] = self._last_routing_result.stats.utilization_percent
        return stats

    def _convert_route_metadata_to_dict(self) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """Export route-bundle metadata in a plain, FASM-friendly shape."""
        if self._routing_solution is None:
            return {}

        metadata: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for (hyperval, dest_idx), bundle in self._routing_solution.items():
            source_id = hyperval.source_id
            profile = bundle.transport_profile
            metadata[(source_id, dest_idx)] = {
                "paths": bundle.to_node_id_paths(),
                "transport_mode": profile.mode.value,
                "fractured": bundle.fractured,
                "num_chunks": bundle.num_chunks,
                "chunk_indices": list(bundle.chunk_indices),
                "chunk_widths": list(bundle.chunk_widths),
                "chunk_offsets": list(bundle.chunk_offsets),
                "lane_assignments": dict(bundle.lane_assignments),
                "active_pack_nodes": sorted(bundle.active_pack_nodes),
                "active_unpack_nodes": sorted(bundle.active_unpack_nodes),
                "total_bitwidth": profile.total_bitwidth,
                "lane_width": profile.lane_width,
                "lane_index": profile.lane_index,
            }

        return metadata

    def _validate_solution(self) -> Tuple[bool, List[str]]:
        """
        Validate the mapper's current internal solution state for correctness.

        Derives string-based placement/routing dicts from ``self._placement``
        and ``self._routing_solution``, then runs mapper-specific checks
        (placement completeness, operation support, resource conflicts) followed
        by the shared routing validator.

        Returns:
            Tuple of (is_valid: bool, errors: list of error messages)
        """
        placement = self._convert_placement_to_dict()
        routes = self._convert_routing_to_dict()

        errors = []

        if self._debug:
            print("\n" + "="*60)
            print("SOLUTION VALIDATION")
            print("="*60)

        # --- Mapper-specific checks (not in shared validator) ---

        # 1. Check all DFG nodes are placed
        placed_nodes = set(placement.keys())
        all_nodes = set(n.id for n in self._dfg.nodes)
        missing = all_nodes - placed_nodes

        if missing:
            error = f"Missing placements for nodes: {missing}"
            errors.append(error)
            if self._debug:
                print(f"x {error}")
        elif self._debug:
            print(f"v All {len(all_nodes)} DFG nodes placed")

        # 2. Check operation supported on FU
        op_support_errors = []
        for node_id, fu_id in placement.items():
            dfg_node = None
            for n in self._dfg.nodes:
                if n.id == node_id:
                    dfg_node = n
                    break
            if dfg_node is None:
                continue
            mrrg_fu = self._mrrg.get_node(fu_id)
            if mrrg_fu is None:
                continue
            if not mrrg_fu.can_execute(dfg_node.operation):
                error = f"Operation {dfg_node.operation.value} not supported on FU {fu_id}"
                op_support_errors.append(error)
                errors.append(error)

        if op_support_errors:
            if self._debug:
                print(f"x Operation support errors: {len(op_support_errors)}")
                for err in op_support_errors[:5]:
                    print(f"   - {err}")
                if len(op_support_errors) > 5:
                    print(f"   ... and {len(op_support_errors) - 5} more")
        elif self._debug:
            print(f"v All operations supported on placed FUs")

        # 3. Check temporal-class correctness (multi-context invariant)
        if self._mrrg.II > 1:
            temporal_errors = []
            for node_id, fu_id in placement.items():
                dfg_node = None
                for n in self._dfg.nodes:
                    if n.id == node_id:
                        dfg_node = n
                        break
                if dfg_node is None:
                    continue
                mrrg_node = self._mrrg.get_node(fu_id)
                if mrrg_node is None:
                    continue
                scheduled_time = get_node_schedule_time(dfg_node)
                if scheduled_time is None:
                    continue
                expected_ctx = scheduled_time % self._mrrg.II
                actual_ctx = mrrg_node.cycle
                if expected_ctx != actual_ctx:
                    # Only enforce temporal class placement when the MRRG is
                    # actually time-expanded for this FU. Many unit tests build
                    # minimal II>1 graphs with only a single-cycle FU instance
                    # to exercise timing logic; those should not fail purely
                    # because an equivalent FU node at the expected context does
                    # not exist in the MRRG.
                    expected_fu = self._mrrg.get_node_by_cycle(
                        expected_ctx,
                        mrrg_node.get_base_name(),
                    )
                    if expected_fu is None or expected_fu.node_type != mrrg_node.node_type:
                        continue
                    error = (
                        f"Temporal class mismatch: {node_id} scheduled at "
                        f"t={scheduled_time} (context {expected_ctx}) "
                        f"placed on FU at cycle {actual_ctx}"
                    )
                    temporal_errors.append(error)
                    errors.append(error)
            if temporal_errors:
                if self._debug:
                    print(f"x Temporal class errors: {len(temporal_errors)}")
                    for err in temporal_errors[:5]:
                        print(f"   - {err}")
            elif self._debug:
                print("v All placements in correct temporal class")

        # 4. Check no resource conflicts (same FU used twice in same cycle)
        fu_usage = {}
        for node_id, fu_id in placement.items():
            mrrg_node = self._mrrg.get_node(fu_id)
            if mrrg_node is not None:
                key = (mrrg_node.get_base_name(), mrrg_node.cycle)

                if key in fu_usage:
                    error = (
                        f"Resource conflict: {node_id} and {fu_usage[key]} "
                        f"both use {mrrg_node.get_base_name()} at cycle {mrrg_node.cycle}"
                    )
                    errors.append(error)
                    if self._debug:
                        print(f"x {error}")
                else:
                    fu_usage[key] = node_id

        if self._debug and not any("Resource conflict" in e for e in errors):
            print(f"v No resource conflicts ({len(fu_usage)} FU usages)")

        # 4b. Check multi-cycle FU occupancy conflicts (e.g., back-to-back loads)
        II = self._target_ii if self._target_ii is not None else self._mrrg.II
        mc_occ: dict = {}  # base_name -> Counter[cycle]
        mc_placements: dict = {}  # base_name -> [(occupied_set, dfg_node_id)]
        for node_id, fu_id in placement.items():
            mrrg_node = self._mrrg.get_node(fu_id)
            if mrrg_node is None or mrrg_node.latency <= 1:
                continue
            base = mrrg_node.get_base_name()
            occupied = set(
                (mrrg_node.cycle + c) % II for c in range(mrrg_node.latency)
            )
            if base not in mc_placements:
                mc_placements[base] = []
            for prev_occupied, prev_id in mc_placements[base]:
                overlap = occupied & prev_occupied
                if overlap:
                    error = (
                        f"Multi-cycle FU conflict: {node_id} and {prev_id} "
                        f"overlap on {base} at cycle(s) {sorted(overlap)}"
                    )
                    errors.append(error)
                    if self._debug:
                        print(f"x {error}")
            mc_placements[base].append((occupied, node_id))

        if self._debug and not any("Multi-cycle" in e for e in errors):
            mc_count = sum(len(v) for v in mc_placements.values())
            if mc_count > 0:
                print(f"v No multi-cycle FU conflicts ({mc_count} multi-cycle placements)")

        # --- Delegate routing checks to shared validator ---
        from cgrasolve.algorithms.common.routing_validation import validate_routing_solution

        shared_valid, shared_errors = validate_routing_solution(
            dfg=self._dfg,
            mrrg=self._mrrg,
            placement=self._placement,
            routing_solution=self._routing_solution,
            debug=self._debug,
        )
        errors.extend(shared_errors)

        if self._debug:
            print("="*60)
            if len(errors) == 0:
                print("vvv SOLUTION VALIDATION PASSED vvv")
            else:
                print(f"xxx SOLUTION VALIDATION FAILED ({len(errors)} errors) xxx")
            print("="*60)

        return len(errors) == 0, errors
