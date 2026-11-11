"""ASAP Scheduler for Heuristic Mapping"""

from __future__ import annotations
from typing import Dict, List, Optional

from cgrasolve.graph.dfg import DFG, DFGEdge, DFGNode
from cgrasolve.schedules import LatencySpecification
from cgrasolve.algorithms.scheduler.types import (
    ModuloSchedule,
    SchedulingAttempt,
    SchedulingFailureReason,
    SchedulingResult,
)


class ASAPScheduler:
    """ASAP Scheduler for DFG"""

    # Shwet TODO: Operation latencies are not used in the ASAP scheduler.

    def __init__(self, dfg: DFG, latency_spec: LatencySpecification, fixed_schedule: Dict[DFGNode, int] = None, debug: bool = False) -> None:
        """Initialize the ASAP Scheduler

        Args:
            dfg: Data flow graph
            latency_spec: Latency specification
            fixed_schedule: Optional pre-scheduled nodes
            debug: Enable debug printing
        """
        self._asap_schedule = {}
        self._max_latency = 0
        self._latency_spec = latency_spec
        self._dfg = dfg
        self._debug = debug
        self._last_result: Optional[SchedulingResult] = None

        if fixed_schedule is None:
            fixed_schedule = {}

        for node, time in fixed_schedule.items():
            self._asap_schedule[node] = time
            self._max_latency = max(self._max_latency, time)

    def schedule(self) -> int:
        """Schedule the DFG

        Args:
            dfg: Data flow graph
            latency_spec: Latency specification

        Returns:
            The maximum latency of the schedule
        """

        # Set the number of scheduled and iteration for meta-scheduling.
        num_scheduled: int = 0
        num_iterations: int = 0
        num_rescheduled: int = 0
        max_iterations: int = 1000
        max_latency: int = 0

        if self._debug:
            print(f"\n[DEBUG] Starting ASAP scheduling for {len(self._dfg.get_nodes())} nodes")

            # Print loop-back edges
            loop_back_edges = [edge for edge in self._dfg.get_edges() if edge.is_loop_back]
            if loop_back_edges:
                print(f"[DEBUG] Found {len(loop_back_edges)} loop-back edges:")
                for edge in loop_back_edges:
                    print(f"[DEBUG]   {edge.source.id} ({edge.source.operation.value}) -> {edge.destination.id} ({edge.destination.operation.value})")
            else:
                print(f"[DEBUG] No loop-back edges found")

        # Mark all nodes in the DFG as unscheduled with an ASAP time of -1.
        for node in self._dfg.get_nodes():
            if node.asap_time is not None:
                self._asap_schedule[node] = node.asap_time
                num_scheduled += 1
                if self._debug:
                    print(f"[DEBUG] Node {node.id} pre-scheduled at time {node.asap_time}")
            else:
                self._asap_schedule[node] = -1

        if self._debug:
            print(f"[DEBUG] Initial scheduled nodes: {num_scheduled}/{len(self._dfg.get_nodes())}")

        # Iterate until all nodes are scheduled or the maximum number of iterations is reached.
        prev_scheduled = -1
        stall_count = 0

        while num_scheduled < len(self._dfg.get_nodes()) and num_iterations < max_iterations:
            
            nodes_scheduled_this_iter = 0

            for node in self._asap_schedule.keys():

                # Skip already scheduled nodes.
                if self._asap_schedule[node] != -1:
                    continue

                # Schedule current operation.
                latest_start: int = self._schedule_asap_operation(node)

                # Validation and rescheduling if necessary.
                if latest_start != -1:
                    if self._debug:
                        print(f"[DEBUG] Scheduled {node.id} ({node.operation.value}) at time {latest_start}")

                    self._asap_schedule[node] = latest_start
                    num_scheduled += 1
                    nodes_scheduled_this_iter += 1

                    # Verify bounds.
                    rescheduled_ops: List[DFGNode] = self._verify_bounds_asap(node)
                    num_rescheduled += len(rescheduled_ops)

                    if rescheduled_ops and self._debug:
                        print(f"[DEBUG] Rescheduled {len(rescheduled_ops)} ops after scheduling {node.id}")

                    for op in rescheduled_ops:
                        if self._asap_schedule[op] == -1:
                            num_scheduled -= 1

                    # Shwet TODO: Handle alias edges.
                    # In CGRA-ME, alias edges are handled separately from dataflow edges as they don't represent data dependencies.
                    # Alias edges represent operations that affect the same memory location.
                    # They signify that two memory operations must be separated by at least dist + 1 cycles.
                    # In ASAP scheduler, we will need a check here to make sure that for such edges, the dest and source are separated by at least dist + 1 cycles.
                    # Constraint: cycle(load) - cycle(store) > dist

                    # Predicate edges flow through the same scheduling path as
                    # dataflow edges; LatencySpecification falls back from
                    # INPUT_PRED/OUTPUT_PRED to INPUT/OUTPUT timings when the
                    # arch shares the data fabric with predicate signals.

                    max_latency = max(max_latency, self._asap_schedule[node])
                else:
                    if self._debug and num_iterations % 1000 == 0:
                        # Show why node couldn't be scheduled
                        preds = self._dfg.get_predecessors(node.id)
                        unscheduled_preds = [p for p in preds if self._asap_schedule[p] == -1]
                        if unscheduled_preds:
                            pred_names = [f"{p.id}({p.operation.value})" for p in unscheduled_preds[:3]]
                            print(f"[DEBUG] Cannot schedule {node.id} ({node.operation.value}): waiting for {len(unscheduled_preds)} predecessors: {pred_names}")

                # Increment iteration count.
                num_iterations += 1

            # Detect stalls
            if num_scheduled == prev_scheduled:
                stall_count += 1
                if stall_count > 5 and self._debug:
                    print(f"[DEBUG] Stalled at {num_scheduled}/{len(self._dfg.get_nodes())} nodes for {stall_count} iterations")
                    unscheduled = [n for n in self._dfg.get_nodes() if self._asap_schedule[n] == -1]
                    if unscheduled:
                        print(f"[DEBUG] Unscheduled nodes: {[f'{n.id}({n.operation.value})' for n in unscheduled[:5]]}")
                if stall_count > 10:
                    break
            else:
                stall_count = 0

            prev_scheduled = num_scheduled

            if self._debug and nodes_scheduled_this_iter > 0:
                print(f"[DEBUG] Iteration done: {num_scheduled}/{len(self._dfg.get_nodes())} scheduled (iter={num_iterations})")

        # Check if max iterations reached.
        if num_iterations >= max_iterations:
            if self._debug:
                print(f"[DEBUG] Max iterations reached!")
            print(f"✗ Maximum number of iterations reached.")
            print(f"  Scheduled: {num_scheduled}/{len(self._dfg.get_nodes())} nodes")
            print(f"  Iterations: {num_iterations}")
            raise ValueError("Maximum number of iterations reached.")

        if self._debug:
            print(f"[DEBUG] Scheduling complete: {num_scheduled}/{len(self._dfg.get_nodes())} nodes in {num_iterations} iterations")

        unscheduled = [node for node in self._dfg.get_nodes() if self._asap_schedule[node] == -1]
        if unscheduled:
            unscheduled_ids = ", ".join(node.id for node in unscheduled[:8])
            raise ValueError(
                f"Scheduling stalled with {len(unscheduled)} unscheduled nodes: {unscheduled_ids}"
            )

        # The scheduling is complete. Modify the DFG to add the scheduling information.
        for node in self._dfg.get_nodes():
            node.asap_time = self._asap_schedule[node]
            node.scheduled_time = self._asap_schedule[node]

        node_times = {
            node.id: self._asap_schedule[node]
            for node in self._dfg.get_nodes()
        }
        schedule = ModuloSchedule(
            ii=self._dfg.initiation_interval or 1,
            node_times=node_times,
            asap_times=dict(node_times),
            alap_times={},
            mobility={},
            diagnostics={"max_latency": max_latency, "scheduler": "asap"},
        )
        self._last_result = SchedulingResult(
            success=True,
            schedule=schedule,
            attempts=[
                SchedulingAttempt(
                    ii=schedule.ii,
                    success=True,
                    node_times=dict(node_times),
                    diagnostics=dict(schedule.diagnostics),
                )
            ],
        )

        return max_latency

    @property
    def last_result(self) -> Optional[SchedulingResult]:
        """Return the most recent structured scheduling result."""
        return self._last_result

    def _verify_bounds_asap(self, op: DFGNode) -> List[DFGNode]:
        """Verify the bounds of an ASAP operation
        
        Args:
            op: Operation to verify
            asap_times: Dictionary of operation IDs to their ASAP times
            latency_spec: Latency specification
            dfg: Data flow graph
        """

        # Current latest start time.
        latest_start: int = self._asap_schedule[op]

        # Locked operations.
        locked_ops: List[DFGNode] = []

        # Verify upper bound latency for producers.
        for producer in self._dfg.get_predecessors(op.id):

            # wait required.
            incoming_wait: int = latest_start - self._asap_schedule[producer]

            # Verify upper bound latency.
            if incoming_wait > self._latency_spec.get_network_latency_upper(producer.operation, op.operation):
                
                # Add operation to locked operations (as it must not be rescheduled).
                # Reschedule producer.
                locked_ops.append(op)
                self._reschedule_forward(producer, op)

        # Verify upper bound latency for consumers if scheduled.
        for consumer in self._dfg.get_successors(op.id):

            if self._asap_schedule[consumer] != -1:

                # Outgoing wait.
                outgoing_wait: int = self._asap_schedule[consumer] - latest_start

                # Verify upper bound latency.
                if outgoing_wait > self._latency_spec.get_network_latency_upper(op.operation, consumer.operation):
                    
                    # Reschedule consumer.
                    self._reschedule_forward(op, consumer)

        return locked_ops

    def _schedule_asap_operation(self, op: DFGNode) -> int:
        """Schedule an ASAP operation

        Args:
            op: Operation to schedule
            asap_times: Dictionary of operation IDs to their ASAP times
            latency_spec: Latency specification
            dfg: Data flow graph

        Returns:
            The latest start time of the operation
        """

        # Get all the incoming neighbors of the node.
        incoming_edges: List[DFGEdge] = self._dfg.get_incoming_edges(op.id)
        incoming_nodes: List[DFGNode] = [edge.source for edge in incoming_edges]

        # Consumer latest start
        op_latest_start: int = 0

        for producer in incoming_nodes:

            # Skip loop-back edges - these represent values from the previous iteration
            # and are not scheduling dependencies for the first iteration
            # Note: In CGRA-Solve, loop-back edges are annotated when PHI nodes are removed as well.
            # We are still working with the pre-processed DFG, ICMP nodes, and BR nodes are not present.
            # Added a new pre-processing step for scheduling. PHI nodes are preserved. ICMP and BR nodes are removed.
            # Might need to be careful about placing PHI nodes.

            edge = None
            for e in incoming_edges:
                if e.source == producer:
                    edge = e
                    break

            if edge and edge.is_loop_back:
                if self._debug:
                    print(f"[DEBUG]   Skipping loop-back edge from {producer.id} to {op.id}")
                continue

            # Producer must be scheduled before op
            if self._asap_schedule[producer] == - 1:
                if self._debug:
                    print(f"[DEBUG]   Cannot schedule {op.id}: predecessor {producer.id} ({producer.operation.value}) not scheduled")
                return -1

            # Compute producer start time, latency, and network latency to op
            producer_start_time: int = self._asap_schedule[producer]
           
            producer_latency: int = self._latency_spec.get_op_latency(producer.operation)
            lower_bound, upper_bound = self._latency_spec.get_network_latency(producer.operation, op.operation)


            op_latest_start = max(op_latest_start, producer_start_time + producer_latency + lower_bound)

        # Assign the op latest start time to the asap times
        self._asap_schedule[op] = op_latest_start

        # Get all the outgoing edges of the operation.
        outgoing_edges: List[DFGEdge] = self._dfg.get_outgoing_edges(op.id)
        outgoing_nodes: List[DFGNode] = [edge.destination for edge in outgoing_edges]

        # Make sure that the schedules of the outgoing nodes that have already been scheduled are not violated.
        for successor in outgoing_nodes:

            if self._asap_schedule[successor] != -1:

                # Compute current latency between op and successor based on schedule.
                current_latency: int = self._asap_schedule[successor] - self._asap_schedule[op]

                # Make sure latency spec has the needed edge.
                if self._latency_spec.has_network_latency(op.operation, successor.operation):
                    lower_bound, upper_bound = self._latency_spec.get_network_latency(op.operation, successor.operation)
                else:
                    raise ValueError(f"No network latency found for {op.operation} to {successor.operation}")

                # Make sure current latency is not less than required latency.
                if current_latency > upper_bound:
                    
                    locked_ops: List[DFGNode] = []
                    self._reschedule_forward(op, successor, locked_ops)

        return self._asap_schedule[op]

    def _reschedule_forward(
        self,
        op: DFGNode,
        consumer: DFGNode,
        reschedule_ops: Optional[List[DFGNode]] = None,
    ) -> int:
        """Reschedule an operation forward
        
        Args:
            op: Operation to reschedule
            asap_times: Dictionary of operation IDs to their ASAP times
            latency_spec: Latency specification
            dfg: Data flow graph
            reschedule_ops: List of operations to reschedule
        """
        
        if reschedule_ops is None:
            reschedule_ops = []

        num_unscheduled_ops: int = 0

        # Compute the updated start time for the operation.
        updated_start: int = self._asap_schedule[consumer] - self._latency_spec.get_network_latency_upper(op.operation, consumer.operation)

        # Add to asap times.
        self._asap_schedule[op] = updated_start

        # Check min time violation.
        min_allowed_start: int = self._asap_schedule[consumer] - self._latency_spec.get_network_latency_lower(op.operation, consumer.operation)

        if updated_start > min_allowed_start:
            # If consumer is present in reschedule ops, remove it.
            if consumer in reschedule_ops:
                reschedule_ops.remove(consumer)

            # Check if this still needs to be unscheduled.
            num_unscheduled_ops += self._unschedule_asap_operation(op, reschedule_ops)
        
        # Add operation to reschedule ops.
        reschedule_ops.append(op)

        # Now we must also move around other successors of the current operation (other than the consumer).
        successors: List[DFGNode] = self._dfg.get_successors(op.id)
        for successor in successors:
            if successor != consumer and self._asap_schedule[successor] != -1:

                # Check lower bound latency violation as operation moved forward.
                if self._asap_schedule[successor] - self._asap_schedule[op] < self._latency_spec.get_network_latency_lower(op.operation, successor.operation):

                    # Unschedule the successor.
                    num_unscheduled_ops += self._unschedule_asap_operation(successor, reschedule_ops)

        # Now make sure producers don't face upper bound latency violation.
        producers: List[DFGNode] = self._dfg.get_predecessors(op.id)
        for producer in producers:
            if self._asap_schedule[op] - self._asap_schedule[producer] > self._latency_spec.get_network_latency_upper(producer.operation, op.operation):
                    
                    # Unschedule the producer.
                    locked_ops: List[DFGNode] = []
                    locked_ops.append(producer)
                    num_unscheduled_ops += self._reschedule_forward(producer, op, locked_ops)

        return num_unscheduled_ops

    def _unschedule_asap_operation(
        self,
        op: DFGNode,
        reschedule_ops: Optional[List[DFGNode]] = None,
    ) -> int:
        """Unschedule an ASAP operation. Returns the number of unscheduled operations.
        
        Args:
            op: Operation to unschedule
            asap_times: Dictionary of operation IDs to their ASAP times
            latency_spec: Latency specification
            dfg: Data flow graph
            reschedule_ops: List of operations to reschedule
        """
        if reschedule_ops is None:
            reschedule_ops = []

        if self._asap_schedule[op] == -1:
            return 0

        # Operations in reschedule ops are locked.
        if op in reschedule_ops:
            raise ValueError(f"Unable to resolve upper bound latency violation for {op.id}")

        # Now truly unschedule the operation.
        num_unscheduled_ops: int = 0
        self._asap_schedule[op] = -1
        reschedule_ops.append(op)
        num_unscheduled_ops += 1

        # Now unschedule all the successors if present.
        if len(self._dfg.get_outgoing_edges(op.id)) == 0:
            return num_unscheduled_ops

        successors: List[DFGNode] = self._dfg.get_successors(op.id)
        for successor in successors:
            if self._asap_schedule[successor] != -1:
                num_unscheduled_ops += self._unschedule_asap_operation(successor, reschedule_ops)

        return num_unscheduled_ops

    def _creates_cycle(self, producer: DFGNode, consumer: DFGNode) -> bool:
        """Check if producer depends on consumer (creating a cycle).

        Args:
            producer: The potential producer node
            consumer: The potential consumer node (typically a PHI node)

        Returns:
            True if producer depends on consumer (cycle detected), False otherwise
        """
        # Use DFS to check if there's a path from consumer to producer
        visited = set()
        stack = [consumer]

        while stack:
            current = stack.pop()

            if current == producer:
                return True

            if current in visited:
                continue

            visited.add(current)

            # Add all successors to the stack
            for successor in self._dfg.get_successors(current.id):
                if successor not in visited:
                    stack.append(successor)

        return False
