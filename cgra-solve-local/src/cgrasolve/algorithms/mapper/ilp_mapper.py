"""ILP Mapping algorithm."""

from typing import Protocol, List, Dict, Tuple
import gurobipy as gp
from gurobipy import GRB
import os
import pickle
from pathlib import Path

from cgrasolve.graph.hyperdfg import HyperDFG, HyperVal, HyperNode
from cgrasolve.graph.dfg import DFG, DFGEdge, DFGNode
from cgrasolve.graph.mrrg import MRRG, MRRGNode
from cgrasolve.algorithms.mapper.utility import ILPMapperUtility
from cgrasolve.graph.utils.traversal import bfs
class ILPMapper(Protocol):
    """ILP Mapping algorithm."""

    # -- Version 1 ----------------------------------------------------------------
    @classmethod
    def map_v1(cls, hyperdfg: HyperDFG, mrrg: MRRG) -> Dict:
        """Map a DFG onto an MRRG using ILP.

        Returns:
            dict: Result dictionary with keys:
                - status (str): 'success' or 'failed'
                - placement (dict): {dfg_node_id: fu_node_id} mapping
                - routes (dict): {(val_id, sink_k): [routing_node_ids]} routing paths
                - objective (float): Objective value (if successful)
                - runtime (float): Solver runtime in seconds (if successful)
                - model_status (str): Gurobi model status name
                - validation (dict): Validation results with 'is_valid' (bool) and 'errors' (list) (if successful)
                - reason (str): Failure reason (if failed)
        """

        # Create the ILP model
        model = gp.Model()

        # Set the model parameters
        model.Params.OutputFlag = 1
        model.Params.DisplayInterval = 2
        model.Params.LogFile = "mapping_run.log"
        model.Params.TimeLimit = 1000
        model.Params.MIPGap = 0.2
        model.Params.Threads = os.cpu_count()
        model.Params.SolutionLimit = 1

        # Get nodes and edges
        hdfg_nodes: List[HyperNode] = hyperdfg.get_nodes()
        fu_nodes: List[MRRGNode] = mrrg.get_fu_nodes()
        routing_nodes: List[MRRGNode] = mrrg.get_routing_nodes()
        hypervals: List[HyperVal] = hyperdfg.get_edges()

        # Precompute connectivity maps
        routing_fanout_map: Dict[int, List[int]] = mrrg.create_routing_fanout_map()
        routing_fanin_map: Dict[int, List[int]] = mrrg.create_routing_fanin_map()

        # Grab function nodes connected to routing nodes (ROUTE -> FU)
        # This is used to build the implied placement constraints.
        routing_fu_map: Dict[Tuple[int, int], bool] = {}
        for i, routing_node in enumerate(routing_nodes):
            connected_fus = mrrg.get_sink_fus_connected_to_src_routing_node(routing_node.id)
            for fu in connected_fus:
                try:
                    p = next(idx for idx, node in enumerate(fu_nodes) if node.id == fu.id)
                    routing_fu_map[(p, i)] = True
                except StopIteration:
                    continue

        # Grab routing nodes connected to FU nodes (FU -> ROUTE)
        # This is used to build the initial fanout constraints.
        fu_routing_map: Dict[int, List[int]] = {}
        for p, fu_node in enumerate(fu_nodes):
            connected_routing = mrrg.get_sink_routing_nodes_connected_to_src_fu(fu_node.id)
            routing_indices = []
            for routing_node in connected_routing:
                try:
                    i = next(idx for idx, node in enumerate(routing_nodes) if node.id == routing_node.id)
                    routing_indices.append(i)
                except StopIteration:
                    continue
            if routing_indices:
                fu_routing_map[p] = routing_indices

        # Build the ILP model using helper function
        model, F_pq, R_ij, V_ijk = ILPMapper.build_mapping_ilp_v1(
            model, hdfg_nodes, fu_nodes, routing_nodes, hypervals,
            routing_fanout_map, routing_fanin_map, routing_fu_map, fu_routing_map, mrrg
        )

        # Update model
        model.update()

        # Solve the model
        model.optimize()

        # Print model status
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)

        status_codes = {
            1: "LOADED",
            2: "OPTIMAL",
            3: "INFEASIBLE",
            4: "INF_OR_UNBD",
            5: "UNBOUNDED",
            6: "CUTOFF",
            7: "ITERATION_LIMIT",
            8: "NODE_LIMIT",
            9: "TIME_LIMIT",
            10: "SOLUTION_LIMIT",
            11: "INTERRUPTED",
            12: "NUMERIC",
            13: "SUBOPTIMAL",
            14: "INPROGRESS",
            15: "USER_OBJ_LIMIT"
        }

        status_name = status_codes.get(model.status, f"UNKNOWN({model.status})")
        print(f"Model status: {status_name} (code: {model.status})")

        # Track if we found a solution
        has_solution = False

        if model.status == 2:  # OPTIMAL
            print(f"✓ Optimal solution found!")
            try:
                print(f"  Objective value: {model.ObjVal:.2f}")
                print(f"  Runtime: {model.Runtime:.2f} seconds")
            except:
                pass
            has_solution = True
        elif model.status == 10:  # SOLUTION_LIMIT
            print(f"✓ Solution limit reached")
            if model.SolCount > 0:
                print(f"  Found {model.SolCount} solution(s)")
                try:
                    print(f"  Best objective: {model.ObjVal:.2f}")

                except:
                    pass
                has_solution = True
        elif model.status == 9:  # TIME_LIMIT
            print(f"⏱ Time limit reached")
            if model.SolCount > 0:
                try:
                    print(f"  Best solution found: {model.ObjVal:.2f}")
                    print(f"  MIP gap: {model.MIPGap*100:.2f}%")
                except:
                    pass
                has_solution = True
        elif model.status == 3:  # INFEASIBLE
            print(f"✗ Problem is infeasible - no valid mapping exists")
            print(f"\nComputing IIS (Irreducible Inconsistent Subsystem)...")
            print(f"This will identify the minimal set of conflicting constraints.")

            print("\n❌ Model is INFEASIBLE - Computing IIS...")
            
            # Compute Irreducible Inconsistent Subsystem
            model.computeIIS()
            
            # Write IIS to file
            model.write("model_iis.ilp")
            print("  IIS written to model_iis.ilp")
            
            # Count constraints in IIS by name
            iis_constrs = {}
            for constr in model.getConstrs():
                if constr.IISConstr:
                    name_prefix = constr.ConstrName.split('_')[0] if '_' in constr.ConstrName else constr.ConstrName
                    iis_constrs[name_prefix] = iis_constrs.get(name_prefix, 0) + 1
            
            print("\n  Conflicting constraint groups:")
            for name, count in sorted(iis_constrs.items(), key=lambda x: -x[1]):
                print(f"    {name}: {count} constraints")
            
            # Show a few example conflicting constraints
            print("\n  Example conflicting constraints:")
            count = 0
            for constr in model.getConstrs():
                if constr.IISConstr:
                    print(f"    {constr.ConstrName}")
                    count += 1
                    if count >= 10:
                        break
                    
        elif model.status == 13:  # SUBOPTIMAL
            print(f"⚠ Suboptimal solution found")
            if model.SolCount > 0:
                try:
                    print(f"  Best solution: {model.ObjVal:.2f}")
                except:
                    pass
                has_solution = True

        print(f"\nSolution count: {model.SolCount}")
        print("="*70)

        # Extract and return solution if found
        if has_solution:
            # Extract placement and routing
            placement = ILPMapperUtility.extract_placements_v1(F_pq, hdfg_nodes, fu_nodes)
            routes = ILPMapperUtility.extract_routes_v1(V_ijk, hypervals, routing_nodes, mrrg,
                                                         placement=placement, hdfg_nodes=hdfg_nodes)

            # Print the solution summary
            ILPMapperUtility.print_solution_v1(placement, routes, hdfg_nodes, hypervals)

            # Validate the solution
            is_valid, validation_errors = ILPMapperUtility.validate_solution_v1(
                placement, routes, hdfg_nodes, hypervals, mrrg
            )

            # Safely get objective value and runtime
            try:
                objective = model.ObjVal
            except:
                objective = None

            try:
                runtime = model.Runtime
            except:
                runtime = None

            return {
                'status': 'success',
                'placement': placement,
                'routes': routes,
                'objective': objective,
                'runtime': runtime,
                'model_status': status_name,
                'validation': {
                    'is_valid': is_valid,
                    'errors': validation_errors
                }
            }
        else:
            return {
                'status': 'failed',
                'placement': {},
                'routes': {},
                'model_status': status_name,
                'reason': 'No solution found' if model.status != 3 else 'Infeasible'
            }

    @staticmethod
    def build_mapping_ilp_v1(model: gp.Model, hdfg_nodes: List[HyperNode], fu_nodes: List[MRRGNode], routing_nodes: List[MRRGNode], hypervals: List[HyperVal],
                      routing_fanout_map: Dict[int, List[int]], routing_fanin_map: Dict[int, List[int]], routing_fu_map: Dict[Tuple[int, int], bool], fu_routing_map: Dict[int, List[int]], mrrg: MRRG) -> Tuple[gp.Model, Dict[int, Dict[int, float]], Dict[int, Dict[int, float]], Dict[int, Dict[int, Dict[int, float]]]]:
        """Build full DFG→MRRG ILP with legality, placement, routing, and continuity."""

        num_hdfg_nodes     = len(hdfg_nodes)
        num_fu_nodes      = len(fu_nodes)
        num_routing_nodes = len(routing_nodes)
        num_vals          = len(hypervals)

        # --- Precompute fast ID→index maps ----------------------------------------
        dfg_id2idx = {n.id: i for i, n in enumerate(hdfg_nodes)}
        fu_id2idx  = {n.id: i for i, n in enumerate(fu_nodes)}
        rt_id2idx  = {n.id: i for i, n in enumerate(routing_nodes)}

        # --- Create variables -----------------------------------------------------
        F_pq  = model.addVars(num_fu_nodes, num_hdfg_nodes, vtype=GRB.BINARY, name="F")
        R_ij  = model.addVars(num_routing_nodes, num_vals, vtype=GRB.BINARY, name="R")
        V_ijk = model.addVars(num_routing_nodes, num_vals,
                            max(h.cardinality for h in hypervals),
                            vtype=GRB.BINARY, name="V")

        # --- 1. Operation placement ----------------------------------------------
        print(f"[1/10] Placement ({num_hdfg_nodes})")
        model.addConstrs(
            (gp.quicksum(F_pq[p, q] for p in range(num_fu_nodes)) == 1
            for q in range(num_hdfg_nodes)),
            name="place_once"
        )

        # --- 2. FU exclusivity ----------------------------------------------------
        print(f"[2/10] FU exclusivity ({num_fu_nodes})")
        model.addConstrs(
            (gp.quicksum(F_pq[p, q] for q in range(num_hdfg_nodes)) <= 1
            for p in range(num_fu_nodes)),
            name="fu_unique"
        )

        # --- 3. Legality pruning --------------------------------------------------
        print(f"[3/10] FU legality pruning")
        for p in range(num_fu_nodes):
            for q in range(num_hdfg_nodes):
                if hdfg_nodes[q].operation not in fu_nodes[p].supported_operations:
                    F_pq[p, q].ub = 0

        # --- 4. Routing node exclusivity -----------------------------------------
        print(f"[4/10] Routing exclusivity ({num_routing_nodes})")
        model.addConstrs(
            (gp.quicksum(R_ij[i, j] for j in range(num_vals)) <= 1
            for i in range(num_routing_nodes)),
            name="route_unique"
        )

        # --- 5. Fanout routing continuity ----------------------------------------
        print(f"[5/10] Fanout continuity")

        valid_rfan = []
        for i, fanouts in routing_fanout_map.items():
            # Skip nodes with no fanouts (terminal nodes like FU inputs)
            if not fanouts:
                continue
            
            for j in range(num_vals):
                for k in range(hypervals[j].cardinality):
                    valid_rfan.append((i, j, k))

        print(f"  Generated {len(valid_rfan)} fanout continuity constraints (skipped terminal nodes)")

        model.addConstrs(
            (V_ijk[i, j, k] <= gp.quicksum(V_ijk[m, j, k] for m in routing_fanout_map[i])
            for (i, j, k) in valid_rfan),
            name="fanout"
        )

        # --- 5b. Fanin routing continuity ----------------------------------------
        # For each routing node (except root nodes from FUs), if it's used,
        # at least one of its fanin (predecessor) nodes must also be used.
        # This ensures paths are connected from source to sink.
        print(f"[5b/10] Fanin continuity")

        # Identify root routing nodes (those directly connected from FUs)
        root_routing_indices = set()
        for _, routing_indices in fu_routing_map.items():
            root_routing_indices.update(routing_indices)

        valid_rfanin = []
        for i, fanins in routing_fanin_map.items():
            # Skip root nodes (they get their input from FUs, not other routing nodes)
            if i in root_routing_indices:
                continue
            # Skip nodes with no fanins (shouldn't happen for non-root nodes, but be safe)
            if not fanins:
                continue

            for j in range(num_vals):
                for k in range(hypervals[j].cardinality):
                    valid_rfanin.append((i, j, k))

        print(f"  Generated {len(valid_rfanin)} fanin continuity constraints (skipped {len(root_routing_indices)} root nodes)")

        model.addConstrs(
            (V_ijk[i, j, k] <= gp.quicksum(V_ijk[m, j, k] for m in routing_fanin_map[i])
            for (i, j, k) in valid_rfanin),
            name="fanin"
        )

        # --- 6. Implied placement (routing→FU) -----------------------------------
        # If a terminal routing node is used, the corresponding FU must be placed
        print(f"[6/10] Implied placement")
        valid_impl = []
        for (p, i) in routing_fu_map.keys():
            for j in range(num_vals):
                for k in range(hypervals[j].cardinality):
                    dest_id = hypervals[j].destination_ids[k]
                    q = dfg_id2idx.get(dest_id)
                    if q is None:
                        continue
                    if hdfg_nodes[q].operation in fu_nodes[p].supported_operations:
                        valid_impl.append((p, i, j, k, q))
        model.addConstrs(
            ( V_ijk[i, j, k] <= F_pq[p, q] for (p, i, j, k, q) in valid_impl),
            name="implied_place"
        )


        # --- 6b. Terminal node exclusivity ----------------------------------------
        # Terminal routing nodes (those with no routing fanouts) can only be active
        # if they feed the specific FU where the sink is placed.
        # This prevents activating terminal nodes that don't connect to the actual sink.
        print(f"[6b/10] Terminal node exclusivity")

        # Identify terminal routing nodes (no routing fanouts)
        terminal_routing_indices = set()
        for i, fanouts in routing_fanout_map.items():
            if not fanouts:
                terminal_routing_indices.add(i)

        print(f"  Found {len(terminal_routing_indices)} terminal routing nodes")

        # For each terminal node i, it can only be active for (j,k) if:
        # - Node i feeds some FU p, AND
        # - The destination of val j sink k is placed at FU p
        # Constraint: V[i,j,k] <= sum(F[p,q] for all p that i feeds into, where q is dest of j,k)
        terminal_constrs = []
        for i in terminal_routing_indices:
            # Find which FUs this terminal node feeds
            fus_fed_by_i = [p for (p, ri) in routing_fu_map.keys() if ri == i]
            if not fus_fed_by_i:
                continue

            for j in range(num_vals):
                for k in range(hypervals[j].cardinality):
                    dest_id = hypervals[j].destination_ids[k]
                    q = dfg_id2idx.get(dest_id)
                    if q is None:
                        continue
                    # Build list of valid placements: FU p where i feeds p AND q's op is supported
                    valid_placements = [
                        (p, q) for p in fus_fed_by_i
                        if hdfg_nodes[q].operation in fu_nodes[p].supported_operations
                    ]
                    if valid_placements:
                        terminal_constrs.append(
                            V_ijk[i, j, k] <= gp.quicksum(F_pq[p, q] for (p, q) in valid_placements)
                        )
                    else:
                        # Terminal node i cannot reach destination q - force V[i,j,k] = 0
                        V_ijk[i, j, k].ub = 0

        for idx, constr in enumerate(terminal_constrs):
            model.addConstr(constr, name=f"terminal_excl_{idx}")

        print(f"  Generated {len(terminal_constrs)} terminal exclusivity constraints")

        # --- 6c. No-orphan termination guard -------------------------------------
        print(f"[6c] No-orphan termination")
        # Build constraint list without storing lists in tuples
        orphan_constrs = []
        for (p, i) in routing_fu_map.keys():
            compat_ops = [q for q in range(num_hdfg_nodes)
                        if hdfg_nodes[q].operation in fu_nodes[p].supported_operations]
            if not compat_ops:
                continue
            for j in range(num_vals):
                for k in range(hypervals[j].cardinality):
                    # Create constraint inline without tuple storage
                    orphan_constrs.append(
                        V_ijk[i, j, k] <= gp.quicksum(F_pq[p, q] for q in compat_ops)
                    )
        # Add all constraints at once
        for idx, constr in enumerate(orphan_constrs):
            model.addConstr(constr, name=f"no_orphan_term_{idx}")

        # --- 7. Initial fanout (FU→routing) --------------------------------------
        print(f"[7/10] Initial fanout")
        valid_fan = []
        for j in range(num_vals):
            src_id = hypervals[j].source_id
            if src_id not in dfg_id2idx:
                continue
            q = dfg_id2idx[src_id]
            for p in fu_routing_map:
                if hdfg_nodes[q].operation in fu_nodes[p].supported_operations:
                    valid_fan.append((p, q, j))
                    
        # Exactly one route-start per sink k if q is on p (no source-side duplication).
        model.addConstrs(
            (gp.quicksum(V_ijk[i, j, k] for i in fu_routing_map[p]) == F_pq[p, q]
            for (p, q, j) in valid_fan
            for k in range(hypervals[j].cardinality)),
            name="fanout_init_eq_per_sink"
        )

        # --- 8. Routing resource usage (portable) --------------------------------
        print(f"[8/10] Routing usage")
        model.addConstrs(
            (R_ij[i, j] >= V_ijk[i, j, k]
            for i in range(num_routing_nodes)
            for j in range(num_vals)
            for k in range(hypervals[j].cardinality)),
            name="route_usage"
        )

        # --- 9. Multiplexer input constraints ------------------------------------
        print(f"[9/10] Multiplexer input constraints")
        mux_nodes = [i for i in range(num_routing_nodes)
                    if len(mrrg.get_fan_in_nodes(routing_nodes[i].id)) > 1]
        model.addConstrs(
            (R_ij[i, j] == gp.quicksum(
                R_ij[rt_id2idx[m.id], j]
                for m in mrrg.get_fan_in_nodes(routing_nodes[i].id))
            for i in mux_nodes for j in range(num_vals)),
            name="mux"
        )

        # --- 10. Objective: minimize routing usage -------------------------------
        print(f"[10/10] Objective")
        model.setObjective(
            gp.quicksum(R_ij[i, j] for i in range(num_routing_nodes)
                                    for j in range(num_vals)),
            GRB.MINIMIZE
        )

        print(f"  ✓ Model construction complete: {model.NumConstrs} constraints, "
            f"{model.NumVars} variables.")
        return model, F_pq, R_ij, V_ijk

    # -- Version 2 ----------------------------------------------------------------

    @staticmethod
    def _get_path_map_cache_path(mrrg_name: str, k: int) -> Path:
        """Get the cache file path for path maps.

        Args:
            mrrg_name: Name of the MRRG (e.g., 'adres_1_detailed')
            k: Number of shortest paths

        Returns:
            Path to cache file (e.g., benchmarks/architectures/adres/adres_1_detailed_path_map_k20.pkl)
        """
        # Extract arch_type from mrrg_name (e.g., 'adres' from 'adres_1_detailed')
        arch_type = mrrg_name.split('_')[0]
        cache_dir = Path(f"benchmarks/architectures/{arch_type}")
        return cache_dir / f"{mrrg_name}_path_map_k{k}.pkl"

    @classmethod
    def map_v2(cls, dfg: DFG, mrrg: MRRG) -> Dict:
        """Map a DFG onto an MRRG using ILP.

        Returns:
            dict: Result dictionary with keys:
                - status (str): 'success' or 'failed'
                - placement (dict): {dfg_node_id: fu_node_id} mapping
                - routes (dict): {(val_id, sink_k): [routing_node_ids]} routing paths
                - objective (float): Objective value (if successful)
                - runtime (float): Solver runtime in seconds (if successful)
                - model_status (str): Gurobi model status name
                - validation (dict): Validation results with 'is_valid' (bool) and 'errors' (list) (if successful)
                - reason (str): Failure reason (if failed)
        """
        # Create the ILP model
        model = gp.Model()

        # Set the model parameters
        model.Params.OutputFlag = 1
        model.Params.DisplayInterval = 2
        model.Params.LogFile = "mapping_run.log"
        model.Params.TimeLimit = 600
        model.Params.MIPGap = 0.2
        model.Params.Threads = os.cpu_count()
        model.Params.SolutionLimit = 1

        # Get nodes and edges
        dfg_nodes: List[DFGNode] = dfg.get_nodes()
        fu_nodes: List[MRRGNode] = mrrg.get_fu_nodes()
        routing_nodes: List[MRRGNode] = mrrg.get_routing_nodes()
        dfg_edges: List[DFGEdge] = dfg.get_edges()

        # Precompute connectivity maps
        routing_fanout_map: Dict[str, List[str]] = mrrg.create_routing_fanout_map()

        # Grab function nodes connected to routing nodes (ROUTE -> FU)
        # This is used to build the implied placement constraints.
        routing_fu_map: Dict[Tuple[int, int], bool] = {}
        for i, routing_node in enumerate(routing_nodes):
            connected_fus = mrrg.get_sink_fus_connected_to_src_routing_node(routing_node.id)
            for fu in connected_fus:
                try:
                    p = next(idx for idx, node in enumerate(fu_nodes) if node.id == fu.id)
                    routing_fu_map[(p, i)] = True
                except StopIteration:
                    continue

        # Grab routing nodes connected to FU nodes (FU -> ROUTE)
        # This is used to build the initial fanout constraints.
        fu_routing_map: Dict[int, List[int]] = {}
        for p, fu_node in enumerate(fu_nodes):
            connected_routing = mrrg.get_sink_routing_nodes_connected_to_src_fu(fu_node.id)
            routing_indices = []
            for routing_node in connected_routing:
                try:
                    i = next(idx for idx, node in enumerate(routing_nodes) if node.id == routing_node.id)
                    routing_indices.append(i)
                except StopIteration:
                    continue
            if routing_indices:
                fu_routing_map[p] = routing_indices

        # map of all paths in the MRRG FU space
        import time
        path_start = time.time()

        k = 20
        cache_path = cls._get_path_map_cache_path(mrrg.name, k)

        if cache_path.exists():
            print(f"Loading cached path map from {cache_path}")
            with open(cache_path, 'rb') as f:
                fu_path_map = pickle.load(f)
        else:
            print(f"Computing path map (k={k}) using optimized version...")
            # Use optimized version with subgraph filtering and spatial pruning
            fu_path_map: Dict[Tuple[int, int], List[List[str]]] = mrrg.get_k_shortest_paths_in_mrrg_fu_space(k=k)
            # Save to cache
            print(f"Saving path map to {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(fu_path_map, f)

        path_elapsed = time.time() - path_start

        # Print path elapsed time
        print(f"Path map computation took {path_elapsed:.2f} seconds")

        # Build the ILP model using helper function
        model, F_ou, E_oupv, P_uvq = ILPMapper.build_mapping_ilp_v2(
            model, dfg_nodes, fu_nodes, routing_nodes, dfg_edges,
            routing_fanout_map, routing_fu_map, fu_routing_map, mrrg, fu_path_map
        )

        # Update model
        model.update()

        # Solve the model
        model.optimize()

        # Print model status
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)

        status_codes = {
            1: "LOADED",
            2: "OPTIMAL",
            3: "INFEASIBLE",
            4: "INF_OR_UNBD",
            5: "UNBOUNDED",
            6: "CUTOFF",
            7: "ITERATION_LIMIT",
            8: "NODE_LIMIT",
            9: "TIME_LIMIT",
            10: "SOLUTION_LIMIT",
            11: "INTERRUPTED",
            12: "NUMERIC",
            13: "SUBOPTIMAL",
            14: "INPROGRESS",
            15: "USER_OBJ_LIMIT"
        }

        status_name = status_codes.get(model.status, f"UNKNOWN({model.status})")
        print(f"Model status: {status_name} (code: {model.status})")

        # Track if we found a solution
        has_solution = False

        if model.status == 2:  # OPTIMAL
            print(f"✓ Optimal solution found!")
            try:
                print(f"  Objective value: {model.ObjVal:.2f}")
                print(f"  Runtime: {model.Runtime:.2f} seconds")

                # print the active variables
                print("\n  Active variables:")
                for var in model.getVars():
                    if var.X > 0.5:
                        print(f"    {var.VarName} = {var.X:.2f}")
            except:
                pass
            has_solution = True
        elif model.status == 10:  # SOLUTION_LIMIT
            print(f"✓ Solution limit reached")
            if model.SolCount > 0:
                print(f"  Found {model.SolCount} solution(s)")
                try:
                    print(f"  Best objective: {model.ObjVal:.2f}")

                    # print the active variables
                    print("\n  Active variables:")
                    for var in model.getVars():
                        if var.X > 0.5:
                            print(f"    {var.VarName} = {var.X:.2f}")

                except:
                    pass
                has_solution = True
        elif model.status == 9:  # TIME_LIMIT
            print(f"⏱ Time limit reached")
            if model.SolCount > 0:
                try:
                    print(f"  Best solution found: {model.ObjVal:.2f}")
                    print(f"  MIP gap: {model.MIPGap*100:.2f}%")
                except:
                    pass
                has_solution = True
        elif model.status == 3:  # INFEASIBLE
            print(f"✗ Problem is infeasible - no valid mapping exists")
            print(f"\nComputing IIS (Irreducible Inconsistent Subsystem)...")
            print(f"This will identify the minimal set of conflicting constraints.")

            print("\n❌ Model is INFEASIBLE - Computing IIS...")
            
            # Compute Irreducible Inconsistent Subsystem
            model.computeIIS()
            
            # Write IIS to file
            model.write("model_iis.ilp")
            print("  IIS written to model_iis.ilp")
                        
            # Count constraints in IIS by name
            iis_constrs = {}
            for constr in model.getConstrs():
                if constr.IISConstr:
                    name_prefix = constr.ConstrName.split('_')[0] if '_' in constr.ConstrName else constr.ConstrName
                    iis_constrs[name_prefix] = iis_constrs.get(name_prefix, 0) + 1
            
            # Print all conflicting constraints
            print("\n  All conflicting constraints:")
            for constr in model.getConstrs():
                if constr.IISConstr:
                    print(f"    {constr.ConstrName}")

            print("\n  Conflicting constraint groups:")
            for name, count in sorted(iis_constrs.items(), key=lambda x: -x[1]):
                print(f"    {name}: {count} constraints")
            
            # Show a few example conflicting constraints
            print("\n  Example conflicting constraints:")
            count = 0
            for constr in model.getConstrs():
                if constr.IISConstr:
                    print(f"    {constr.ConstrName}")
                    count += 1
                    if count >= 10:
                        break
                    
        elif model.status == 13:  # SUBOPTIMAL
            print(f"⚠ Suboptimal solution found")
            if model.SolCount > 0:
                try:
                    print(f"  Best solution: {model.ObjVal:.2f}")
                except:
                    pass
                has_solution = True

        print(f"\nSolution count: {model.SolCount}")
        print("="*70)

        # Extract and return solution if found
        if has_solution:
            # Extract placement and routing
            placement = ILPMapperUtility.extract_placements_v2(F_ou, dfg_nodes, fu_nodes)
            routes = ILPMapperUtility.extract_routes_v2(
                E_oupv, P_uvq, dfg_nodes, dfg_edges, fu_nodes, fu_path_map, mrrg,
                placement=placement
            )

            # Print the solution summary
            ILPMapperUtility.print_solution_v2(placement, routes, dfg_nodes, dfg_edges)

            # Validate the solution
            is_valid, validation_errors = ILPMapperUtility.validate_solution_v2(
                placement, routes, dfg_nodes, dfg_edges, mrrg
            )

            # Safely get objective value and runtime
            try:
                objective = model.ObjVal
            except:
                objective = None

            try:
                runtime = model.Runtime
            except:
                runtime = None

            return {
                'status': 'success',
                'placement': placement,
                'routes': routes,
                'objective': objective,
                'runtime': runtime,
                'model_status': status_name,
                'validation': {
                    'is_valid': is_valid,
                    'errors': validation_errors
                }
            }
        else:
            return {
                'status': 'failed',
                'placement': {},
                'routes': {},
                'model_status': status_name,
                'reason': 'No solution found' if model.status != 3 else 'Infeasible'
            }

    @staticmethod
    def build_mapping_ilp_v2(model: gp.Model, dfg_nodes: List[DFGNode], fu_nodes: List[MRRGNode], routing_nodes: List[MRRGNode], dfg_edges: List[DFGEdge],
                      routing_fanout_map: Dict[int, List[int]], routing_fu_map: Dict[Tuple[int, int], bool], fu_routing_map: Dict[int, List[int]], mrrg: MRRG, path_map: Dict[Tuple[int, int], List[List[str]]]):
        """Build full DFG→MRRG ILP with legality, placement, routing, and continuity.

        Args:
            path_map: Dictionary mapping (FU_index_i, FU_index_j) to list of paths,
                     where each path is a list of node IDs (strings), not MRRGNode objects.
        """

        num_dfg_nodes     = len(dfg_nodes)
        num_fu_nodes      = len(fu_nodes)
        num_routing_nodes = len(routing_nodes)
        num_dfg_edges         = len(dfg_edges)

        # --- Precompute fast ID→index maps ----------------------------------------
        dfg_id2idx = {n.id: i for i, n in enumerate(dfg_nodes)}
        fu_id2idx  = {n.id: i for i, n in enumerate(fu_nodes)}
        rt_id2idx  = {n.id: i for i, n in enumerate(routing_nodes)}

        # --- Create variables -----------------------------------------------------
        # Placement: DFG node o → FU u
        F_ou = model.addVars(
            num_dfg_nodes, num_fu_nodes,  # (o, u)
            vtype=GRB.BINARY,
            name="F_ou"
        )
        
        # Path selection: path q connecting FU pair (u, v)
        P_uvq = model.addVars(
            ((u, v, q)
            for (u, v), paths in path_map.items()
            for q in range(len(paths))),
            vtype=GRB.BINARY,
            name="P_uvq"
        )

        # Edge mapping: DFG edge (o → p) assigned to FU pair (u → v)
        E_oupv = model.addVars(
            num_dfg_nodes,  # o: source DFG node index
            num_fu_nodes,   # u: source FU index
            num_dfg_nodes,  # p: destination DFG node index
            num_fu_nodes,   # v: destination FU index
            vtype=GRB.BINARY,
            name="E_oupv"
        )

        # Bound all E variables to 0 initially
        for var in E_oupv.values():
            var.ub = 0

        # Helper map to get compatible FUs for each DFG node.
        comp_map = {
            o: [fu_id2idx[f.id] for f in mrrg.get_compatible_fus(dfg_nodes[o].operation)]
            for o in range(num_dfg_nodes)
        }

        # Compute the valid (u,v) pairs per (o,p) once.
        valid_pairs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for e in range(num_dfg_edges):
            o = dfg_id2idx[dfg_edges[e].source.id]
            p = dfg_id2idx[dfg_edges[e].destination.id]
            u_list = comp_map[o]
            v_list = comp_map[p]
            pairs = []
            is_self = (o == p)
            for u in u_list:

                if is_self:
                    if (u, u) in path_map and len(path_map[(u, u)]) > 0:
                        pairs.append((u, u))
                    continue

                neighbor_nodes = [node for node in mrrg.get_sink_fu_neighbors(fu_nodes[u].id)]
                neighbor_indices = {fu_id2idx[n.id] for n in neighbor_nodes if n.id in fu_id2idx and fu_id2idx[n.id] != u}
                for v in v_list:
                    if (u, v) in path_map and len(path_map[(u, v)]) > 0: # Removing neighbor indices check for now
                        pairs.append((u, v))
            valid_pairs[(o, p)] = pairs

        edge_zero_pairs = []
        for e in range(num_dfg_edges):
            o = dfg_id2idx[dfg_edges[e].source.id]
            p = dfg_id2idx[dfg_edges[e].destination.id]
            pairs = valid_pairs.get((o, p), [])
            if not pairs:
                edge_zero_pairs.append((e, o, p))

        print(f"# DFG edges with zero valid (u,v) pairs: {len(edge_zero_pairs)}")
        for e, o, p in edge_zero_pairs:
            print(f"  edge_id={e}, DFG {dfg_nodes[o].id} -> {dfg_nodes[p].id}")
            
        # Upper-bound valid variables with 1
        for (o, p), uvs in valid_pairs.items():
            for (u, v) in uvs:
                E_oupv[(o, u, p, v)].ub = 1

        # -- 1. Functional Unit Exclusivity ----------------------------------------------
        model.addConstrs(
            (gp.quicksum(F_ou[o, u] for o in range(num_dfg_nodes)) <= 1
            for u in range(num_fu_nodes)),
            name="fu_exclusivity"
        )

        # -- 2. Functional Unit Placement ----------------------------------------------
        model.addConstrs(
            (gp.quicksum(F_ou[o, u] for u in range(num_fu_nodes)) == 1
            for o in range(num_dfg_nodes)),
            name="place_once"
        )

        # -- 3. Fanin Requirement (Restrictive) ----------------------------------------------
        # valid_fanin = []
        # u_compat_map = {}
        # for (o, p), uvs in valid_pairs.items():

        #     v_to_us = {}
        #     for (u, v) in uvs:
        #         v_to_us.setdefault(v, []).append(u)

        #     for v, us in v_to_us.items():
        #         valid_fanin.append((o, p, v))
        #         u_compat_map[(o, p, v)] = us


        # model.addConstrs(
        #     (F_ou[p, v] <= gp.quicksum(E_oupv[o, u, p, v] for u in u_compat_map[(o, p, v)])
        #     for (o, p, v) in valid_fanin),
        #     name=f"fanin"
        # )

        # -- 4. Fanout Requirement ----------------------------------------------
        valid_fanout: List[Tuple[int, int, int, int]] = []
        for (o, p), uvs in valid_pairs.items():
            for (u, v) in uvs:
                valid_fanout.append((o, u, p, v))

        model.addConstrs(
            ( E_oupv[o, u, p, v] <= F_ou[o, u]
                for (o, u, p, v) in valid_fanout
            ),
            name="fanout"
        )

        model.addConstrs(
            (E_oupv[o, u, p, v] <= F_ou[p, v]
            for (o, u, p, v) in valid_fanout),
            name="dest_binding"
        )

        # --- 5. Path Edge Requirement ----------------------------------------------
        valid_path_edges: List[Tuple[int, int, int, int]] = []
        path_edge_map: Dict[Tuple[int, int], List[List[str]]] = {}  # Lists of node IDs, not MRRGNode objects

        for (o, p), uvs in valid_pairs.items():
            for (u, v) in uvs:
                valid_path_edges.append((o, u, p, v))
                path_edge_map[(u, v)] = path_map[(u, v)]

        # enforce: E[o,u,p,v] ⇒ at least one path (u,v,q) active
        model.addConstrs(
            (
                E_oupv[o, u, p, v] <= gp.quicksum(
                    P_uvq[u, v, q] for q in range(len(path_edge_map[(u, v)]))
                )
                for (o, u, p, v) in valid_path_edges
            ),
            name="path_edge"
        )

        # 6. Paths Mutually Exclusive Requirement (Too strict - think about mutli-ported nodes) ----------------------------------------------
        resource_to_paths: Dict[MRRGNode, List[Tuple[int, int, int]]] = {}

        for (u, v), paths in path_map.items():
            for q, path_nodes in enumerate(paths):

                # We only want to consider intermediate nodes, not source/destination FUs
                intermediate_node_ids = path_nodes[1:-1] if len(path_nodes) > 2 else []

                for node_id in intermediate_node_ids:  # each intermediate resource (node ID)
                    # Convert node ID to MRRGNode object
                    r = mrrg.get_node(node_id)
                    if r is None:
                        continue
                    if r not in resource_to_paths:
                        resource_to_paths[r] = []
                    resource_to_paths[r].append((u, v, q))

        num_constraints = 0
        for r, tuples in resource_to_paths.items():
            # Optional: skip resources only used by one path
            if len(tuples) <= 1:
                continue
            
            capacity = r.capacity
            # Sum of all paths using resource r must be ≤ capacity
            model.addConstr(
                gp.quicksum(P_uvq[u, v, q] for (u, v, q) in tuples) <= capacity,
                name=f"path_exclusivity[{r.id}]"
            )

        # for r, tuples in resource_to_paths.items():
        #     if len(tuples) > 1:  # contested resource
        #         print(f"Resource {r.id} used by these paths:")
        #         for (u, v, q) in tuples:
        #             print(f"   (u={u}, v={v}, path={q})")

        # 7. Force each DFG edge to be mapped once.
        model.addConstrs(
            (gp.quicksum(E_oupv[o, u, p, v] for (u, v) in valid_pairs[(o, p)]) == 1
            for (o, p) in valid_pairs.keys()),
            name="edge_mapping"
        )

        # 8. Cost Optimization ----------------------------------------------

        # Shwet TODO: Add weigths.
        # Setting these to scalar 1 for now.
        k_ou = 1
        l_oupv = 1
        m_uvq = 1

        # -- Placement Cost ----------------------------------------------
        placement_cost = gp.quicksum(
            k_ou * F_ou[o, u]
            for o in range(num_dfg_nodes)
            for u in comp_map[o])

        # -- Edge Cost ----------------------------------------------
        edge_cost_list = [(o,u,p,v) for e in range(num_dfg_edges)
                        for o,p in [(dfg_id2idx[dfg_edges[e].source.id], dfg_id2idx[dfg_edges[e].destination.id])]
                        for u in comp_map[o] for v in comp_map[p]
                        if v in [fu_id2idx[n.id] for n in mrrg.get_sink_fu_neighbors(fu_nodes[u].id)]]

        edge_cost = gp.quicksum(l_oupv * E_oupv[o,u,p,v] for (o,u,p,v) in edge_cost_list)

        # -- Path Cost ----------------------------------------------
        path_cost = gp.quicksum(
            m_uvq * P_uvq[u, v, q]
            for (u, v), paths in path_map.items()
            for q in range(len(paths)))

        # -- Total Cost ----------------------------------------------
        total_cost = placement_cost + edge_cost + path_cost

        # -- Set Objective ----------------------------------------------
        model.setObjective(total_cost, GRB.MINIMIZE)

        return model, F_ou, E_oupv, P_uvq