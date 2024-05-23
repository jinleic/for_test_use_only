import os

import juliacall
import numpy as np
from juliacall import Main as jl
from phisolve import problems
from phisolve.problems.boxqp import BoxQP
from scipy.sparse import csc_matrix

script_dir = os.path.dirname(os.path.abspath(__file__))

jl.seval("using Pkg")
jl.seval(f'Pkg.activate("{script_dir}")')
jl.seval("Pkg.instantiate()")
jl.seval("using ArgParse")
jl.seval("using GZip")
jl.seval("using JSON3")
jl.seval("using CUDA")
jl.seval("using SparseArrays")
jl.seval("using PDQP")
src_path = os.path.join(script_dir, "scripts")
jl.seval(f'include("{os.path.join(src_path, "solve.jl")}")')
if jl.CUDA.functional():
    print("GPU is functional")
else:
    print("GPU is not functional")

tolerance = 1e-5  # Example tolerance value
time_sec_limit = 3600  # Example time limit in seconds

def pdqp(problem: BoxQP, samples, device):
    Q = problem.Q
    w = problem.w
    lb = problem.bounds[0]
    ub = problem.bounds[1]
    n = problem.nvar

    num_variables = n
    num_constraints = 0  # No additional constraints beyond 0 <= x <= 1
    variable_lower_bound = lb
    variable_upper_bound = ub
    isfinite_variable_lower_bound = np.ones(n, dtype=bool)
    isfinite_variable_upper_bound = np.ones(n, dtype=bool)
    objective_constant = 0.0
    constraint_matrix = np.vstack(
        [np.identity(n), -np.identity(n)]
    )  # No additional constraints
    right_hand_side = np.hstack([np.ones(n), np.zeros(n)])  # No constraints
    num_equalities = 0

    Q_julia = jl.convert(jl.SparseMatrixCSC, Q)

    constraint_matrix_julia = jl.convert(jl.SparseMatrixCSC, constraint_matrix)
    constraint_matrix_julia_t = jl.convert(jl.SparseMatrixCSC, constraint_matrix.T)

    qp = jl.PDQP.QuadraticProgrammingProblem(
        num_variables,
        num_constraints,
        variable_lower_bound.tolist(),
        variable_upper_bound.tolist(),
        isfinite_variable_lower_bound.tolist(),
        isfinite_variable_upper_bound.tolist(),
        Q_julia,
        w.tolist(),
        objective_constant,
        constraint_matrix_julia,
        constraint_matrix_julia_t,
        right_hand_side.tolist(),
        num_equalities,
    )

    # Construct restart parameters
    restart_params = jl.PDQP.construct_restart_parameters(
        jl.PDQP.ADAPTIVE_KKT,  # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        jl.PDQP.KKT_GREEDY,  # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,  # restart_frequency_if_fixed
        0.36,  # artificial_restart_threshold
        0.2,  # sufficient_reduction_for_restart
        0.8,  # necessary_reduction_for_restart
        0.2,  # primal_weight_update_smoothing
    )

    # Construct termination criteria
    termination_params = jl.PDQP.construct_termination_criteria(
        eps_optimal_absolute=tolerance,
        eps_optimal_relative=tolerance,
        time_sec_limit=time_sec_limit,
        iteration_limit=jl.typemax(jl.Int32),
        kkt_matrix_pass_limit=jl.Inf,
    )

    # Construct PDHG parameters
    params = jl.PDQP.PdhgParameters(
        10,
        True,
        1.0,
        1.0,
        True,
        0,
        True,
        40,
        termination_params,
        restart_params,
        jl.PDQP.ConstantStepsizeParams(),
    )

    results = []

    if device == "gpu" or device == "cuda":
        for sample in samples:
            if sample is None:
                continue
            output = jl.PDQP.optimize_gpu(
                params, qp, juliacall.convert(jl.Vector[jl.Float64], sample)
            )
            results.append(np.array(output.primal_solution))
    else:
        for sample in samples:
            if sample is None:
                continue
            output = jl.PDQP.optimize(
                params, qp, juliacall.convert(jl.Vector[jl.Float64], sample)
            )
            results.append(np.array(output.primal_solution))
            
    return results
