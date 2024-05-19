#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from juliacall import Main as jl
import juliacall
from scipy.sparse import csc_matrix

jl.seval("using Pkg")
jl.seval('Pkg.activate("./")')
jl.seval("Pkg.instantiate()")
jl.seval("using ArgParse")
jl.seval("using GZip")
jl.seval("using JSON3")
jl.seval("using CUDA")
jl.seval("using SparseArrays")
jl.seval("using PDQP")
if jl.CUDA.functional():
    print("GPU is functional")
else:
    print("GPU is not functional")
jl.seval('include("scripts/solve.jl")')

gpu_flag = False

# Define the tolerance and time_sec_limit (provide your own values)
tolerance = 1e-5  # Example tolerance value
time_sec_limit = 3600  # Example time limit in seconds


# In[2]:


n = 5
Q = np.array(
    [
        [4, 1, 0, 0, 0],
        [1, 3, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 3, 1],
        [0, 0, 0, 1, 4],
    ]
)
w = np.array([1, 1, 1, 1, 1])


num_variables = n
num_constraints = 0  # No additional constraints beyond 0 <= x <= 1
variable_lower_bound = np.zeros(n)
variable_upper_bound = np.ones(n)
isfinite_variable_lower_bound = np.zeros(n, dtype=bool)
isfinite_variable_upper_bound = np.zeros(n, dtype=bool)
objective_constant = 0.0
constraint_matrix = np.identity(n)  # No additional constraints
right_hand_side = np.zeros(n)  # No constraints
num_equalities = 0

Q_julia = jl.convert(jl.SparseMatrixCSC, Q)

constraint_matrix_julia = jl.convert(jl.SparseMatrixCSC, constraint_matrix)

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
    constraint_matrix_julia,
    right_hand_side.tolist(),
    num_equalities,
)

print(qp)

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
    2,
    True,
    40,
    termination_params,
    restart_params,
    jl.PDQP.ConstantStepsizeParams(),
)


# In[3]:


import numpy as np
from juliacall import Main as jl

jl.seval('using SparseArrays')

Q = np.array([
    [2, 2],
    [2, 4]
])

w = np.array([2.0, -1.0])

num_variables = 2
num_constraints = 1
variable_lower_bound = np.array([0.0, 1.0])
variable_upper_bound = np.array([1.0, 2.0])
isfinite_variable_lower_bound = np.array([True, True])
isfinite_variable_upper_bound = np.array([True, True])

constraint_matrix_dense = np.array([[-1.0, -1.0]])

constraint_matrix_t_dense = constraint_matrix_dense.T

right_hand_side = np.array([-3.0])

constraint_lower_bound = np.array([-np.inf])
constraint_upper_bound = np.array([3.0])

num_equalities = 0

Q_julia = jl.convert(jl.SparseMatrixCSC, Q)
constraint_matrix_julia = jl.convert(jl.SparseMatrixCSC, constraint_matrix_dense)
constraint_matrix_t_julia = jl.convert(jl.SparseMatrixCSC, constraint_matrix_t_dense)

qp = jl.PDQP.QuadraticProgrammingProblem(
    num_variables,
    num_constraints,
    variable_lower_bound.tolist(),
    variable_upper_bound.tolist(),
    isfinite_variable_lower_bound.tolist(),
    isfinite_variable_upper_bound.tolist(),
    Q_julia,
    w.tolist(),
    0.0,  # Objective offset
    constraint_matrix_julia,
    constraint_matrix_t_julia,
    right_hand_side.tolist(),
    num_equalities
)

print(qp)


# In[4]:


print("Warm Start ??")
ini = np.array([1,1])
print(qp)
if gpu_flag:
    output = jl.PDQP.optimize_gpu(params, qp, juliacall.convert(jl.Vector[jl.Float64], ini))
else:
    output = jl.PDQP.optimize(params, qp, juliacall.convert(jl.Vector[jl.Float64], ini))
result = output.primal_solution
res = np.array(result)
print(res)


# In[5]:


print("Cold Start")
ini = np.array([0,0])
print(qp)
if gpu_flag:
    output = jl.PDQP.optimize_gpu(params, qp, juliacall.convert(jl.Vector[jl.Float64], ini))
else:
    output = jl.PDQP.optimize(params, qp, juliacall.convert(jl.Vector[jl.Float64], ini))
result = output.primal_solution
res = np.array(result)
print(res)


# In[6]:


print("Julia load mps instance:")
qp = jl.PDQP.qps_reader_to_standard_form("./trivial_qp_model.mps")
print(qp)
if gpu_flag:
    output = jl.PDQP.optimize_gpu(params, qp)
else:
    output = jl.PDQP.optimize(params, qp)

result = output.primal_solution
res = np.array(result)
print(res)


# In[ ]:




