from juliacall import Main as jl

jl.seval('using Pkg')
jl.seval('Pkg.activate("./")')

jl.seval('Pkg.instantiate()')

jl.seval("using ArgParse")
jl.seval("using GZip")
jl.seval("using JSON3")
jl.seval("using CUDA")
jl.seval("using PDQP")

qp = jl.PDQP.qps_reader_to_standard_form("trivial_qp_model.mps")
print(qp)
if jl.CUDA.functional():
    print("GPU is functional")
else:
    print("GPU is not functional")