include("test_func.jl")

#Copy-paste from
#https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/test/problems/hs10.jl
include("hs10.jl")

nlp = hs10_autodiff()

hv_test(nlp)
jv_test(nlp)
