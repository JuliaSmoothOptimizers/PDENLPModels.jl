using SparsityDetection, SparseArrays

###############################################################################
#
# Tangi, November 30th 2020
#
# Initial test of jacobian_sparsity from SparsityDetection
#https://github.com/SciML/SparsityDetection.jl
#
# Motivation: do this on cons(nlp, x) or cons!(nlp, x, cx)
#
# Issue: Is there a turnaround to make it work with the expected format?
#        Here, c is working but none of the other four.
#
# Put an issue? https://github.com/SciML/SparsityDetection.jl/issues
#
###############################################################################
function c(dx, x) #The one working
 dx[1] = -3 * x[1]^2 + 2 * x[1] * x[2]
 dx[2] = - x[2]^2 + 1
 nothing 
end

input = rand(2)
output = similar(input)
sparsity_pattern = jacobian_sparsity(c,output,input) #this one works

function c2(dx :: AbstractVector, x :: AbstractVector)
 dx[1:2] .= [-3 * x[1]^2 + 2 * x[1] * x[2]; - x[2]^2 + 1] 
 nothing 
end

#Doesn't work:
input = rand(2)
output = similar(input)
sparsity_pattern = jacobian_sparsity(c2, output, input)

function c3(dx, x)
    cx = [-3 * x[1]^2 + 2 * x[1] * x[2]; - x[2]^2 + 1]  #cons(nlp, x)
    for i=1:2
       dx[i] = cx[i] 
    end
    nothing
end

input = rand(2)
output = similar(input)
sparsity_pattern = jacobian_sparsity(c3, output, input)

function f1(dx, x)
    for i in 1:length(x)
        dx[i] = [-3 * x[1]^2 + 2 * x[1] * x[2]; - x[2]^2 + 1][i]
    end
end
input = rand(2)
output = similar(input)
sparsity_pattern1 = sparsity!(f1, output, input) #returns 0s

function f2(dx, x)
    A = rand(2,2)
    for i in 1:length(x)
        dx[i] = A[i,:]'*x
    end
end
input = rand(2)
output = similar(input)
sparsity_pattern1 = sparsity!(f2, output, input)
