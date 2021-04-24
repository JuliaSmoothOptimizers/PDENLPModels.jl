using ForwardDiff
import Gridap.Arrays.kernel_cache
import Gridap.Arrays.apply_kernel!
struct ForwardDiffJacobianKernel2 <: Gridap.Arrays.Kernel
    m::Any
end
function kernel_cache(k::ForwardDiffJacobianKernel2, f, x)
    cfg = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk{length(x)}())
    n = length(x)
    m = k.m
    j = zeros(eltype(x), m, n)
    (j, cfg)
end

@inline function apply_kernel!(cache, k::ForwardDiffJacobianKernel2, f, x)
    j, cfg = cache
    #@warn size(j,1) != length(x) #@notimplemenetdif
    #@warn size(j,2) != length(x) #@notimplemenetdif
    #@show ForwardDiff.jacobian(f,x, cfg = cfg)
    ForwardDiff.jacobian!(j, f, x, cfg)
    j
end

function autodiff_array_jacobian2(a, i_to_x, m, j_to_i)

    i_to_xdual = Gridap.Arrays.apply(i_to_x) do x
        cfg = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk{length(x)}())
        xdual = cfg.duals
        xdual
    end

    j_to_f = Gridap.Arrays.to_array_of_functions(a, i_to_xdual, j_to_i)
    j_to_x = Gridap.Arrays.reindex(i_to_x, j_to_i)

    k = ForwardDiffJacobianKernel2(m)
    Gridap.Arrays.apply(k, j_to_f, j_to_x)

end
