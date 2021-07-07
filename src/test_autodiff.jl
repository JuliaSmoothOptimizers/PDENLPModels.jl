using ForwardDiff
import Gridap.Arrays.return_cache
import Gridap.Arrays.evaluate!
struct ForwardDiffJacobianMap2 <: Gridap.Arrays.Map
  m::Any
end
function return_cache(k::ForwardDiffJacobianMap2, f, x)
  cfg = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk{length(x)}())
  n = length(x)
  m = k.m
  j = zeros(eltype(x), m, n)
  (j, cfg)
end

@inline function evaluate!(cache, k::ForwardDiffJacobianMap2, f, x)
  j, cfg = cache
  #@warn size(j,1) != length(x) #@notimplemenetdif
  #@warn size(j,2) != length(x) #@notimplemenetdif
  #@show ForwardDiff.jacobian(f,x, cfg = cfg)
  ForwardDiff.jacobian!(j, f, x, cfg)
  j
end

function autodiff_array_jacobian2(a, i_to_x, m, j_to_i)
  i_to_xdual = lazy_map(i_to_x) do x #Gridap.Arrays.apply(i_to_x) do x
    cfg = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk{length(x)}())
    xdual = cfg.duals
    xdual
  end

  j_to_f = Gridap.Arrays.to_array_of_functions(a, i_to_xdual, j_to_i)
  j_to_x = lazy_map(Reindex(i_to_x), j_to_i) #Gridap.Arrays.reindex(i_to_x, j_to_i)

  k = ForwardDiffJacobianMap2(m)
  lazy_map(k, j_to_f, j_to_x)
end

function _jacobian2(f, uh, fuh::Gridap.FESpaces.DomainContribution)
  terms = Gridap.FESpaces.DomainContribution()
  for trian in Gridap.FESpaces.get_domains(fuh)
    g = Gridap.FESpaces._change_argument(f, trian, uh)
    cell_u = get_cell_dof_values(uh)
    cell_id = get_cell_to_bgcell(trian)
    _temp = g(cell_u) # a bit savage
    ncu = length(_temp[1])
    cell_grad = autodiff_array_jacobian2(g, cell_u, ncu, cell_id)
    Gridap.FESpaces.add_contribution!(terms, trian, cell_grad)
  end
  terms
end
