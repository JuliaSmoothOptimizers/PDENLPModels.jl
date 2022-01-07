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

# Maybe this?
function autodiff_array_jacobian2(a, i_to_x, m, j_to_i)
  i_to_cfg = lazy_map(Gridap.Arrays.ConfigMap(ForwardDiff.jacobian), i_to_x)
  i_to_xdual = lazy_map(Gridap.Arrays.DualizeMap(ForwardDiff.jacobian), i_to_x)
  i_to_ydual = a(i_to_xdual)
  i_to_result = lazy_map(Gridap.Arrays.AutoDiffMap(ForwardDiff.jacobian), i_to_ydual, i_to_x, i_to_cfg)
  i_to_result
end

#=
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
=#

function _jacobian2(f, uh, fuh::Gridap.FESpaces.DomainContribution)
  terms = Gridap.FESpaces.DomainContribution()
  for trian in Gridap.FESpaces.get_domains(fuh)
    g = Gridap.FESpaces._change_argument(jacobian, f, trian, uh)
    cell_u = get_cell_dof_values(uh)
    # cell_id = get_cell_to_bgcell(trian)
    cell_id = Gridap.FESpaces._compute_cell_ids(uh,trian)
    _temp = g(cell_u) # a bit savage
    ncu = length(_temp[1])
    cell_grad = autodiff_array_jacobian2(g, cell_u, ncu, cell_id)
    Gridap.FESpaces.add_contribution!(terms, trian, cell_grad)
  end
  terms
end

function _jacobian2(f, uh::Gridap.MultiField.MultiFieldFEFunction, fuh::Gridap.CellData.DomainContribution)
  terms = Gridap.CellData.DomainContribution()
  U = Gridap.FESpaces.get_fe_space(uh)
  for trian in Gridap.FESpaces.get_domains(fuh)
    g = Gridap.FESpaces._change_argument(jacobian,f,trian,uh)
    cell_u = lazy_map(Gridap.Fields.DensifyInnerMostBlockLevelMap(), Gridap.FESpaces.get_cell_dof_values(uh))
    cell_id = Gridap.FESpaces._compute_cell_ids(uh,trian)
    # cell_grad = autodiff_array_jacobian(g,cell_u,cell_id)
    _temp = g(cell_u) # a bit savage
    ncu = length(_temp[1])
    cell_grad = autodiff_array_jacobian2(g, cell_u, ncu, cell_id)
    monolithic_result=cell_grad
    blocks        = [] # TO-DO type unstable. How can I infer the type of its entries?
    blocks_coords = Tuple{Int,Int}[]
    nfields = length(U.spaces)
    cell_dofs_field_offsets = Gridap.MultiField._get_cell_dofs_field_offsets(uh)
    for j=1:nfields
      view_range_j=cell_dofs_field_offsets[j]:cell_dofs_field_offsets[j+1]-1
      for i=1:nfields
        view_range_i=cell_dofs_field_offsets[i]:cell_dofs_field_offsets[i+1]-1
        # TO-DO: depending on the residual being differentiated, we may end with
        #        blocks [i,j] full of zeros. I guess that it might desirable to early detect
        #        these zero blocks and use a touch[i,j]==false block in ArrayBlock.
        #        How can we detect that we have a zero block?
        block=lazy_map(x->view(x,view_range_i,view_range_j),monolithic_result)
        append!(blocks,[block])
        append!(blocks_coords,[(i,j)])
      end
    end
    cell_grad=lazy_map(Gridap.Fields.BlockMap((nfields,nfields),blocks_coords),blocks...)
    Gridap.FESpaces.add_contribution!(terms,trian,cell_grad)
  end
  terms
end
