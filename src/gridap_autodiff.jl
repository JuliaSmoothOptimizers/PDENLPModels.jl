#=
This is a specialization of Gridap.MultiField._hessian,
see https://github.com/gridap/Gridap.jl/blob/1dae8117dc5ad5b6276a3f2961a847ecbabc696b/src/MultiField/MultiFieldFEAutodiff.jl#L119,
for vector of size 1.

Typically, from Gridap 0.17, `op.res(y, u, λ)` returns a DomainContribution composed of vectors of size 1 instead of real numbers.
=#
function _hessianv1(f,uh::Gridap.MultiField.MultiFieldFEFunction,fuh::Gridap.FESpaces.DomainContribution)
  terms = Gridap.FESpaces.DomainContribution()
  U = Gridap.FESpaces.get_fe_space(uh)
  for trian in Gridap.FESpaces.get_domains(fuh)
    g = Gridap.FESpaces._change_argument(hessian,f,trian,uh)
    cell_u = lazy_map(Gridap.FESpaces.DensifyInnerMostBlockLevelMap(),Gridap.FESpaces.get_cell_dof_values(uh))
    cell_id = Gridap.FESpaces._compute_cell_ids(uh,trian)
    cell_grad = Gridap.Arrays.autodiff_array_hessian(x -> map(x -> x[1], g(x)), cell_u, cell_id)
    monolithic_result=cell_grad
    blocks        = [] # TO-DO type unstable. How can I infer the type of its entries?
    blocks_coords = Tuple{Int,Int}[]
    nfields = length(U.spaces)
    cell_dofs_field_offsets=Gridap.MultiField._get_cell_dofs_field_offsets(uh)
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
    cell_grad=lazy_map(Gridap.FESpaces.BlockMap((nfields,nfields),blocks_coords),blocks...)
    Gridap.FESpaces.add_contribution!(terms,trian,cell_grad)
  end
  terms
end

# For SingleField functions, we can use the default Gridap's function
_hessianv1(f,uh,fuh) = Gridap.FESpaces._hessian(f,uh,fuh)
