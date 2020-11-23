"""
`_split_FEFunction(:: AbstractVector,  :: FESpace, :: Union{FESpace, Nothing})`

Split the vector x into two FEFunction corresponding to the solution `y` and the control `u`.
Returns nothing for the control `u` if Ycon == nothing.

Do not verify the compatible length.
"""
function _split_FEFunction(x    :: AbstractVector,
                           Ypde :: FESpace,
                           Ycon :: FESpace)

    nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
    nvar_con = Gridap.FESpaces.num_free_dofs(Ycon)

    yh = FEFunction(Ypde, x[1:nvar_pde])
    uh = FEFunction(Ycon, x[1+nvar_pde:nvar_pde+nvar_con])

 return yh, uh
end

function _split_FEFunction(x    :: AbstractVector,
                           Ypde :: FESpace,
                           Ycon :: Nothing)

    nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
    yh = FEFunction(Ypde, x[1:nvar_pde])

 return yh, nothing
end

"""
`_split_vector(:: AbstractVector,  :: FESpace, :: Union{FESpace, Nothing})`

Split the vector x into three vectors: y, u, k.
Returns nothing for the control `u` if Ycon == nothing.

Do not verify the compatible length.
"""
function _split_vector(x    :: AbstractVector,
                       Ypde :: FESpace,
                       Ycon :: FESpace)

    nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
    nvar_con = Gridap.FESpaces.num_free_dofs(Ycon)
    #nparam   = length(x) - (nvar_pde + nvar_con)

    y = x[1:nvar_pde]
    u = x[1+nvar_pde:nvar_pde+nvar_con]
    k = x[nvar_pde+nvar_con+1:length(x)]

 return y, u, k
end

function _split_vector(x    :: AbstractVector,
                       Ypde :: FESpace,
                       Ycon :: Nothing)

    nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
    #nparam   = length(x) - nvar_pde

    y = x[1:nvar_pde]
    k = x[nvar_pde+1:length(x)]

 return y, [], k
end
