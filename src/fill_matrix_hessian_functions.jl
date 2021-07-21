function fill_hess_coo_numeric!(V,a::Gridap.FESpaces.GenericSparseMatrixAssembler,matdata;n=0)
  nini = n
  for (cellmat_rc,cellidsrows,cellidscols) in zip(matdata...)
    cell_rows = get_cell_dof_ids(a.test,cellidsrows)
    cell_cols = get_cell_dof_ids(a.trial,cellidscols)
    cellmat_r = Gridap.FESpaces.attach_constraints_cols(a.trial,cellmat_rc,cellidscols)
    cell_vals = Gridap.FESpaces.attach_constraints_rows(a.test,cellmat_r,cellidsrows)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    vals_cache = array_cache(cell_vals)
    nini = _fill_hess!(
      a.matrix_type,nini,V,rows_cache,cols_cache,vals_cache,cell_rows,cell_cols,cell_vals,a.strategy)
  end

  nini
end

@noinline function _fill_hess!(
  a::Type{M},nini,V,rows_cache,cols_cache,vals_cache,cell_rows,cell_cols,cell_vals,strategy) where M

  n = nini
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    n = _fill_hess_at_cell!(M,n,V,rows,cols,vals,strategy)
  end
  n
end

@inline function _fill_hess_at_cell!(::Type{M},nini,V,rows,cols,vals,strategy) where M
  n = nini
  for (j,gidcol) in enumerate(cols)
    if gidcol > 0 && Gridap.FESpaces.col_mask(strategy,gidcol)
      _gidcol = Gridap.FESpaces.col_map(strategy,gidcol)
      for (i,gidrow) in enumerate(rows)
        if gidrow > 0 && Gridap.FESpaces.row_mask(strategy,gidrow)
          _gidrow = Gridap.FESpaces.row_map(strategy,gidrow)
          if Gridap.FESpaces.is_entry_stored(M,_gidrow,_gidcol) && _gidcol ≤ _gidrow
            n += 1
            @inbounds V[n] = vals[i,j]
          end
        end
      end
    end
  end
  n
end

@inline function _fill_hess_at_cell!(
  ::Type{M},nini,V,rows::Gridap.FESpaces.BlockArrayCoo,cols::Gridap.FESpaces.BlockArrayCoo,vals::Gridap.FESpaces.BlockArrayCoo,strategy) where M
  n = nini
  for B in Gridap.FESpaces.eachblockid(vals)
    if Gridap.FESpaces.is_nonzero_block(vals,B)
      i,j = B.n
      n = _fill_hess_at_cell!(M,n,V,rows[Gridap.FESpaces.Block(i)],cols[Gridap.FESpaces.Block(j)],vals[B],strategy)
    end
  end
  n
end

function fill_hessstruct_coo_numeric!(I,J,a::Gridap.FESpaces.GenericSparseMatrixAssembler,matdata;n=0)
  nini = n
  for (cellmat_rc,cellidsrows,cellidscols) in zip(matdata...)
    cell_rows = get_cell_dof_ids(a.test,cellidsrows)
    cell_cols = get_cell_dof_ids(a.trial,cellidscols)
    cellmat_r = Gridap.FESpaces.attach_constraints_cols(a.trial,cellmat_rc,cellidscols)
    cell_vals = Gridap.FESpaces.attach_constraints_rows(a.test,cellmat_r,cellidsrows)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    vals_cache = array_cache(cell_vals)
    nini = _fill_hessstruct!(
      a.matrix_type,nini,I,J,rows_cache,cols_cache,vals_cache,cell_rows,cell_cols,cell_vals,a.strategy)
  end

  nini
end

@noinline function _fill_hessstruct!(
  a::Type{M},nini,I,J,rows_cache,cols_cache,vals_cache,cell_rows,cell_cols,cell_vals,strategy) where M

  n = nini
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    n = _fill_hessstruct_at_cell!(M,n,I,J,rows,cols,vals,strategy)
  end
  n
end

@inline function _fill_hessstruct_at_cell!(::Type{M},nini,I,J,rows,cols,vals,strategy) where M
  n = nini
  for (j,gidcol) in enumerate(cols)
    if gidcol > 0 && Gridap.FESpaces.col_mask(strategy,gidcol)
      _gidcol = Gridap.FESpaces.col_map(strategy,gidcol)
      for (i,gidrow) in enumerate(rows)
        if gidrow > 0 && Gridap.FESpaces.row_mask(strategy,gidrow)
          _gidrow = Gridap.FESpaces.row_map(strategy,gidrow)
          if Gridap.FESpaces.is_entry_stored(M,_gidrow,_gidcol) && _gidcol ≤ _gidrow
            n += 1
            @inbounds I[n] = _gidrow
            @inbounds J[n] = _gidcol
          end
        end
      end
    end
  end
  n
end

@inline function _fill_hessstruct_at_cell!(
  ::Type{M},nini,I,J,rows::Gridap.FESpaces.BlockArrayCoo,cols::Gridap.FESpaces.BlockArrayCoo,vals::Gridap.FESpaces.BlockArrayCoo,strategy) where M
  n = nini
  for B in Gridap.FESpaces.eachblockid(vals)
    if Gridap.FESpaces.is_nonzero_block(vals,B)
      i,j = B.n
      n = _fill_hessstruct_at_cell!(M,n,I,J,rows[Gridap.FESpaces.Block(i)],cols[Gridap.FESpaces.Block(j)],vals[B],strategy)
    end
  end
  n
end