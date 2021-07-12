#=
Using the following function
function AffineFEOperator(weakform::Function,assem::Assembler)

  trial = get_trial(assem)
  test = get_test(assem)

  u = get_cell_shapefuns_trial(trial)
  v = get_cell_shapefuns(test)

  uhd = zero(trial)
  matcontribs, veccontribs = weakform(u,v)
  data = collect_cell_matrix_and_vector(matcontribs,veccontribs,uhd)
  A,b = assemble_matrix_and_vector(assem,data)

  AffineFEOperator(trial,test,A,b)
end
make a function that returns a matrix:
in the format (y,u,v).
=#
