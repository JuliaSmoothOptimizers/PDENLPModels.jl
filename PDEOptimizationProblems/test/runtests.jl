#using Main.PDEOptimizationProblems, Main.PDENLPModels, Test
using PDENLPModels, Test
using PDEOptimizationProblems

# Test that every problem can be instantiated.
for prob in names(PDEOptimizationProblems)
  prob == :PDEOptimizationProblems && continue
  print(prob)
  prob_fn = eval(prob)
  nlp = prob_fn()
  println(" nvar=",nlp.meta.nvar," ncon=",nlp.meta.ncon)
  obj(nlp, nlp.meta.x0)
  nlp.meta.ncon != 0 && cons(nlp, nlp.meta.x0)
end
