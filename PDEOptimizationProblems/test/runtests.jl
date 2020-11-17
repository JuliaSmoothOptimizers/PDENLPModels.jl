using Main.PDEOptimizationProblems, Test

# Test that every problem can be instantiated.
for prob in names(PDEOptimizationProblems)
  prob == :PDEOptimizationProblems && continue
  println(prob)
  prob_fn = eval(prob)
  prob_fn()
end
