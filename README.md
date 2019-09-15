# JVODE.jl

## Work In Process

JVODE.jl is a pure Julia translation of the famous
[CVODE](https://computing.llnl.gov/projects/sundials/cvode) solver, which
includes Adams-Moulton formulas, with the order varying between 1 and 12, and
the Backward Differentiation Formulas, with order varying between 1 and 5. Thus,
it is capable of solving both non-stiff and stiff ordinary differential equation
problems.
