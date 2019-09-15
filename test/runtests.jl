using Test
using JVODE

prob = ODEProblem((du, u, p, t)->@.(du = p*u), [1, 2.], (0.0, 1.0), 0.9)

@inferred init(prob, Adams())
