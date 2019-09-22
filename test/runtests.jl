using Test
using JVODE
using Sundials

prob = ODEProblem((du, u, p, t)->begin
                      du[1] = u[2]
                      du[2] = (1-u[1]^2)*3 * u[2] - u[1]
                  end, [2, 0.], (0.0, 1.39283880203), 0.9)

integ = @inferred init(prob, Adams(), reltol=0.0, abstol=1e-6)
step!(integ)
integs = init(prob, CVODE_Adams())
step!(integs)
@test integ.dt == integs.dt
