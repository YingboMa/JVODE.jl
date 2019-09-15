module JVODE

using Parameters: @unpack
using DiffEqBase: ODEProblem

using Reexport: @reexport
@reexport using DiffEqBase

@enum Solver::Bool ADAMS BDF
@enum NLSolver::Bool FUNCTIONAL NEWTON
@enum ITask::Bool NORMAL ONE_STEP
# TODO: Optional Inputs and Outputs

# Basic JVODE constants
const ADAMS_Q_MAX = 12            # max value of q for lmm == ADAMS
const BDF_Q_MAX   =  5            # max value of q for lmm == BDF
const Q_MAX       =  ADAMS_Q_MAX  # max value of q for either lmm
const L_MAX       =  (Q_MAX+1)    # max value of L for either lmm
const NUM_TESTS   =  5            # number of error test quantities


mutable struct JVodeMem{uType,tType,Rtol,Atol,F,D}
    f::F
    data::D
    solver::Solver
    nlsolver::NLSolver
    rtol::Rtol
    atol::Atol
    zn::NTuple{L_MAX,Vector{uType}} # Nordsieck history array

    # vectors with length `length(u0)`
    ewt::Vector{uType} # error weight vector
    uprev::Vector{uType}
    acor::Vector{uType}
    tempv::Vector{uType}
    ftemp::Vector{uType}

    # step data
    q::Int                        # current order
    qprime::Int                   # order to be used on the next step {q-1, q, q+1}
    qwait::Int                    # number of steps to wait before order change
    L::Int                        # L = q+1
    h::tType                      # current step size
    hprime::tType                 # next step size
    eta::tType                    # eta = hprime / h
    hscale::tType                 # the step size information in `zn`
    tn::tType                     # current time
    tau::NTuple{L_MAX,tType}       # tuple of previous `q+1` successful step sizes
    tq::NTuple{NUM_TESTS,tType}   # tuple of test quantities
    coeff::NTuple{L_MAX,uType}     # coefficients of l(x)
    rl2::uType                    # 1/l[2]
    gamma::uType                  # gamma = h * rl2
    gammap::uType                 # `gamma` at the last setup call
    gamrat::uType                 # gamma/gammap
    crate::uType                  # estimated corrector convergence rate
    acnrm::uType                  # | acor | wrms
    mnewt::Int                    # Newton iteration counter

    # Limits
    qmax::Int                     # q <= qmax
    mxstep::Int                   # maximum number of internal steps for one user call
    maxcor::Int                   # maximum number of `nlsolve`
    mxhnil::Int                   # maximum number of warning messages issued to the
                                  # user that `t + h == t` for the next internal step

    hmin::tType                   # |h| >= hmin
    hmax_inv::tType               # |h| <= 1/hmax_inv
    etamax::tType                 # eta <= etamax

    # counters
    nst::Int                      # number of internal steps taken
    nfe::Int                      # number of f calls
    ncfn::Int                     # number of corrector convergence failures
    netf::Int                     # number of error test failures
    nni::Int                      # number of Newton iterations performed
    nsetups::Int                  # number of setup calls
    nhnil::Int                    # number of messages issued to the user that
                                  # `t + h == t` for the next iternal step
    lrw::Int                      # number of real words in CVODE work vectors
    liw::Int                      # no. of integer words in CVODE work vectors

    # saved vales
    qu::Int                       # last successful q value used
    nstlp::Int                    # step number of last setup call
    hu::tType                     # last successful h value used
    saved_tq5::tType              # saved value of tq[5]
    jcur::Bool                    # Is the Jacobian info used by
                                  # linear solver current?
    tolsf::Float64                # tolerance scale factor
    setupNonNull::Bool            # Does setup do something?

    # Arrays for Optional Input and Optional Output

    #long int *cv_iopt::Int  /* long int optional input, output */
    #real     *cv_ropt::Int  /* real optional input, output     */
    function JVodeMem(prob::ODEProblem,
                      ::Type{Rtol}=Float64, ::Type{Atol}=Float64) where {Rtol,Atol}
        @unpack f, u0, tspan, p = prob
        obj = new{typeof(u0),eltype(tspan),Rtol,Atol,typeof(f),typeof(p)}()
        obj.f, obj.data = f, p
        return obj
    end
end


"""
    jvode(f, u0, tspan; rtol=1e-3, atol=1e-6)

DVODE: Variable-coefficient Ordinary Differential Equation solver,
with fixed-leading-coefficient implementation.
This version is in double precision.

DVODE solves the initial value problem for stiff or nonstiff
systems of first order ODEs,
    dy/dt = f(t,y) ,  or, in component form,
    dy(i)/dt = f(i) = f(i,t,y(1),y(2),...,y(NEQ)) (i = 1,...,NEQ).
DVODE is a package based on the EPISODE and EPISODEB packages, and
on the ODEPACK user interface standard, with minor modifications.
----------------------------------------------------------------------
Authors:
              Peter N. Brown and Alan C. Hindmarsh
              Center for Applied Scientific Computing, L-561
              Lawrence Livermore National Laboratory
              Livermore, CA 94551
and
              George D. Byrne
              Illinois Institute of Technology
              Chicago, IL 60616
----------------------------------------------------------------------
References:

1. P. N. Brown, G. D. Byrne, and A. C. Hindmarsh, "VODE: A Variable
   Coefficient ODE Solver," SIAM J. Sci. Stat. Comput., 10 (1989),
   pp. 1038-1051.  Also, LLNL Report UCRL-98412, June 1988.
2. G. D. Byrne and A. C. Hindmarsh, "A Polyalgorithm for the
   Numerical Solution of Ordinary Differential Equations,"
   ACM Trans. Math. Software, 1 (1975), pp. 71-96.
3. A. C. Hindmarsh and G. D. Byrne, "EPISODE: An Effective Package
   for the Integration of Systems of Ordinary Differential
   Equations," LLNL Report UCID-30112, Rev. 1, April 1977.
4. G. D. Byrne and A. C. Hindmarsh, "EPISODEB: An Experimental
   Package for the Integration of Systems of Ordinary Differential
   Equations with Banded Jacobians," LLNL Report UCID-30132, April
   1976.
5. A. C. Hindmarsh, "ODEPACK, a Systematized Collection of ODE
   Solvers," in Scientific Computing, R. S. Stepleman et al., eds.,
   North-Holland, Amsterdam, 1983, pp. 55-64.
6. K. R. Jackson and R. Sacks-Davis, "An Alternative Implementation
   of Variable Step-Size Multistep Formulas for Stiff ODEs," ACM
   Trans. Math. Software, 6 (1980), pp. 295-318.
"""
function jvode(f, u0, tspan; rtol=1e-3, atol=1e-6)
    maxorder = (12, 5)
    maxstep0 = 500
    maxwarns = 10
end

end # module
