module JVODE

using LinearAlgebra

using Parameters: @unpack
using DiffEqBase: ODEProblem, @..

using Reexport: @reexport
@reexport using DiffEqBase
export Adams, BDF


###
### Types
###
abstract type JVNLSolver <: DiffEqBase.AbstractNLSolverAlgorithm
end
struct JVFunctional <: JVNLSolver
end
struct JVNewton <: JVNLSolver
end

abstract type JVODEAlgorithm <: DiffEqBase.AbstractODEAlgorithm
end
struct Adams{NLAlg} <: JVODEAlgorithm
    nlsolve::NLAlg
end
Adams() = Adams(JVFunctional())
struct BDF{NLAlg} <: JVODEAlgorithm
    nlsolve::NLAlg
end
BDF() = BDF(JVNewton())

# TODO: Optional Inputs and Outputs (in `cvode.h`)

# Basic JVODE constants
const ADAMS_Q_MAX = 12            # max value of q for lmm == Adams
const BDF_Q_MAX   =  5            # max value of q for lmm == BDF
const Q_MAX       =  ADAMS_Q_MAX  # max value of q for either lmm
const L_MAX       =  (Q_MAX+1)    # max value of L for either lmm
const NUM_TESTS   =  5            # number of error test quantities

mutable struct JVOptions{Rtol,Atol}
    reltol::Rtol
    abstol::Atol
end

mutable struct JVIntegrator{Alg,uType,tType,uEltype,solType,Rtol,Atol,F,P} <: DiffEqBase.AbstractODEIntegrator{Alg,true,uType,tType}
    sol::solType
    opts::JVOptions{Atol,Rtol}
    zn::NTuple{L_MAX,uType} # Nordsieck history array

    # vectors with length `length(u0)`
    ewt::uType # error weight vector
    u::uType
    acor::uType
    tempv::uType
    ftemp::uType

    # step data
    q::Int                        # current order
    qprime::Int                   # order to be used on the next step {q-1, q, q+1}
    qwait::Int                    # number of steps to wait before order change
    #L::Int                        # L = q+1
    dt::tType                     # current step size
    dtprime::tType                # next step size
    eta::tType                    # eta = dtprime / dt
    dtscale::tType                # the step size information in `zn`
    t::tType                      # current time
    tau::NTuple{L_MAX,tType}      # tuple of previous `q+1` successful step sizes
    tq::NTuple{NUM_TESTS,tType}   # tuple of test quantities
    coeff::NTuple{L_MAX,uEltype}  # coefficients of l(x)
    rl2::uEltype                  # 1/l[2]
    gamma::uEltype                # gamma = h * rl2
    gammap::uEltype               # `gamma` at the last setup call
    gamrat::uEltype               # gamma/gammap
    crate::uEltype                # estimated corrector convergence rate
    acnrm::uEltype                # | acor | wrms
    mnewt::Int                    # Newton iteration counter

    # Limits
    qmax::Int                     # q <= qmax
    mxstep::Int                   # maximum number of internal steps for one user call
    maxcor::Int                   # maximum number of `nlsolve`
    mxhnil::Int                   # maximum number of warning messages issued to the
                                  # user that `t + h == t` for the next internal step

    dtmin::tType                  # |h| >= hmin
    dtmax_inv::tType              # |h| <= 1/hmax_inv
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
    dtu::tType                    # last successful h value used
    saved_tq5::tType              # saved value of tq[5]
    jcur::Bool                    # Is the Jacobian info used by
                                  # linear solver current?
    tolsf::Float64                # tolerance scale factor
    setupNonNull::Bool            # Does setup do something?

    # Arrays for Optional Input and Optional Output

    #long int *cv_iopt::Int  # long int optional input, output */
    #real     *cv_ropt::Int  # real optional input, output     */
    function JVIntegrator(prob::ODEProblem, ::solType, ::Alg, ::Rtol, ::Atol) where {solType,Alg,Rtol,Atol}
        @unpack f, u0, tspan, p = prob
        obj = new{Alg,typeof(u0),eltype(tspan),eltype(u0),solType,Rtol,Atol,typeof(f),typeof(p)}()
        return obj
    end
end

###
### Default Constants
###
const HMIN_DEFAULT     = 0.0
const HMAX_INV_DEFAULT = 0.0
const MXHNIL_DEFAULT   = 10
const MXSTEP_DEFAULT   = 5000

###
### Routine-Specific Constants
###
macro define(name, val)
    esc(:(const $name = $val))
end
begin

    # CVodeDky */

    @define FUZZ_FACTOR 100.0

    # CVHin */

    @define HLB_FACTOR 100.0
    @define HUB_FACTOR 0.1
    @define H_BIAS     0.5
    @define MAX_ITERS  4

    # CVSet */

    @define CORTES 0.1

    # CVStep return values */

    @define SUCCESS_STEP      0
    @define REP_ERR_FAIL     -1
    @define REP_CONV_FAIL    -2
    @define SETUP_FAILED     -3
    @define SOLVE_FAILED     -4

    # CVStep control constants */

    @define PREDICT_AGAIN    -5
    @define DO_ERROR_TEST     1

    # CVStep */

    @define THRESH 1.5
    @define ETAMX1 10000.0
    @define ETAMX2 10.0
    @define ETAMX3 10.0
    @define ETAMXF 0.2
    @define ETAMIN 0.1
    @define ETACF  0.25
    @define ADDON  0.000001
    @define BIAS1  6.0
    @define BIAS2  6.0
    @define BIAS3  10.0
    @define ONEPSM 1.000001

    @define SMALL_NST    10   # nst > SMALL_NST => use ETAMX3          */
    @define MXNCF        10   # max no. of convergence failures during */
    # one step try                           */
    @define MXNEF         7   # max no. of error test failures during  */
    # one step try                           */
    @define MXNEF1        3   # max no. of error test failures before  */
    # forcing a reduction of order           */
    @define SMALL_NEF     2   # if an error failure occurs and         */
    # SMALL_NEF <= nef <= MXNEF1, then       */
    # reset eta =  MIN(eta, ETAMXF)          */
    @define LONG_WAIT    10   # number of steps to wait before         */
    # considering an order change when       */
    # q==1 and MXNEF1 error test failures    */
    # have occurred                          */

    # CVnls return values */

    @define SOLVED            0
    @define CONV_FAIL        -1
    @define SETUP_FAIL_UNREC -2
    @define SOLVE_FAIL_UNREC -3

    # CVnls input flags */

    @define FIRST_CALL      0
    @define PREV_CONV_FAIL -1
    @define PREV_ERR_FAIL  -2

    # CVnls other constants */

    @define FUNC_MAXCOR 3  # maximum no. of corrector iterations   */
    # for iter == FUNCTIONAL                */
    @define NEWT_MAXCOR 3  # maximum no. of corrector iterations   */
    # for iter == NEWTON                    */

    @define CRDOWN 0.3 # constant used in the estimation of the   */
    # convergence rate (crate) of the          */
    # iterates for the nonlinear equation      */
    @define DGMAX  0.3 # iter == NEWTON, |gamma/gammap-1| > DGMAX */
    # => call lsetup                           */

    @define RDIV        2  # declare divergence if ratio del/delp > RDIV  */
    @define MSBP       20  # max no. of steps between lsetup calls        */

    @define TRY_AGAIN  99  # control constant for CVnlsNewton - should be */
    # distinct from CVnls return values            */
end

###
### CVODE Implementation
###
function DiffEqBase.__init(prob::ODEProblem, alg::JVODEAlgorithm; reltol=1e-3, abstol=1e-6)
    # analogous to `CVodeMalloc`
    @unpack f, u0, tspan, p = prob
    # copy input paramters
    uType = typeof(u0)
    uEltype = eltype(u0)
    tType = eltype(tspan)
    timeseries = uType[]
    ts = tType[]

    destats = DiffEqBase.DEStats(0)
    # TODO: dense output
    dense = true
    #ks = Vector{uType}(undef, 0)
    #id = InterpolationData(f,timeseries,ts,ks,dense,cache)

    sol = DiffEqBase.build_solution(prob,alg,ts,timeseries,
                                    dense=dense, #k=ks, interp=id, TODO
                                    calculate_error = false, destats=destats)
    integrator = JVIntegrator(prob, sol, alg, reltol, abstol)

    integrator.sol = sol
    integrator.t = prob.tspan |> first
    integrator.opts = JVOptions(reltol, abstol)
    integrator.dtmin = HMIN_DEFAULT
    integrator.dtmax_inv = HMAX_INV_DEFAULT
    integrator.mxhnil = MXHNIL_DEFAULT
    integrator.mxstep = MXSTEP_DEFAULT
    integrator.maxcor = alg.nlsolve isa JVNewton ? NEWT_MAXCOR : FUNC_MAXCOR

    maxorder = alg isa Adams ? ADAMS_Q_MAX : BDF_Q_MAX

    # allocate the vectors
    integrator.u = copy(u0)
    integrator.zn = ntuple(i->i<=maxorder+1 ? similar(integrator.u) : similar(integrator.u, 0), Val(L_MAX))
    integrator.ewt = similar(integrator.u)
    integrator.acor = similar(integrator.u)
    integrator.tempv = similar(integrator.u)
    integrator.ftemp = similar(integrator.u)

    # set the `ewt` vector
    setewt!(integrator, integrator.u)

    # set step paramters
    integrator.q = 1
    integrator.qwait = integrator.q + 1
    integrator.etamax = maxorder

    # TODO: set the linear solver

    # initialize `zn[1]` in the history array
    copyto!(integrator.zn[1], u0)

    # initialize all counters
    integrator.nst = integrator.nfe = integrator.ncfn = integrator.netf = integrator.nni = integrator.nsetups = integrator.nhnil = integrator.lrw = integrator.liw = 0

    # initialize misc
    integrator.qu = 0
    integrator.dtu = zero(integrator.dtu)
    integrator.tolsf = 1
    return integrator
end

DiffEqBase.has_reinit(integrator::JVIntegrator) = true
function DiffEqBase.reinit!(integrator::JVIntegrator, u0=integrator.uprev;
                            alg=integrator.alg, tspan=integrator.tspan,
                            reltol=integrator.reltol, abstol=integrator.abstol,
                            p=integrator.p) # TODO: opts
    # TODO: `CVReInit`
end

function DiffEqBase.step!(integrator::JVIntegrator, dt=nothing)#, stop_at_tdt=false)
    @unpack f, p, tspan = integrator.sol.prob
    @unpack zn, t, dtmax_inv, dtmin, nst, mxstep, mxhnil = integrator
    tout = dt === nothing ? tspan[end] : t + dt
    if nst === 0 # first step
        f(zn[2], zn[1], p, t)
        integrator.nfe = 1
        integrator.dt = zero(integrator.dt)
        # TODO: user initdt
        if iszero(integrator.dt)
            dt_ok = initdt!(integrator, tout)
            dt_ok || error("tout=$tout too close to t0=$t to start integration")
        end
        #rh = abs(integrator.dt) * dtmax_inv
        #rh > one(rh) && (integrator.dt /= rh)
        abs(integrator.dt) < dtmin && (integrator.dt *= dtmin / abs(integrator.dt))
        integrator.dtscale = integrator.dt
        lmul!(integrator.dt, zn[2])
    end
    # If not the first call, check if tout already reached
    if dt !== nothing && nst > 0 && (integrator.t - tout)*integrator.dt >= zero(integrator.dt)
        # integrator.t = tout
        # interpolate???
        return integrator
    end

    nstloc = 0
    while true
        nextdt = integrator.dt
        nextq  = integrator.q

        # reset and check ewt
        nst > 0 && setewt!(integrator, zn[1])

        # check for too many steps
        if nstloc >= mxstep
            @warn "At t=$t, mxstep=$mxstep steps taken on this call before reaching tout=$tout"
            copyto!(integrator.u, integrator.zn[1])
            break
        end

        # TODO: check for too much accuracy requested

        # check for `dt` below roundoff
        t = integrator.t
        if t + integrator.dt == t
            integrator.nhnil += 1
            integrator.nhnil <= mxhnil && @warn "internal t=$t and step size dt=$(integrator.dt) are such that t + dt = t on the next step. The solver will continue anyway."
            integrator.nhnil == mxhnil && @warn "The above warning has been issued $mxhnil times and will not be issued again for this problem."
        end

        kflag = jvstep(integrator)
        t = integrator.t
        if kflag !== SUCCESS_STEP
            handlefailure(integrator, kflag)
            copyto!(integrator.u, zn[1])
            break
        end

        nstloc += 1

        if dt === nothing
            copyto!(integrator.u, zn[1])
            nextq  = integrator.qprime
            nextdt = integrator.dtprime
        end

        if (t-tout)*integrator.dt >= zero(integrator.t)
            copyto!(integrator.u, zn[1])
            nextq  = integrator.qprime
            nextdt = integrator.dtprime
            break
        end
    end
    return integrator
end

function jvstep(integrator::JVIntegrator)
    # TODO
    return SUCCESS_STEP
end

function initdt!(integrator::JVIntegrator, tout)::Bool
    @unpack zn, t = integrator
    # test for tout too close to t
    tdiff = tout - t
    iszero(tdiff) && return false
    tdir = sign(tdiff)
    tdist = abs(tdiff)
    tround = eps() * max(abs(t), abs(tout))
    tdist < 2tround && return false

    # Set lower and upper bounds on h0, and take geometric mean. Exit with this
    # value if the bounds cross each other.
    hlb = HLB_FACTOR * tround
    hub = let integrator=integrator, tdist=tdist
        temp1 = integrator.tempv
        temp2 = integrator.acor
        map!(abs, temp1, zn[1])
        map!(abs, temp2, zn[2])
        @.. temp1 = temp2 / muladd(HUB_FACTOR, temp1, integrator.opts.abstol)
        hub_inv = norm(temp1, Inf)
        hub = HUB_FACTOR * tdist
        hub*hub_inv > one(hub) && (hub = inv(hub_inv))
        hub
    end
    hg = sqrt(hlb*hub)
    if hub < hlb
        tdir == -1 && (hg = -hg)
        integrator.dt = hg
        return true
    end

    # Loop up to MAX_ITERS times to find h0.
    # Stop if new and previous values differ by a factor < 2.
    # Stop if hnew/hg > 2 after one iteration, as this probably means
    # that the ydd value is bad because of cancellation error.
    count = 0
    yddnorm = (integrator, hg) -> begin
        # This routine computes an estimate of the second derivative of y
        # using a difference quotient, and returns its WRMS norm.
        @unpack zn, tempv, ewt, u, t = integrator
        @unpack f, p = integrator.sol.prob
        @.. u = muladd(hg, zn[2], zn[1])
        f(tempv, u, p, t+hg)
        integrator.nfe += 1
        @. tempv = tempv - zn[2]
        lmul!(inv(hg), tempv)
        yddnrm = wrmsnorm(tempv, ewt)
    end
    local hnew
    while true
        hgs = hg*tdir
        yddnrm = yddnorm(integrator, hgs)
        hnew = (yddnrm*hub*hub > 2) ? sqrt(2/yddnrm) : sqrt(hg*hub)
        count += 1
        count >= MAX_ITERS && break
        hrat = hnew/hg
        (hrat > 0.5) && (hrat < 2) && break
        if (count >= 2) && (hrat > 2)
            hnew = hg
            break
        end
        hg = hnew
    end

    # Apply bounds, bias factor, and attach sign */

    h0 = H_BIAS*hnew
    h0 < hlb && (h0 = hlb)
    h0 > hub && (h0 = hub)
    tdir == -1 && (h0 = -h0)
    integrator.dt = h0
    return true
end

setewt!(integrator::JVIntegrator, u) = (@.. integrator.ewt = inv(integrator.opts.reltol * abs(u) + integrator.opts.abstol); return)
wrmsnorm(x, w) = sum(((x,w),)->abs2(x*w), zip(x, w))/length(x) |> sqrt

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
