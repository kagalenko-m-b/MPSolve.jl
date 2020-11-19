module MPSolve

include("MpsNumberTypes.jl")

using .MpsNumberTypes

export mps_roots,mps_barycentric_coeffs

abstract type PolyType end
struct Monomial <: PolyType end
struct Secular <: PolyType end

struct MContext{PolyType}
    cntxt_ptr::Ptr{Cvoid}
    poly::Ptr{Cvoid}
    #
    function MContext(coeffs::AbstractVector)
        degree = length(coeffs) - 1
        cntxt_ptr = new_cntxt_ptr()
        monomial_poly  = ccall(
        (:mps_monomial_poly_new, :libmps),
        Ptr{Cvoid}, (Ptr{Cvoid}, Int),
        cntxt_ptr, degree
        )
        context = new{Monomial}(cntxt_ptr, monomial_poly)
        set_coefficients!(context, coeffs)
        set_input_poly(context)
        return context
    end
    #
    function MContext(a_coeffs::AbstractVector, b_coeffs::AbstractVector)
        degree = length(a_coeffs)
        if length(b_coeffs) != degree
            throw(ArgumentError("input arrays must have equal length"))
        end
        cntxt_ptr = new_cntxt_ptr()
        sec_eqn = ccall((:mps_secular_equation_new_raw, :libmps), Ptr{Cvoid},
                        (Ptr{Cvoid}, Culong),
                        cntxt_ptr, degree
                        )
        context = new{Secular}(cntxt_ptr, sec_eqn)
        set_coefficients!(context, a_coeffs, b_coeffs)
        set_input_poly(context)
        return context
    end
end


function new_cntxt_ptr()
    cntxt_ptr = ccall((:mps_context_new, :libmps), Ptr{Cvoid}, ())
end

function set_coefficient!(context::MContext{Monomial}, cf::Complex{Float64}, index)
    ccall(
        (:mps_monomial_poly_set_coefficient_d, :libmps),
        Cvoid, (Ref{Cvoid}, Ref{Cvoid}, Clong, Cdouble, Cdouble),
        context.cntxt_ptr, context.poly, index, cf.re, cf.im
    )
end

function set_coefficient!(context::MContext{Monomial}, cf, index)
    c_re = Mpq(real(cf))
    c_im = Mpq(imag(cf))
    ccall(
        (:mps_monomial_poly_set_coefficient_q, :libmps),
        Cvoid, (Ref{Cvoid}, Ref{Cvoid}, Clong, Ref{Mpq}, Ref{Mpq}),
        context.cntxt_ptr, context.poly, index, c_re, c_im
    )
    mps_clear!([c_re, c_im])
end

function set_coefficient!(
    context::MContext{Monomial}, cf::Complex{T}, index
) where T<:Union{Int64,Int32,Int16,Int8}
    c_re = Clonglong(real(cf))
    c_im = Clonglong(real(cf))
    ccall(
        (:mps_monomial_poly_set_coefficient_int, :libmps),
        Cvoid, (Ref{Cvoid}, Ref{Cvoid}, Clong, Clonglong, Clonglong),
        context.cntxt_ptr, context.poly, index, c_re, c_im
    )
end

function set_coefficients!(context::MContext{Monomial}, coeffs::AbstractVector)
    for (idx, cf) in enumerate(coeffs)
        set_coefficient!(context, complex(cf), idx - 1)
    end
    return nothing
end

function set_coefficients!(
    context::MContext{Secular}, a_coeffs::AbstractVector, b_coeffs::AbstractVector
)
    for (k, (a,b)) in enumerate(zip(a_coeffs, b_coeffs))
        a_re = Mpq(real(a))
        a_im = Mpq(imag(a))
        b_re = Mpq(real(b))
        b_im = Mpq(imag(b))
        ccall(
            (:mps_secular_equation_set_coefficient_q, :libmps),
            Cvoid, (Ref{Cvoid}, Ref{Cvoid}, Clong, Ref{Mpq}, Ref{Mpq}, Ref{Mpq}, Ref{Mpq}),
            context.cntxt_ptr, context.poly, k - 1, a_re, a_im, b_re, b_im
        )
        mps_clear!([a_re, a_im, b_re, b_im])
    end
    return nothing
end

function set_input_poly(context::MContext)
    ccall((:mps_context_set_input_poly, :libmps), Cvoid,
          (Ref{Cvoid}, Ref{Cvoid}), context.cntxt_ptr, context.poly)
end

@enum mps_algorithm begin
    MPS_ALGORITHM_STANDARD_MPSOLVE
    MPS_ALGORITHM_SECULAR_GA
end

function select_algorithm(context::MContext, alg::mps_algorithm)
    ccall(
        (:mps_context_select_algorithm, :libmps), Cvoid,
        (Ref{Cvoid}, Cint), context.cntxt_ptr, alg
    )
end

@enum mps_output_goal begin
    MPS_OUTPUT_GOAL_ISOLATE
    MPS_OUTPUT_GOAL_APPROXIMATE
    MPS_OUTPUT_GOAL_COUNT
end

function set_output_goal(context::MContext, goal:: mps_output_goal)
    ccall(
        (:mps_context_set_output_goal, :libmps),
        Cvoid, (Ref{Cvoid}, Cint),
        context.cntxt_ptr, goal
    )
end

function set_output_precision(context::MContext, output_precision::Integer)
    ccall(
        (:mps_context_set_output_prec, :libmps), Cvoid,
        (Ref{Cvoid}, Clong),
        context.cntxt_ptr, output_precision
    )
end

function mpsolve(context::MContext)
    ccall((:mps_mpsolve, :libmps), Cvoid, (Ptr{Cvoid},), context.cntxt_ptr)
end

function solve_poly(context::MContext, output_precision::Integer)
    select_algorithm(context, MPS_ALGORITHM_SECULAR_GA)
    set_output_goal(context, MPS_OUTPUT_GOAL_APPROXIMATE)
    set_output_precision(context, output_precision)
    mpsolve(context)
end


function get_degree(context::MContext)
    ccall((:mps_context_get_degree, :libmps), Cint, (Ref{Cvoid},), context.cntxt_ptr)
end

function get_roots(context::MContext, output_precision::Int)
    if output_precision <= 53
        roots,radii = get_roots_f64(context)
    else
        roots,radii = get_roots_big(context)
    end
    return roots, radii
end

function get_roots_big(context::MContext)
    degree = get_degree(context)
    roots_m = Array{Mpsc}(undef, degree)
    for n=1:degree
        roots_m[n] = Mpsc()
    end
    rds = Array{Rdpe}(undef, degree)
    GC.@preserve roots_m rds begin
        roots_ptr = pointer(roots_m)
        rds_ptr = pointer(rds)
        ccall(
            (:mps_context_get_roots_m, :libmps), Cint,
            (Ref{Cvoid}, Ref{Ptr{Mpsc}}, Ref{Ptr{Rdpe}}),
            context.cntxt_ptr, roots_ptr, rds_ptr
        )
    end
    roots = complex.(roots_m)
    mps_clear!(roots_m)
    radii = Float64.(rds)
    return (roots, radii)
end

function get_roots_f64(context::MContext)
    degree =  get_degree(context)
    roots_c = Array{Cplx}(undef, degree)
    radii = Array{Cdouble}(undef, degree)
    GC.@preserve roots_c radii begin
        roots_ptr = pointer(roots_c)
        radii_ptr = pointer(radii)
        ccall(
            (:mps_context_get_roots_d, :libmps), Cint,
            (Ref{Cvoid}, Ref{Ptr{Cplx}}, Ref{Ptr{Cdouble}}),
            context.cntxt_ptr, roots_ptr, radii_ptr
        )
    end
    roots = complex.(roots_c)
    (roots,radii)
end


"""
    (approximations, radii) = mps_roots(coefficients, output_precision=53)

Approximate the roots of a polynomial specified by the array
of its coefficients. Output precision is specified in bits.

# Example
```jldoctest
julia> N = 64;

julia> cfs = zeros(Int, N + 1); cfs[end] = -(cfs[1] = 1);

julia> (app, rad) = mps_roots(cfs, 100);

julia> all(map(x->abs((x^N - 1)/(N*x^(N - 1))), app) < rad)
true
```
"""
function mps_roots(coefficients::AbstractVector,  output_precision::Integer=53)
    return mps_roots(coefficients; output_precision=output_precision)
end

"""
    (approximations, radii) = mps_roots(A, B, output_precision=53)

Approximate the roots of a secular equation. Output precision is specified in bits.

# Example
```jldoctest
julia>  S(x) = 1/(x - 2) + 3/(x - 4) + 5/(x - 6) - 1;

julia> rt,rad=mps_roots([1, 3, 5],[2, 4, 6]);

julia> rtb,radb=mps_roots([1, 3, 5],[2, 4, 6],output_precision=256);

julia> all(abs.(rt-rtb) .< max.(rad,abs.(rtb)*eps()))
true

```
"""
function mps_roots(A::AbstractVector, B::AbstractVector, output_precision::Integer=53)
    return mps_roots(A, B; output_precision=output_precision)
end

function mps_roots(coeffs...; output_precision::Integer=53)
    context = MContext(coeffs...)
    solve_poly(context, output_precision)
    (roots, radii) = get_roots(context, output_precision)
    free_context(context)
    (roots, radii)
end


raw"""
    D,L,F = barycentric_coeffs_mps(f::Function; M::Int=0, D::AbstractVector=[])

Coefficients of barycentric form of Lagrange interpolation of the form used by MPSolve

       f(z)  ->

                /    L[0]                    L[M]              \
            -> |  ----------*F[0] + ... + ----------*F[M]  - 1  |
                \  z - D[0]                z - D[M]            /

                                  /  /    L[0]                 L[M]    \
                                 /  |  ---------- + ... +   ----------  |
                                /    \  z - D[0]             z - D[M]  /

    If D not specified, set them to the roots of unity:

                    /  im*pi*k  \
        D[k] = exp |  ---------  | ,   F[k] = f(D[k])
                    \     M     /

    For polynomial f() of the degree M, f(z) === f(C,D,L,F,z), where

     f(C,D,L,F,z)= any(z.==D) ? C*F[findfirst(z.==D)] : C*(eta(D,L.*F,z) - 1)/eta(D,L,z)

    See Berrut, Trefethen, "Barycentric Lagrange Interpolation," SIAM Review,
    Vol. 46, No. 3, pp. 501-517, 2004
"""
function mps_barycentric_coeffs(f::Function; M::Int=0, D::AbstractVector=[])
    F0 = f(0)
    T = promote_type(typeof(F0), Irrational)
    if isempty(D)
        D =  exp.(2im*T(pi)*(1:M)/M)
        L = D/M
    elseif M == 0
        M == length(D)
        L = [1/(prod(D[k] .- D[1:k - 1])*prod(D[k] .- D[k + 1:end])) for k in 1:M]
    else
        throw(ArgumentError("specify either M or D"))
    end
    F = f.(D)
    C = F0*sum(L./D) - sum((L.*F)./D) #;println("C=$(C)")
    C != 0 || error("function appears to be a polynomial of the degree less than $(M)")
    F /= C
    return D,L,F
end


function free_poly(context::MContext{Monomial})
    ccall(
        (:mps_monomial_poly_free, :libmps), Cvoid,
        (Ptr{Cvoid}, Ptr{Cvoid}),
        context.cntxt_ptr, context.poly
    )
end

function free_poly(context::MContext{Secular})
    ccall(
        (:mps_secular_equation_free, :libmps), Cvoid,
        (Ptr{Cvoid}, Ptr{Cvoid}),
        context.cntxt_ptr, context.poly
    )
end

function free_context(context::MContext)
    free_poly(context)
    ccall((:mps_context_free, :libmps), Cvoid, (Ptr{Cvoid},), context.cntxt_ptr)
end
end
