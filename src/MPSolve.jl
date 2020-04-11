module MPSolve

import Base: complex
using Base.GMP: Limb
using Base.MPFR: MPFRRoundingMode,MPFRRoundNearest
export mps_roots

struct Mpz
    alloc::Cint
    size::Cint
    d::Ptr{Limb}

    Mpz() = Mpz(0)
    Mpz(b::T) where T<:Signed = Mpz(BigInt(b))
    function Mpz(b::BigInt)
        m = Ref{Mpz}()
        ccall((:__gmpz_init_set, :libgmp), Cvoid, (Ref{Mpz},Ref{BigInt}), m, b)
        return m[]
    end
end

Base.convert(::Type{Mpz}, x::T) where T<:Signed = Mpz(x)

function BigInt(m::Mpz)
    b = BigInt()
    ccall((:__gmpz_set, :libgmp), Cvoid, (Ref{BigInt},Ref{Mpz}), b, m)
    return b
end
             
struct Mpq
    num::Mpz
    den::Mpz

    Mpq() = Mpq(0)
    Mpq(i::T) where T<:Signed = Mpq(i, 1)
    Mpq(q::Rational{T}) where T<:Signed = Mpq(q.num, q.den)
    Mpq(num::T, den::S) where  {T, S<:Signed} = Mpq(BigInt(num), BigInt(den))
    function Mpq(num::BigInt, den::BigInt)
        q = Ref{Mpq}()
        ccall((:__gmpq_init, :libgmp), Cvoid, (Ref{Mpq},), q)
        ccall((:__gmpq_set_num, :libgmp), Cvoid, (Ref{Mpq},Ref{BigInt}),
              q, num)
        ccall((:__gmpq_set_den, :libgmp), Cvoid, (Ref{Mpq},Ref{BigInt}),
              q, den)
        return q[]
    end
    function Mpq(f::BigFloat)
        q = Ref{Mpq}()
        ccall((:__gmpq_init, :libgmp), Cvoid, (Ref{Mpq},), q)
        ccall((:mpfr_get_q, :libmpfr), Cvoid, (Ref{Mpq}, Ref{BigFloat}),
              q, f)
        return q[]
    end
end

Base.convert(::Type{Mpq}, x::T) where T<:Union{Signed,Rational} = Mpq(x)

Base.Rational(q::Mpq) = Rational(BigInt(q.num), BigInt(q.den))

(::Type{T})(q::Mpq) where T<: AbstractFloat = T(Rational(q))

_gmp_clear!(m::Mpz) = begin
    ccall((:__gmpz_clear, :libgmp), Cvoid, (Ref{Mpz},), m)
end

_gmp_clear!(m::Mpq) = begin
    ccall((:__gmpq_clear, :libgmp), Cvoid, (Ref{Mpq},), m)
end

# Arbitrary precision floating point type from gmp.h used by MPSolve
# is different from Julia's BigFloat
struct Mpf
    _mp_prec::Cint
    _mp_size::Cint
    _mp_exp::Clong
    _mp_d::Ptr{Limb}

    function Mpf(b::BigFloat)
        f = Ref{Mpf}()
        ccall((:__gmpf_init, :libgmp), Cvoid, (Ref{Mpf},), f)
        ccall((:mpfr_get_f, :libmpfr), Cint,
              (Ref{Mpf},Ref{BigFloat},MPFRRoundingMode),
              f, b, MPFRRoundNearest)
        return f[]
    end
 
    function Mpf(d::Float64)
        f = Ref{Mpf}()
        ccall((:__gmpf_init_set_d, :libgmp), Cvoid, (Ref{Mpf},Cdouble), f, d)
        return f[]
    end
    
    function Mpf(i::Int64)
        f = Ref{Mpf}()
        ccall((:__gmpf_init_set_si, :libgmp), Cvoid, (Ref{Mpf}, Clong), f, i)
        return f[]
    end
end

Base.convert(::Type{Mpf}, x::Int64) = Mpf(x)
Base.convert(::Type{Mpf}, x::T) where T<:Union{Int32,Int16,Int8} = Mpf(Int64(x))

function BigFloat(f::Mpf)
    b = BigFloat()
    ccall((:mpfr_set_f, :libmpfr), Cint,
          (Ref{BigFloat},Ref{Mpf},MPFRRoundingMode),
          b, f, MPFRRoundNearest)
    return b
end

struct Mpsc
    r::Mpf
    i::Mpf
end
Mpsc() = Mpsc(0, 0)

complex(m::Mpsc) = complex(BigFloat(m.r), BigFloat(m.i))

_gmp_clear!(f::Mpf) = begin
    ccall((:__gmpf_clear, :libgmp), Cvoid, (Ref{Mpf},), f)
    nothing
end

_gmp_clear!(c::Mpsc) = _gmp_clear!([c.r, c.i])

_gmp_clear!(m::Array{T}) where T<:Union{Mpz, Mpq, Mpf, Mpsc} = (_gmp_clear!.(m);
                                                                nothing)

struct Rdpe
    r::Cdouble
    e::Clong
end

Float64(d::Rdpe) = ccall((:rdpe_get_d, :libmps), Cdouble, (Ref{Rdpe},) ,d)

struct Cplx
     r::Cdouble
     i::Cdouble
end

Cplx(x) = Cplx(x, 0.0)
Cplx(x::Real, y::Real) = Cplx(Float64(x), Float64(y))
Cplx(z::Complex) = Cplx(Float64(z.re), Float64(z.im))
complex(m::Cplx) = Complex(m.r, m.i)

function new_context()
    context = ccall((:mps_context_new, :libmps), Ptr{Cvoid}, ())
end

function new_monomial(degree :: Integer)
    context = new_context()
    monomial_poly  = ccall((:mps_monomial_poly_new, :libmps), 
                           Ptr{Cvoid}, (Ptr{Cvoid}, Int), context, degree)
    return (context, monomial_poly)
end

function set_coefficients(context, monomial_poly, coefficients)
    local index
    function set_coefficient(cf::Complex{Float64})
        ccall((:mps_monomial_poly_set_coefficient_d, :libmps),
              Cvoid, (Ref{Cvoid}, Ref{Cvoid}, Clong, Cdouble, Cdouble),
              context, monomial_poly, index, cf.re, cf.im)
    end

    function set_coefficient(cf::Complex{T}) where  T<:Union{Signed,Rational,
                                                             BigFloat}
        c_re = Mpq(cf.re)
        c_im = Mpq(cf.im)
        ccall((:mps_monomial_poly_set_coefficient_q, :libmps),
              Cvoid, (Ref{Cvoid}, Ref{Cvoid}, Clong, Ref{Mpq}, Ref{Mpq}),
              context, monomial_poly, index, c_re, c_im)
        _gmp_clear!([c_re, c_im])
    end

    function set_coefficient(cf::Complex{T}) where T<:Integer
        c_re = Clonglong(cf.re)
        c_im = Clonglong(cf.im)
        ccall((:mps_monomial_poly_set_coefficient_int, :libmps),
              Cvoid, (Ref{Cvoid}, Ref{Cvoid}, Clong, Clonglong, Clonglong),
              context, monomial_poly, index, c_re, c_im)
    end

    set_coefficient(cf::T) where T<:Union{Float64,Signed,Rational,
                                          BigFloat,Integer} =
        set_coefficient(complex(cf))
    
    for (idx, cf) in enumerate(coefficients)
        index = idx-1
        set_coefficient(cf)
    end
end

function set_input_poly(context, poly)
    ccall((:mps_context_set_input_poly, :libmps), Cvoid, 
          (Ref{Cvoid}, Ref{Cvoid}), context, poly)
end

@enum mps_algorithm begin
    MPS_ALGORITHM_STANDARD_MPSOLVE
    MPS_ALGORITHM_SECULAR_GA
end

function select_algorithm(context, alg::mps_algorithm)
    ccall((:mps_context_select_algorithm, :libmps), Cvoid,
          (Ref{Cvoid}, Cint), context, alg)
end

@enum mps_output_goal begin
    MPS_OUTPUT_GOAL_ISOLATE
    MPS_OUTPUT_GOAL_APPROXIMATE
    MPS_OUTPUT_GOAL_COUNT
end

function set_output_goal(context, goal:: mps_output_goal)
    ccall((:mps_context_set_output_goal, :libmps), Cvoid,
          (Ref{Cvoid}, Cint), context, goal)
end

function set_output_precision(context, output_precision::Integer)
    ccall((:mps_context_set_output_prec, :libmps), Cvoid,
          (Ref{Cvoid}, Clong), context, output_precision)
end

function mpsolve(context)
    ccall((:mps_mpsolve, :libmps), Cvoid, (Ptr{Cvoid},), context)
end

function solve_poly(context, output_precision)
    select_algorithm(context, MPS_ALGORITHM_SECULAR_GA)
    set_output_goal(context, MPS_OUTPUT_GOAL_APPROXIMATE)
    set_output_precision(context, output_precision)
    mpsolve(context)
end


get_degree(context) = ccall((:mps_context_get_degree, :libmps), Cint,
                            (Ref{Cvoid},), context)

function get_roots(context)
    degree = get_degree(context)
    roots_m = Array{Mpsc}(undef, degree)
    for n=1:degree
        roots_m[n] = Mpsc()
    end
    rds = Array{Rdpe}(undef, degree)
    GC.@preserve roots_m rds begin
        roots_ptr = pointer(roots_m)
        rds_ptr = pointer(rds)
        ccall((:mps_context_get_roots_m, :libmps), Cint,
              (Ref{Cvoid}, Ref{Ptr{Mpsc}}, Ref{Ptr{Rdpe}}),
              context, roots_ptr, rds_ptr)
    end
    roots = complex.(roots_m)
    _gmp_clear!(roots_m)
    radii = Float64.(rds)
    (roots, radii)
end

function get_roots_d(context)
    degree =  get_degree(context)
    roots_c = Array{Cplx}(undef, degree)
    radii = Array{Cdouble}(undef, degree)
    GC.@preserve roots_c radii begin
        roots_ptr = pointer(roots_c)
        radii_ptr = pointer(radii)
        ccall((:mps_context_get_roots_d, :libmps), Cint,
              (Ref{Cvoid}, Ref{Ptr{Cplx}}, Ref{Ptr{Cdouble}}),
              context, roots_ptr, radii_ptr)
    end
    roots = complex.(roots_c)
    (roots,radii)
end

function free_monomial(context, monomial_poly)
    ccall((:mps_monomial_poly_free, :libmps), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}), context, monomial_poly)
    free_context(context)
end

function free_context(context)
    ccall((:mps_context_free, :libmps), Cvoid, (Ptr{Cvoid},), context)
end

"""
    (approximations, radii) = mps_roots(coefficients, output_precision)

Approximate the roots of the polynomial specified by the array
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
function mps_roots(coefficients::Array, output_precision::Integer=53)
    degree = length(coefficients)-1
    context,monomial_poly = new_monomial(degree)
    set_coefficients(context, monomial_poly, coefficients)
    set_input_poly(context, monomial_poly)
    solve_poly(context, output_precision)
    if output_precision <= 53
        (roots, radii) = get_roots_d(context)
    else
        (roots, radii) = get_roots(context)
    end
    free_monomial(context, monomial_poly)
    (roots, radii)
end

function new_secular(a_coeffs::Array, b_coeffs::Array)
    degree = length(a_coeffs) - 1
    if length(b_coeffs) != degree + 1
        throw(ArgumentError("input arrays must have equal length"))
    end
    context = new_context()
    a_cplx = Cplx.(a_coeffs)
    b_cplx = Cplx.(b_coeffs)
    GC.@preserve a_cplx b_cplx begin
        a_ptr = pointer(a_cplx)
        b_ptr = pointer(b_cplx)
        sec_eqn = ccall((:mps_secular_equation_new, :libmps), Ptr{Cvoid},
                        (Ptr{Cvoid}, Ref{Cplx}, Ref{Cplx}, Culong),
                        context, a_ptr, b_ptr, degree + 1)
    end
    return context,sec_eqn
end

function free_secular(context, secular_equation)
    ccall((:mps_polynomial_free, :libmps), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}), context, secular_equation)
    free_context(context)
end


function mps_roots(a_coeffs::Array, b_coeffs::Array, output_precision::Integer=53)
    context,secular_equation =  new_secular(a_coeffs, b_coeffs)
    set_input_poly(context, secular_equation)
    select_algorithm(context, MPS_ALGORITHM_STANDARD_MPSOLVE)
    mpsolve(context)
     if output_precision <= 53
        (roots, radii) = get_roots_d(context)
    else
        (roots, radii) = get_roots(context)
    end
    free_secular(context, secular_equation)
    return (roots, radii)
end
end
