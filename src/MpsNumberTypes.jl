module MpsNumberTypes

using Base.GMP: Limb
using Base.MPFR: MPFRRoundingMode,MPFRRoundNearest

export Mpq,Mpf,Mpsc,Rdpe,Cplx,mps_clear!

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
        ccall((:__gmpq_set_num, :libgmp), Cvoid, (Ref{Mpq},Ref{BigInt}), q, num)
        ccall((:__gmpq_set_den, :libgmp), Cvoid, (Ref{Mpq},Ref{BigInt}), q, den)
        return q[]
    end
    function Mpq(f::BigFloat)
        q = Ref{Mpq}()
        ccall((:__gmpq_init, :libgmp), Cvoid, (Ref{Mpq},), q)
        ccall((:mpfr_get_q, :libmpfr), Cvoid, (Ref{Mpq}, Ref{BigFloat}), q, f)
        return q[]
    end
end
Mpq(f::AbstractFloat) = Mpq(big(f))
Base.convert(::Type{Mpq}, x::T) where T<:Union{Signed,Rational} = Mpq(x)
Base.Rational(q::Mpq) = Rational(BigInt(q.num), BigInt(q.den))
(::Type{T})(q::Mpq) where T<: AbstractFloat = T(Rational(q))
mps_clear!(m::Mpz) = ccall((:__gmpz_clear, :libgmp), Cvoid, (Ref{Mpz},), m)
mps_clear!(m::Mpq) = ccall((:__gmpq_clear, :libgmp), Cvoid, (Ref{Mpq},), m)

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
        ccall(
            (:mpfr_get_f, :libmpfr),
            Cint, (Ref{Mpf},Ref{BigFloat},MPFRRoundingMode),
            f, b, MPFRRoundNearest
        )
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

Base.convert(::Type{Mpf}, x::T) where T<:Union{Int32,Int16,Int8} = Mpf(Int64(x))
Base.convert(::Type{Mpf}, x::AbstractFloat) = Mpf(x)

function BigFloat(f::Mpf)
    b = BigFloat()
    ccall(
        (:mpfr_set_f, :libmpfr),
        Cint, (Ref{BigFloat},Ref{Mpf},MPFRRoundingMode),
        b, f, MPFRRoundNearest
    )
    return b
end

struct Mpsc
    r::Mpf
    i::Mpf
end
Mpsc(x) = Mpsc(x, 0)
Mpsc() = Mpsc(0)

mps_clear!(f::Mpf) = ccall((:__gmpf_clear, :libgmp), Cvoid, (Ref{Mpf},), f)
mps_clear!(c::Mpsc) = mps_clear!([c.r, c.i])
mps_clear!(m::Array{T}) where T<:Union{Mpz, Mpq, Mpf, Mpsc} = (mps_clear!.(m);nothing)

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
Cplx(z::Complex) = Cplx(z.re, z.im)


end
