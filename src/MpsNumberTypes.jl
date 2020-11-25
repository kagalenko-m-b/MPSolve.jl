module MpsNumberTypes

using Base.GMP: Limb
using Base.MPFR: MPFRRoundingMode,MPFRRoundNearest
using MPSolve_jll
import Base: big, complex, convert

export Mpz,Mpq,Mpf,Mpsc,Rdpe,Cplx,mpsf_precision,mpsf_setprecision,mps_clear!

function __init__()
    try
        # set GMP floating types precision to current precsion of julia's BigFloat
        mpsf_setprecision(precision(BigFloat))
    catch ex
        Base.showerror_nostdio(ex, "WARNING: Error during initialization of MpsNumberTypes")
    end
    nothing
end

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
mps_clear!(m::Mpz) = ccall((:__gmpz_clear, :libgmp), Cvoid, (Ref{Mpz},), m)
big(::Type{Mpz}) = BigInt
big(v::Mpz) = BigInt(v)

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
mps_clear!(m::Mpq) = ccall((:__gmpq_clear, :libgmp), Cvoid, (Ref{Mpq},), m)

Mpq(f::AbstractFloat) = Mpq(big(f))
Mpq(c::Complex) = Mpq.(reim(c))
Base.convert(::Type{Mpq}, x::T) where T<:Union{Signed,Rational} = Mpq(x)
Base.Rational(q::Mpq) = Rational(BigInt(q.num), BigInt(q.den))
(::Type{T})(q::Mpq) where T<: AbstractFloat = T(Rational(q))
big(::Type{Mpq}) = BigFloat
big(v::Mpq) = BigFloat(v)

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
            Cint,
            (Ref{Mpf},Ref{BigFloat},MPFRRoundingMode),
            f,
            b,
            MPFRRoundNearest
        )

        return f[]
    end

    function Mpf(b::BigInt)
        f = Ref{Mpf}()
        ccall((:__gmpf_init, :libgmp), Cvoid, (Ref{Mpf},), f)
        ccall(
            (:__gmpf_set_z, :libgmp),
            Cvoid, (Ref{Mpf}, Ref{BigInt}),
            f,
            b
        )

        return f[]
    end

    function Mpf(d::Float64)
        f = Ref{Mpf}()
        ccall((:__gmpf_init_set_d, :libgmp), Cvoid, (Ref{Mpf},Cdouble), f, d)
        return f[]
    end

    function Mpf(i::Int32)
        f = Ref{Mpf}()
        ccall((:__gmpf_init_set_si, :libgmp), Cvoid, (Ref{Mpf}, Clong), f, i)

        return f[]
    end
end

Base.convert(::Type{Mpf}, x::Signed) = Mpf(BigInt(x))
Base.convert(::Type{Mpf}, x::T) where T<:Union{Int32,Int16,Int8} = Mpf(Int32(x))
Base.convert(::Type{Mpf}, x::AbstractFloat) = Mpf(x)

function BigFloat(f::Mpf)
    b = BigFloat()
    ccall(
        (:mpfr_set_f, :libmpfr),
        Cint,
        (Ref{BigFloat},Ref{Mpf},MPFRRoundingMode),
        b,
        f,
        MPFRRoundNearest
    )
    return b
end

big(::Type{Mpf}) = BigFloat
big(v::Mpf) = BigFloat(v)

mpsf_precision() = ccall((:__gmpf_get_default_prec, :libgmp), Culong, ())
mpsf_setprecision(p) = ccall((:__gmpf_set_default_prec, :libgmp), Cvoid, (Culong,), p)

struct Mpsc
    r::Mpf
    i::Mpf
end
Mpsc(x) = Mpsc(x, 0)
Mpsc() = Mpsc(0)
Base.complex(m::Mpsc) = complex(BigFloat(m.r), BigFloat(m.i))

mps_clear!(f::Mpf) = ccall((:__gmpf_clear, :libgmp), Cvoid, (Ref{Mpf},), f)
mps_clear!(c::Mpsc) = mps_clear!([c.r, c.i])
mps_clear!(m...) = mps_clear!.(m)
mps_clear!(m::AbstractArray) = mps_clear!.(m)

struct Rdpe
    r::Cdouble
    e::Clong
end

Float64(d::Rdpe) = ccall((:rdpe_get_d, libmps), Cdouble, (Ref{Rdpe},) ,d)

struct Cplx
     r::Cdouble
     i::Cdouble
end

Cplx() = Cplx(0.0)
Cplx(x) = Cplx(x, 0.0)
Cplx(z::Complex) = Cplx(z.re, z.im)
Base.complex(m::Cplx) = complex(Float64(m.r), Float64(m.i))


end
