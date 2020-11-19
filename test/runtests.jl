# Testing for the MPSolve.jl package.

using MPSolve: Mpz,Mpq,Mpf,Cplx,mps_clear!,mps_roots,mpsf_precision,mps_barycentric_coeffs
using Test

unity_roots(n) = [exp(j*2im*BigFloat(pi)/n) for j = 1:n]

function roots2coeffs(roots)
    coeffs = zeros(big(eltype(roots)), length(roots) + 1)
    coeffs[1] = 1
    for i = 1:length(roots)
        c = -roots[i]*coeffs[1:i + 1]
        c[2:i + 1] += coeffs[1:i]
        coeffs[1:i + 1] = c
    end
    coeffs
end

# function roots2secular_coeffs(roots)
#     M = length(roots)
#     T = eltype(roots)
#     D = exp.(im*T(pi)*range(1, stop=2M-1, length=M)/T(M))
#     L = -D/M
#     F = [-prod(d .- roots) for d in D]
#     return L.*F,D
# end

function solve_test(roots, args...; output_prec=53)
    (app, rad) = mps_roots(args..., output_prec)
    T = real(eltype(app))
    for rt in roots
        (err, ind) = findmin(map(abs, app .- rt))
        @test err <= max(rad[ind], 1e4*eps(T))
    end
end

function test_roots_of_unity(n)
    p = Int64[0 for i = 1:n + 1]
    p[1] = 1
    p[end] = -1
    solve_test(unity_roots(n), p)
end

function test_roots_of_unity_fp(n)
    p = zeros(n+1)
    p[1] = 1.0
    p[end] = -1.0
    solve_test(unity_roots(n), p)
end

function test_roots_of_unity_bigint(n)
    p = [BigInt(0) for i = 1 : n + 1]
    p[1] = BigInt(1)
    p[end] = BigInt(-1)
    solve_test(unity_roots(n), p)
end

function test_roots_of_unity_bigfloat(n)
    p = zeros(BigFloat, n + 1)
    p[1] = 1
    p[end] = -1
    solve_test(unity_roots(n), p)
end

function test_secular_roots_unity(n)
    roots = unity_roots(n)
    f(x) = prod(x .- roots)
    D,L,F = mps_barycentric_coeffs(f, M=n)
    solve_test(roots, L.*F, D, output_prec=256)
end
"""
Test solving the Wilkinson polynomial
"""
function test_wilkinson(n)
    roots = range(1, stop=n)
    C = roots2coeffs(roots)
    solve_test(roots, C)
end

function test_secular_wilkinson(n)
    roots = range(-n, stop=n*big(1.0), length=n)/n
    f(x) = prod(x .- roots)
    D,L,F = mps_barycentric_coeffs(f, M=n)
    solve_test(roots, L.*F, D)
end
"""
Test if solving a polynomial with complex integer coefficients
works.
"""

function test_complex_int()
    sols = [ 2 ; 3+1im ; 5-1im ; -2 ]
    p = roots2coeffs(sols)
    solve_test(sols, p)
end

function test_complex_bigint()
    sols = Complex{BigInt}[ 2 ; 3+1im ; 5-1im ; -2 ]
    p = roots2coeffs(sols)
    solve_test(sols, p)
end

if VERSION < v"1.3"
    error("This module only works with Julia version 1.3 or greater")
end

@testset "Types for interfacing with MPSolve" begin
    @test mpsf_precision() == precision(BigFloat)
    v = [Mpz(), Mpz(Int32(1234567890)), convert(Mpz, big"123456789012345678901234567890")]
    @test eltype(v) == Mpz && eltype(big.(v)) == BigInt
    @test big.(v) == [0, 1234567890, 123456789012345678901234567890]
    mps_clear!(v)
    #
    v = Mpq.(Any[big(pi), rationalize(Float64(pi), tol=eps(BigFloat))])
    @test eltype(v) == Mpq && eltype(big.(v)) == BigFloat
    @test big.(v) == [big(pi), Float64(pi)]
    mps_clear!(v)
    #
    v = Mpf.(Any[Float64(pi), Int32(1234567890)])
    @test eltype(v) == Mpf && eltype(big.(v)) == BigFloat
    #
    v = Mpf(big(pi))
    @test typeof(v) == Mpf && typeof(big(v)) == BigFloat && big(v) == big(pi)
    mps_clear!(v)
    #
    @test iszero(complex(Cplx()))
end

@testset "MPSolve routines" begin
    for n in 99:100
        test_roots_of_unity(n)
        test_roots_of_unity_fp(n)
        test_roots_of_unity_bigint(n)
        test_roots_of_unity_bigfloat(n)
        test_wilkinson(n)
        test_secular_roots_unity(n)
        test_secular_wilkinson(n)
    end
    test_complex_int()
    test_complex_bigint()
end
