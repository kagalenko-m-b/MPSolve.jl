# Testing for the MPSolve.jl package. 

using MPSolve
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

function roots2secular_coeffs(roots)
    M = length(roots)
    R = 1.01
    D = R*exp.(im*pi*range(1/M, stop=2-1/M, length=M))
    L = -D/(M*R^M)
    F = [-prod(d .- roots) for d in D]
    return L.*F,D
end

function solve_test(rts, args...;n_digits=nothing)
    if isnothing(n_digits)
        if real(eltype(args[1])) <:Union{BigFloat,BigInt}
            n_digits=55
        else
            n_digits=53
        end
    end
    (app, rad) = mps_roots(args...,n_digits)
    for i = 1:length(rts)
        (err, ind) = findmin(map(abs, app .- rts[i]))
        if isnan(rad[ind])
            err_radius = 10*eps(real(eltype(app)))
        else
            err_radius = max(rad[ind],
                             abs(app[ind])*eps(real(eltype(app))))
        end
        @test err <= err_radius
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
    E = unity_roots(n)
    A,B = roots2secular_coeffs(E)
    solve_test(E, ComplexF64.(A), ComplexF64.(B), n_digits=54)
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
    roots = big.(range(1, stop=n))
    A,B = roots2secular_coeffs(roots)
    solve_test(roots, A, B)
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

for N = 99:100
    test_roots_of_unity(N)
    test_roots_of_unity_fp(N)
    test_roots_of_unity_bigint(N)
    test_roots_of_unity_bigfloat(N)
    test_secular_roots_unity(N)
    test_wilkinson(N)
end
test_complex_int()
test_complex_bigint()
