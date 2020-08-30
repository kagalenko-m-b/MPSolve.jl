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
    T = eltype(roots)
    D = exp.(im*T(pi)*range(1, stop=2M-1, length=M)/T(M))
    L = -D/M
    F = [-prod(d .- roots) for d in D]
    return L.*F,D
end

function solve_test(rts, args; output_prec=53)
    (app, rad) = mps_roots(args, output_prec)
    T = real(eltype(app))
    for i = 1:length(rts)
        (err, ind) = findmin(map(abs, app .- rts[i]))
        @test err <= max(rad[ind], sqrt(eps(T)))
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
    solve_test(E, (A, B), output_prec=256)
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
    # test_wilkinson(N)
    test_secular_roots_unity(N)
end
test_complex_int()
test_complex_bigint()
