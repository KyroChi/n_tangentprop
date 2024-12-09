import torch

from sympy import symbols, Function, Derivative, diff, re

def compute_exact_implicit(U, m, a, A, n_derivs=3):
    """
    Computes the implicit solution to the Burgers equation and its
    (implicit) derivatives.

    This function uses Sympy to compute the derivatives and then
    evaluates the results numerically.

    Attributes:
    ===========
    U: A torch.Tensor giving the upper and lower bounds of U. Usually
    a torch.linspace(...).reshape(-1, 1) call.
    m: The exponent (lambda) to compute the solution for.
    a: The boundary condition point.
    A: The constrained value U(a) = A.
    n_derivs: How many derivatives to take. Default = 3.

    Returns:
    ========
    Array: [X, U, U', ..., U^{(n)}] of tensors. Plot as plt.plot(X,
    U), etc.
    """
    U_derivs = [torch.Tensor(len(U)) for _ in range(n_derivs + 2)]
    U_derivs[1] = U

    C = - ( a + A ) / ( A*abs(A)**(1/m) )

    U_derivs[0] = -U - C * torch.sign(U) * torch.abs(U)**(1 + 1/m)
    #U_derivs[2] = - 1 / ( 1 + C * (1 + 1/m) * torch.abs(U)**(1/m) )

    X = symbols('X')
    U = Function('U')
    U_X = - 1 / ( 1 + C * (1 + 1/m) * U(X)**(1/m))

    sym_derivs = [U_X]
    for ii in range(n_derivs - 1):
        sym_derivs.append(diff(sym_derivs[-1], X))

    for ii in range(n_derivs):
        for jj in range(len(U_derivs[1])):
            repl = { Derivative(U(X), (X, kk)): U_derivs[kk + 1][jj] for kk in range(ii + 1) }

            U_derivs[ii + 2][jj] = float(re(sym_derivs[ii].xreplace(repl)))

    return U_derivs