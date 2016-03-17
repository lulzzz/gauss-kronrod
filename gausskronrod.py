#!/usr/bin/python3

from math import *
from scipy.integrate import quad

# nodes and weights vor Gauss-Kronrod
gausskronrod = (
    # node               weight Gauss       weight Kronrod
    (+0.949107912342759, 0.129484966168870, 0.063092092629979),
    (-0.949107912342759, 0.129484966168870, 0.063092092629979),
    (+0.741531185599394, 0.279705391489277, 0.140653259715525),
    (-0.741531185599394, 0.279705391489277, 0.140653259715525),
    (+0.405845151377397, 0.381830050505119, 0.190350578064785),
    (-0.405845151377397, 0.381830050505119, 0.190350578064785),
    ( 0.000000000000000, 0.417959183673469, 0.209482141084728),

    (+0.991455371120813, 0.000000000000000, 0.022935322010529),
    (-0.991455371120813, 0.000000000000000, 0.022935322010529),
    (+0.864864423359769, 0.000000000000000, 0.104790010322250),
    (-0.864864423359769, 0.000000000000000, 0.104790010322250),
    (+0.586087235467691, 0.000000000000000, 0.169004726639267),
    (-0.586087235467691, 0.000000000000000, 0.169004726639267),
    (+0.207784955007898, 0.000000000000000, 0.204432940075298),
    (-0.207784955007898, 0.000000000000000, 0.204432940075298)
)


def integrate_gausskronrod(f, a,b, args=()):
    """
    This function computes $\int_a^b \mathrm{d}x f(x)$ using Gauss-Kronrod
    quadrature formula. The integral is transformed
    z  = 2 \\frac{x-a}{b-a}-1
    x  = \\frac{b-a}{2} (z+1) + a
    dz = 2 \\frac{dx}{b-a}
    dx = \\frac{b-a}{2} dz
    \int_a^b \mathrm{d}x f(x) = \\frac{b-a}{2} \int_{-1}^1 \mathrm{d}z f((z+1)*(b-a)/2+a)

    returns integral and an error estimate
    """
    integral_G7  = 0
    integral_K15 = 0

    assert b > a
    dx = (b-a)/2

    for xi,wiG,wiK in gausskronrod:
        zi = (xi+1)/2*(b-a)+a
        fzi = f(zi, *args)

        integral_G7  += wiG*fzi
        integral_K15 += wiK*fzi

    error = (200*abs(integral_G7-integral_K15))**1.5

    return integral_K15*dx,dx*error


def integrate(f, a, b, minintervals=1, limit=200, tol=1e-10, args=()):
    """
    Do adaptive integration using Gauss-Kronrod.
    """
    intervals = []

    for i in range(minintervals):
        left  = a+(b-a)*i/minintervals
        right = a+(b-a)*(i+1)/minintervals
        I,err = integrate_gausskronrod(f,left,right,args)
        intervals.append((left,right,I,err))
    

    while True:
        err2 = 0
        Itotal = 0
        err_max = 0
        maximum = 0

        # search for largest error and its index
        for i in range(len(intervals)):
            I_i,err_i = intervals[i][2:]
            Itotal += I_i
            if err_i > err_max:
                err_max = err_i
                maximum = i

            err2 += err_i**2

        err = sqrt(err2)
        if abs(err/Itotal) < tol:
            return Itotal,err

        # no convergence
        if len(intervals) >= limit:
            return False

        # accuracy is still not good enough, so we split up the partial
        # integral with the larges error: [left,right] => [left,mid], [mid,right]
        left,right = intervals[maximum][0], intervals[maximum][1]

        # split integral
        mid = left+(right-left)/2

        # calculate integrals and errors, replace one item in the list and
        # append the other item to the end of the list
        I_left, err_left  = integrate_gausskronrod(f,left,mid,args)
        I_right,err_right = integrate_gausskronrod(f,mid,right,args)
        intervals[maximum] = (left,mid,I_left,err_left)
        intervals.append((mid,right,I_right,err_right))


if __name__ == "__main__":
    f = lambda x: 1e10

    print("%.15g, %15g" % quad(f, 1, 4))
    print("%.15g, %15g" % integrate_gausskronrod(f, 1,4))
    print("%.15g, %15g" % integrate(f, 1,4))
