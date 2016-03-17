#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    double left, right, I, err;
} interval_t;

/* nodes and weights vor Gauss-Kronrod */
static double gausskronrod[15][3] =
{
    /* node               weight Gauss       weight Kronrod */
    { +0.949107912342759, 0.129484966168870, 0.063092092629979 },
    { -0.949107912342759, 0.129484966168870, 0.063092092629979 },
    { +0.741531185599394, 0.279705391489277, 0.140653259715525 },
    { -0.741531185599394, 0.279705391489277, 0.140653259715525 },
    { +0.405845151377397, 0.381830050505119, 0.190350578064785 },
    { -0.405845151377397, 0.381830050505119, 0.190350578064785 },
    {  0.000000000000000, 0.417959183673469, 0.209482141084728 },

    { +0.991455371120813, 0.000000000000000, 0.022935322010529 },
    { -0.991455371120813, 0.000000000000000, 0.022935322010529 },
    { +0.864864423359769, 0.000000000000000, 0.104790010322250 },
    { -0.864864423359769, 0.000000000000000, 0.104790010322250 },
    { +0.586087235467691, 0.000000000000000, 0.169004726639267 },
    { -0.586087235467691, 0.000000000000000, 0.169004726639267 },
    { +0.207784955007898, 0.000000000000000, 0.204432940075298 },
    { -0.207784955007898, 0.000000000000000, 0.204432940075298 }
};


/** @brief Compute integral using Gauss-Kronrod quadrature
 *
 * This function computes \f$\int_a^b \mathrm{d}x f(x)\f$ using Gauss-Kronrod
 * quadrature formula. The integral is transformed according to
 * \f$z  = 2 \frac{x-a}{b-a}-1\f$
 * \f$x  = \frac{b-a}{2} (z+1) + a\f$
 * \f$dz = 2 \frac{dx}{b-a}\f$
 * \f$dx = \frac{b-a}{2} dz\f$
 * \f$\int_a^b \mathrm{d}x f(x) = \frac{b-a}{2} \int_{-1}^1 \mathrm{d}z f((z+1)*(b-a)/2+a)\f$
 *
 * @param [in]  f callback to integrand
 * @param [in]  a lower limit of integration
 * @param [in]  b upper limit of integration
 * @param [in]  args pointer to arbitrary data that is passed to f
 * @param [out] I calculated value of integral
 * @param [out] err estimated error
 */
double gausskronrod_integrate(double (*f)(double, void *), double a, double b, void *args, double *err)
{
    const double dx = (b-a)/2;
    double integral_G7  = 0;
    double integral_K15 = 0;

    for(int i = 0; i < 15; i++)
    {
        const double xi  = gausskronrod[i][0];
        const double wiG = gausskronrod[i][1];
        const double wiK = gausskronrod[i][2];

        const double zi  = (xi+1)*dx+a;
        const double fzi = f(zi, args);

        integral_G7  += wiG*fzi;
        integral_K15 += wiK*fzi;
    }

    if(err != NULL)
        *err = fabs(dx)*pow(200*fabs(integral_G7-integral_K15),1.5);

    return dx*integral_K15;
}


/** Compute integral using adaptive Gauss-Kronrod quadrature
 *
 * Do adaptive integration using Gauss-Kronrod.
 *
 * @param [in]  f callback to integrand
 * @param [in]  a lower limit of integration
 * @param [in]  b upper limit of integration
 * @param [in]  minintervals split integral in at least minintervals subintervals and perform Gauss-Kronrod quadrature
 * @param [in]  limit maximum number of subintervals
 * @param [in]  tol relative error tolerance
 * @param [in]  args pointer to arbitrary data that is passed to f
 * @param [out] I computed value of integral
 * @param [out] err estimated error
 *
 * @retval -1 if no convergence
 * @retval subintervals number of intervals used
 */
int gausskronrod_integrate_adaptive(double (*f)(double, void *), double a, double b, int minintervals, int limit, double tol, void *args, double *I, double *err)
{
    if(limit < minintervals)
        limit = minintervals;

    interval_t intervals[limit];

    int len;
    for(len = 0; len < minintervals; len++)
    {
        interval_t *interval = &intervals[len];
        interval->left  = a+ len   *(b-a)/minintervals;
        interval->right = a+(len+1)*(b-a)/minintervals;

        interval->I = gausskronrod_integrate(f, interval->left, interval->right, args, &interval->err);
    }
    
    while(true)
    {
        double err2 = 0, Itotal = 0, err_max = 0;
        int maximum = 0;

        /* search for largest error and its index, calculate integral and
         * error²
         */
        for(int i = 0; i < len; i++)
        {
            const double I_i   = intervals[i].I;
            const double err_i = intervals[i].err;

            Itotal += I_i;

            if(err_i > err_max)
            {
                err_max = err_i;
                maximum = i;
            }

            err2 += err_i*err_i;
        }

        if(fabs(sqrt(err2)/Itotal) < tol)
        {
            *I   = Itotal;
            *err = sqrt(err2);
            return len;
        }

        /* no convergence */
        if(len >= limit)
        {
            *I   = Itotal;
            *err = sqrt(err2);
            return -1;
        }

        /* accuracy is still not good enough, so we split up the partial
         * integral with the largest error:
         * [left,right] => [left,mid], [mid,right]
         */
        const double left  = intervals[maximum].left;
        const double right = intervals[maximum].right;
        const double mid   = left+(right-left)/2;

        /* calculate integrals and errors, replace one item in the list and
         * append the other item to the end of the list
         */
        double err_left, err_right;
        double I_left  = gausskronrod_integrate(f, left, mid,   args, &err_left);
        double I_right = gausskronrod_integrate(f, mid,  right, args, &err_right);
        
        intervals[maximum].left  = left;
        intervals[maximum].right = mid;
        intervals[maximum].I     = I_left;
        intervals[maximum].err   = err_left;
       
        intervals[len].left  = mid;
        intervals[len].right = right;
        intervals[len].I     = I_right;
        intervals[len].err   = err_right;

        /* increase len of array intervals */
        len++;
    }
}


double f(double x, void *ptr)
{
    return sin(x)*log(x);
}

int main(int argc, char *argv[])
{
    double I,err;
    double I2,err2;

    I = gausskronrod_integrate(f, 1, 100, NULL, &err);
    int i = gausskronrod_integrate_adaptive(f, 1, 100, 1, 100, 1e-12, NULL, &I2, &err2);

    printf("%.18g, %g\n", I, err);
    printf("%.18g, %g (%d)\n", I2, err2, i);

    return 0;
}
