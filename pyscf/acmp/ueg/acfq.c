#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define MP2_ARGS \
    double *polc, double *azic, double *radc, double *angw, \
    double *dv, int nsph, int nrad, const double q, \
    const double param, const double kfermi, double *dres, double *xres

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEFINE_VARIABLES_OUTER_LOOP \
    double dsubtot = 0; \
    double xsubtot = 0; \
    double k[2]; \
    double p[2]; \
    double ediff, wt0, wt1, d0, d1; \
    double xk = get_xk(radc[i0], q); \
    double xp, mulp, addp, cost, sint; \
    double mulk = 0.5 * (1 - xk); \
    double addk = 0.5 * (1 + xk); \
    if (xk >= 1) { continue; }

#define SET_KVECTORS_INNER_LOOP \
    cost = polc[a0] * mulk + addk; \
    sint = sqrt(1 - cost * cost); \
    k[0] = radc[i0] * sint; \
    k[1] = radc[i0] * cost; \
    cost = polc[a1] * mulp + addp; \
    sint = sqrt(1 - cost * cost); \
    p[0] = radc[i1] * sint; \
    p[1] = radc[i1] * cost;

#define ADD_SUBTOTAL dres[i0] += dsubtot; xres[i0] += xsubtot;

inline double epsx_ueg(double x)
{
    double fac = fabs(1 - x) + 1e-16;
    fac = log(fabs(1 + x) / fac);
    fac *= 0.5 * (1 - x * x) / x;
    fac += 1;
    fac /= M_PI;
    return fac;
}

double epsx_ueg_py(double x)
{
    return epsx_ueg(x);
}

inline double get_xk(double k, double q) {
    return MAX((1 - k * k - q * q) / (2 * k * q), -1);
}

inline double get_xp(double k, double q) {
    return MIN((-1 + k * k + q * q) / (2 * k * q), 1);
}

void get_exch_eigval_contribs(double *ex0, double *ex1, double *ex2,
                              double *radc, double *polc, double q,
                              const double invk, int nrad, int nsph) {
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        double k[2];
        int ind;
        double xk = get_xk(radc[i0], q);
        double mulk = 0.5 * (1 - xk);
        double addk = 0.5 * (1 + xk);
        double cost, sint;
        if (xk < 1) {
            for (int a0 = 0; a0 < nsph; a0++) {
                cost = polc[a0] * mulk + addk;
                sint = sqrt(1 - cost * cost);
                k[0] = radc[i0] * sint;
                k[1] = radc[i0] * cost;
                ind = i0 * nsph + a0;
                ex0[ind] = invk * epsx_ueg(sqrt(k[0] * k[0] + k[1] * k[1]));
                ex1[ind] = invk * epsx_ueg(sqrt(k[0] * k[0] + (k[1] + q) * (k[1] + q)));
            }
        }
        xk = get_xp(radc[i0], q);
        mulk = 0.5 * (1 + xk);
        addk = 0.5 * (xk - 1);
        if (xk > -1) {
            for (int a0 = 0; a0 < nsph; a0++) {
                cost = polc[a0] * mulk + addk;
                sint = sqrt(1 - cost * cost);
                k[0] = radc[i0] * sint;
                k[1] = radc[i0] * cost;
                ind = i0 * nsph + a0;
                ex2[ind] = invk * epsx_ueg(sqrt(k[0] * k[0] + (k[1] - q) * (k[1] - q)));
            }
        }
    }
}

void get_tau_fac_contribs(double *kfac1, double *kfac2, double *radc, double *polc,
                          double q, const double invk, int nrad, int nsph)
{
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        double k[2];
        int ind;
        double xk = get_xk(radc[i0], q);
        double mulk = 0.5 * (1 - xk);
        double addk = 0.5 * (1 + xk);
        double cost, sint;
        const double power = 0.75;
        if (xk < 1) {
            for (int a0 = 0; a0 < nsph; a0++) {
                cost = polc[a0] * mulk + addk;
                sint = sqrt(1 - cost * cost);
                k[0] = radc[i0] * sint;
                k[1] = radc[i0] * cost;
                double kin = k[0] * k[0] + (k[1] + q) * (k[1] + q);
                ind = i0 * nsph + a0;
                kfac1[ind] = pow(0.5 * kin, power) * pow(invk, 2 * (1 - power));
            }
        }
        xk = get_xp(radc[i0], q);
        mulk = 0.5 * (1 + xk);
        addk = 0.5 * (xk - 1);
        if (xk > -1) {
            for (int a0 = 0; a0 < nsph; a0++) {
                cost = polc[a0] * mulk + addk;
                sint = sqrt(1 - cost * cost);
                k[0] = radc[i0] * sint;
                k[1] = radc[i0] * cost;
                double kin = k[0] * k[0] + (k[1] - q) * (k[1] - q);
                ind = i0 * nsph + a0;
                kfac2[ind] = pow(0.5 * kin, power) * pow(invk, 2 * (1 - power));
            }
        }
    }
}

void get_exch_kappa_contribs(double *ex0, double *ex1, double *ex2,
                             double *radc, double *polc, double q,
                              const double invk, int nrad, int nsph) {
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        double k[2];
        int ind;
        double xk = get_xk(radc[i0], q);
        double mulk = 0.5 * (1 - xk);
        double addk = 0.5 * (1 + xk);
        double cost, sint;
        if (xk < 1) {
            for (int a0 = 0; a0 < nsph; a0++) {
                cost = polc[a0] * mulk + addk;
                sint = sqrt(1 - cost * cost);
                k[0] = radc[i0] * sint;
                k[1] = radc[i0] * cost;
                ind = i0 * nsph + a0;
                ex0[ind] = invk * epsx_ueg(sqrt(k[0] * k[0] + k[1] * k[1]));
                ex1[ind] = invk * epsx_ueg(sqrt(k[0] * k[0] + (k[1] + q) * (k[1] + q)));
            }
        }
        xk = get_xp(radc[i0], q);
        mulk = 0.5 * (1 + xk);
        addk = 0.5 * (xk - 1);
        if (xk > -1) {
            for (int a0 = 0; a0 < nsph; a0++) {
                cost = polc[a0] * mulk + addk;
                sint = sqrt(1 - cost * cost);
                k[0] = radc[i0] * sint;
                k[1] = radc[i0] * cost;
                ind = i0 * nsph + a0;
                ex2[ind] = invk * epsx_ueg(sqrt(k[0] * k[0] + (k[1] - q) * (k[1] - q)));
            }
        }
    }
}

void get_kappa_contribs(double *kfactors0, double *kfactors1,
                        double *radc, double *polc, double q,
                        const double kappa, int nrad, int nsph) {
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        double k[2];
        int ind;
        double xk = get_xk(radc[i0], q);
        double mulk = 0.5 * (1 - xk);
        double addk = 0.5 * (1 + xk);
        double cost, sint;
        if (xk < 1) {
            for (int a0 = 0; a0 < nsph; a0++) {
                cost = polc[a0] * mulk + addk;
                sint = sqrt(1 - cost * cost);
                k[0] = radc[i0] * sint;
                k[1] = radc[i0] * cost;
                ind = i0 * nsph + a0;
                kfactors0[ind] = exp(-kappa * (0.5 * q * q + k[1] * q));
                if (kfactors1[ind] > 1) {
                    printf("ERROR\n");
                    exit(-1);
                }
            }
        }
        xk = get_xp(radc[i0], q);
        mulk = 0.5 * (1 + xk);
        addk = 0.5 * (xk - 1);
        if (xk > -1) {
            for (int a0 = 0; a0 < nsph; a0++) {
                cost = polc[a0] * mulk + addk;
                sint = sqrt(1 - cost * cost);
                k[0] = radc[i0] * sint;
                k[1] = radc[i0] * cost;
                ind = i0 * nsph + a0;
                kfactors1[ind] = exp(-kappa * (0.5 * q * q - k[1] * q));
                if (kfactors1[ind] > 1) {
                    printf("ERROR\n");
                    exit(-1);
                }
            }
        }
    }
}

void get_kappahf_contribs(double *kfactors0, double *kfactors1,
                          double *radc, double *polc, double q,
                          const double kappa, int nrad, int nsph,
                          double *ex0, double *ex1, double *ex2) {
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        double k[2];
        int ind;
        double xk = get_xk(radc[i0], q);
        double mulk = 0.5 * (1 - xk);
        double addk = 0.5 * (1 + xk);
        double cost, sint;
        if (xk < 1) {
            for (int a0 = 0; a0 < nsph; a0++) {
                cost = polc[a0] * mulk + addk;
                sint = sqrt(1 - cost * cost);
                k[0] = radc[i0] * sint;
                k[1] = radc[i0] * cost;
                ind = i0 * nsph + a0;
                kfactors0[ind] = exp(-kappa * (0.5 * q * q + k[1] * q - ex1[ind] + ex0[ind]));
            
            }
        }
        xk = get_xp(radc[i0], q);
        mulk = 0.5 * (1 + xk);
        addk = 0.5 * (xk - 1);
        if (xk > -1) {
            for (int a0 = 0; a0 < nsph; a0++) {
                cost = polc[a0] * mulk + addk;
                sint = sqrt(1 - cost * cost);
                k[0] = radc[i0] * sint;
                k[1] = radc[i0] * cost;
                ind = i0 * nsph + a0;
                kfactors1[ind] = exp(-kappa * (0.5 * q * q - k[1] * q - ex2[ind] + ex0[ind]));
            }
        }
    }
}

void calculate_mp2(MP2_ARGS) {
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        double dsubtot = 0;
        double xsubtot = 0;
        double k[2];
        double p[2];
        double d0, d1;
        for (int i1 = 0; i1 < nrad; i1++) {
            for (int a0 = 0; a0 < nsph; a0++) {
            for (int a1 = 0; a1 < nsph; a1++) {
                k[0] = radc[i0] * azic[a0];
                k[1] = radc[i0] * polc[a0];
                p[0] = radc[i1] * azic[a1];
                p[1] = radc[i1] * polc[a1];
                double n0 = k[0] * k[0] + (k[1] + q) * (k[1] + q);
                double n1 = p[0] * p[0] + (p[1] - q) * (p[1] - q);
                double wt0 = dv[i0] * angw[a0];
                double wt1 = dv[i1] * angw[a1];
                double denom = q * q + (k[1] - p[1]) * q;
                dsubtot += ((n0 > 1) && (n1 > 1)) ? (wt0 * wt1 / denom) : 0;
                d0 = (k[1] - p[1] + q);
                d0 = d0 * d0 + k[0] * k[0] + p[0] * p[0];
                d1 = 2 * k[0] * p[0];
                denom *= sqrt(d0 * d0 - d1 * d1);
                xsubtot += ((n0 > 1) && (n1 > 1)) ? (wt0 * wt1 / denom) : 0;
            }}
        }
        ADD_SUBTOTAL;
    }
}

void calculate_kappamp2(MP2_ARGS) {
    double *kfactors0 = malloc(nrad * nsph * sizeof(double));
    double *kfactors1 = malloc(nrad * nsph * sizeof(double));
    const double kappa = param * kfermi * kfermi;
    get_kappa_contribs(kfactors0, kfactors1, radc, polc, q, kappa, nrad, nsph);
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        DEFINE_VARIABLES_OUTER_LOOP;
        double factor;
        for (int i1 = 0; i1 < nrad; i1++) {
            xp = get_xp(radc[i1], q);
            if (xp <= -1) {
                continue;
            }
            mulp = 0.5 * (1 + xp);
            addp = 0.5 * (xp - 1);
            for (int a0 = 0; a0 < nsph; a0++) {
            for (int a1 = 0; a1 < nsph; a1++) {
                SET_KVECTORS_INNER_LOOP;
                ediff = q * q + (k[1] - p[1]) * q;
                factor = 1 - kfactors0[i0 * nsph + a0] * kfactors1[i1 * nsph + a1];
                wt0 = dv[i0] * angw[a0] * mulk * factor;
                wt1 = dv[i1] * angw[a1] * mulp * factor;
                dsubtot += wt0 * wt1 / ediff;
                d0 = (k[1] - p[1] + q);
                d0 = d0 * d0 + k[0] * k[0] + p[0] * p[0];
                d1 = 2 * k[0] * p[0];
                ediff *= sqrt(d0 * d0 - d1 * d1);
                xsubtot += wt0 * wt1 / ediff;
            }}
        }
        ADD_SUBTOTAL;
    }
    free(kfactors0);
    free(kfactors1);
}

void calculate_kappamp2hf(MP2_ARGS) {
    double *kfactors0 = malloc(nrad * nsph * sizeof(double));
    double *kfactors1 = malloc(nrad * nsph * sizeof(double));
    double *ex0 = malloc(nrad * nsph * sizeof(double));
    double *ex1 = malloc(nrad * nsph * sizeof(double));
    double *ex2 = malloc(nrad * nsph * sizeof(double));
    const double kappa = param * kfermi * kfermi;
    const double invk = 1.0 / kfermi;
    get_exch_eigval_contribs(ex0, ex1, ex2, radc, polc, q, invk, nrad, nsph);
    get_kappahf_contribs(kfactors0, kfactors1, radc, polc, q, kappa,
                         nrad, nsph, ex0, ex1, ex2);
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        DEFINE_VARIABLES_OUTER_LOOP;
        double factor;
        for (int i1 = 0; i1 < nrad; i1++) {
            xp = get_xp(radc[i1], q);
            if (xp <= -1) {
                continue;
            }
            mulp = 0.5 * (1 + xp);
            addp = 0.5 * (xp - 1);
            for (int a0 = 0; a0 < nsph; a0++) {
            for (int a1 = 0; a1 < nsph; a1++) {
                SET_KVECTORS_INNER_LOOP;
                ediff = q * q + (k[1] - p[1]) * q;
                ediff += ex0[i0 * nsph + a0] - ex1[i0 * nsph + a0];
                ediff += ex0[i1 * nsph + a1] - ex2[i1 * nsph + a1];
                factor = 1 - kfactors0[i0 * nsph + a0] * kfactors1[i1 * nsph + a1];
                wt0 = dv[i0] * angw[a0] * mulk * factor;
                wt1 = dv[i1] * angw[a1] * mulp * factor;
                dsubtot += wt0 * wt1 / ediff;
                d0 = (k[1] - p[1] + q);
                d0 = d0 * d0 + k[0] * k[0] + p[0] * p[0];
                d1 = 2 * k[0] * p[0];
                ediff *= sqrt(d0 * d0 - d1 * d1);
                xsubtot += wt0 * wt1 / ediff;
            }}
        }
        ADD_SUBTOTAL;
    }
    free(kfactors0);
    free(kfactors1);
    free(ex0);
    free(ex1);
    free(ex2);
}

void calculate_lambdamp22(MP2_ARGS) {
    const double invk = 1.0 / kfermi;
    const double lambda = param * invk * invk;
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        DEFINE_VARIABLES_OUTER_LOOP;
        for (int i1 = 0; i1 < nrad; i1++) {
            xp = get_xp(radc[i1], q);
            if (xp <= -1) {
                continue;
            }
            mulp = 0.5 * (1 + xp);
            addp = 0.5 * (xp - 1);
            for (int a0 = 0; a0 < nsph; a0++) {
            for (int a1 = 0; a1 < nsph; a1++) {
                SET_KVECTORS_INNER_LOOP;
                ediff = q * q + (k[1] - p[1]) * q + lambda;
                ediff = ediff * ediff;
                wt0 = dv[i0] * angw[a0] * mulk * invk;
                wt1 = dv[i1] * angw[a1] * mulp * invk;
                dsubtot += wt0 * wt1 / ediff;
                d0 = (k[1] - p[1] + q);
                d0 = d0 * d0 + k[0] * k[0] + p[0] * p[0];
                d1 = 2 * k[0] * p[0];
                ediff *= sqrt(d0 * d0 - d1 * d1);
                xsubtot += wt0 * wt1 / ediff;
            } }
        }
        ADD_SUBTOTAL;
    }
}

void calculate_lambdamp2l(MP2_ARGS) {
    const double invk = 1.0 / kfermi;
    const double lambda = param * invk * invk;
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        DEFINE_VARIABLES_OUTER_LOOP;
        for (int i1 = 0; i1 < nrad; i1++) {
            xp = get_xp(radc[i1], q);
            if (xp <= -1) {
                continue;
            }
            mulp = 0.5 * (1 + xp);
            addp = 0.5 * (xp - 1);
            for (int a0 = 0; a0 < nsph; a0++) {
            for (int a1 = 0; a1 < nsph; a1++) {
                SET_KVECTORS_INNER_LOOP;
                ediff = q * q + (k[1] - p[1]) * q + lambda;
                wt0 = dv[i0] * angw[a0] * mulk;
                wt1 = dv[i1] * angw[a1] * mulp * log(ediff * kfermi * kfermi);
                dsubtot += wt0 * wt1 / ediff;
                d0 = (k[1] - p[1] + q);
                d0 = d0 * d0 + k[0] * k[0] + p[0] * p[0];
                d1 = 2 * k[0] * p[0];
                ediff *= sqrt(d0 * d0 - d1 * d1);
                xsubtot += wt0 * wt1 / ediff;
            } }
        }
        ADD_SUBTOTAL;
    }
}

void calculate_lambdamp2(MP2_ARGS) {
    const double invk = 1.0 / kfermi;
    const double lambda = param * invk * invk;
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        DEFINE_VARIABLES_OUTER_LOOP;
        for (int i1 = 0; i1 < nrad; i1++) {
            xp = get_xp(radc[i1], q);
            if (xp <= -1) {
                continue;
            }
            mulp = 0.5 * (1 + xp);
            addp = 0.5 * (xp - 1);
            for (int a0 = 0; a0 < nsph; a0++) {
            for (int a1 = 0; a1 < nsph; a1++) {
                SET_KVECTORS_INNER_LOOP;
                ediff = q * q + (k[1] - p[1]) * q + lambda;
                wt0 = dv[i0] * angw[a0] * mulk;
                wt1 = dv[i1] * angw[a1] * mulp;
                dsubtot += wt0 * wt1 / ediff;
                d0 = (k[1] - p[1] + q);
                d0 = d0 * d0 + k[0] * k[0] + p[0] * p[0];
                d1 = 2 * k[0] * p[0];
                ediff *= sqrt(d0 * d0 - d1 * d1);
                xsubtot += wt0 * wt1 / ediff;
            } }
        }
        ADD_SUBTOTAL;
    }
}


void calculate_lambdamp2hf(MP2_ARGS) {
    double *ex0 = malloc(nrad * nsph * sizeof(double));
    double *ex1 = malloc(nrad * nsph * sizeof(double));
    double *ex2 = malloc(nrad * nsph * sizeof(double));
    const double invk = 1.0 / kfermi;
    const double lambda = param * invk * invk;
    get_exch_eigval_contribs(ex0, ex1, ex2, radc, polc, q, invk, nrad, nsph);
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        DEFINE_VARIABLES_OUTER_LOOP;
        for (int i1 = 0; i1 < nrad; i1++) {
            xp = get_xp(radc[i1], q);
            if (xp <= -1) {
                continue;
            }
            mulp = 0.5 * (1 + xp);
            addp = 0.5 * (xp - 1);
            for (int a0 = 0; a0 < nsph; a0++) {
            for (int a1 = 0; a1 < nsph; a1++) {
                SET_KVECTORS_INNER_LOOP;
                ediff = q * q + (k[1] - p[1]) * q + lambda;
                ediff += ex0[i0 * nsph + a0] - ex1[i0 * nsph + a0];
                ediff += ex0[i1 * nsph + a1] - ex2[i1 * nsph + a1];
                wt0 = dv[i0] * angw[a0] * mulk;
                wt1 = dv[i1] * angw[a1] * mulp;
                dsubtot += wt0 * wt1 / ediff;
                d0 = (k[1] - p[1] + q);
                d0 = d0 * d0 + k[0] * k[0] + p[0] * p[0];
                d1 = 2 * k[0] * p[0];
                ediff *= sqrt(d0 * d0 - d1 * d1);
                xsubtot += wt0 * wt1 / ediff;
            }}
        }
        ADD_SUBTOTAL;
    }
    free(ex0);
    free(ex1);
    free(ex2);
}


void calculate_lambdaxmp2(MP2_ARGS) {
    double *ex0 = malloc(nrad * nsph * sizeof(double));
    double *ex1 = malloc(nrad * nsph * sizeof(double));
    double *ex2 = malloc(nrad * nsph * sizeof(double));
    const double invk = 1.0 / kfermi;
    const double lambda = param;
    // sce is this const * kfermi, but everything is divided by kfermi^2
    double sce = 0.46683486370998806 * invk;
    get_exch_eigval_contribs(ex0, ex1, ex2, radc, polc, q, invk, nrad, nsph);
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        DEFINE_VARIABLES_OUTER_LOOP;
        double gap0, gap1;
        double exx0, exx1;
        for (int i1 = 0; i1 < nrad; i1++) {
            xp = get_xp(radc[i1], q);
            if (xp <= -1) {
                continue;
            }
            mulp = 0.5 * (1 + xp);
            addp = 0.5 * (xp - 1);
            for (int a0 = 0; a0 < nsph; a0++) {
            for (int a1 = 0; a1 < nsph; a1++) {
                SET_KVECTORS_INNER_LOOP;
                exx0 = ex0[i0 * nsph + a0];
                exx1 = ex0[i1 * nsph + a1];
                // need to multiply by 0.5 bc we want energy per particle,
                // not the potential for the gap contrib
                gap0 = (sce + 0.5 * exx0);
                gap0 = sqrt(gap0 * gap0 + 0.0001 * exx0 * exx0);
                gap1 = (sce + 0.5 * exx1);
                gap1 = sqrt(gap1 * gap1 + 0.0001 * exx1 * exx1);
                ediff = q * q + (k[1] - p[1]) * q + lambda * (gap0 + gap1);
                wt0 = dv[i0] * angw[a0] * mulk;
                wt1 = dv[i1] * angw[a1] * mulp;
                dsubtot += wt0 * wt1 / ediff;
                d0 = (k[1] - p[1] + q);
                d0 = d0 * d0 + k[0] * k[0] + p[0] * p[0];
                d1 = 2 * k[0] * p[0];
                ediff *= sqrt(d0 * d0 - d1 * d1);
                xsubtot += wt0 * wt1 / ediff;
            }}
        }
        ADD_SUBTOTAL;
    }
    free(ex0);
    free(ex1);
    free(ex2);
}


void calculate_lambdaxmp2hf(MP2_ARGS) {
    double *ex0 = malloc(nrad * nsph * sizeof(double));
    double *ex1 = malloc(nrad * nsph * sizeof(double));
    double *ex2 = malloc(nrad * nsph * sizeof(double));
    const double invk = 1.0 / kfermi;
    const double lambda = param;
    // sce is this const * kfermi, but everything is divided by kfermi^2
    double sce = 0.46683486370998806 * invk;
    get_exch_eigval_contribs(ex0, ex1, ex2, radc, polc, q, invk, nrad, nsph);
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        DEFINE_VARIABLES_OUTER_LOOP;
        double gap0, gap1;
        double exx0, exx1;
        for (int i1 = 0; i1 < nrad; i1++) {
            xp = get_xp(radc[i1], q);
            if (xp <= -1) {
                continue;
            }
            mulp = 0.5 * (1 + xp);
            addp = 0.5 * (xp - 1);
            for (int a0 = 0; a0 < nsph; a0++) {
            for (int a1 = 0; a1 < nsph; a1++) {
                SET_KVECTORS_INNER_LOOP;
                exx0 = ex0[i0 * nsph + a0];
                exx1 = ex0[i1 * nsph + a1];
                // need to multiply by 0.5 bc we want energy per particle,
                // not the potential for the gap contrib
                gap0 = (sce + 0.5 * exx0);
                gap0 = sqrt(gap0 * gap0 + 0.0001 * exx0 * exx0);
                gap1 = (sce + 0.5 * exx1);
                gap1 = sqrt(gap1 * gap1 + 0.0001 * exx1 * exx1);
                ediff = q * q + (k[1] - p[1]) * q + lambda * (gap0 + gap1);
                ediff += ex0[i0 * nsph + a0] - ex1[i0 * nsph + a0];
                ediff += ex0[i1 * nsph + a1] - ex2[i1 * nsph + a1];
                wt0 = dv[i0] * angw[a0] * mulk;
                wt1 = dv[i1] * angw[a1] * mulp;
                dsubtot += wt0 * wt1 / ediff;
                d0 = (k[1] - p[1] + q);
                d0 = d0 * d0 + k[0] * k[0] + p[0] * p[0];
                d1 = 2 * k[0] * p[0];
                ediff *= sqrt(d0 * d0 - d1 * d1);
                xsubtot += wt0 * wt1 / ediff;
            }}
        }
        ADD_SUBTOTAL;
    }
    free(ex0);
    free(ex1);
    free(ex2);
}


void calculate_taump2hf(MP2_ARGS) {
    double *kfactors0 = malloc(nrad * nsph * sizeof(double));
    double *kfactors1 = malloc(nrad * nsph * sizeof(double));
    double *ex0 = malloc(nrad * nsph * sizeof(double));
    double *ex1 = malloc(nrad * nsph * sizeof(double));
    double *ex2 = malloc(nrad * nsph * sizeof(double));
    const double kappa = param * kfermi * kfermi;
    const double invk = 1.0 / kfermi;
    get_exch_eigval_contribs(ex0, ex1, ex2, radc, polc, q, invk, nrad, nsph);
    get_kappahf_contribs(kfactors0, kfactors1, radc, polc, q, kappa,
                         nrad, nsph, ex0, ex1, ex2);
#pragma omp parallel for
    for (int i0 = 0; i0 < nrad; i0++) {
        DEFINE_VARIABLES_OUTER_LOOP;
        double factor;
        for (int i1 = 0; i1 < nrad; i1++) {
            xp = get_xp(radc[i1], q);
            if (xp <= -1) {
                continue;
            }
            mulp = 0.5 * (1 + xp);
            addp = 0.5 * (xp - 1);
            for (int a0 = 0; a0 < nsph; a0++) {
            for (int a1 = 0; a1 < nsph; a1++) {
                SET_KVECTORS_INNER_LOOP;
                ediff = q * q + (k[1] - p[1]) * q;
                ediff += ex0[i0 * nsph + a0] - ex1[i0 * nsph + a0];
                ediff += ex0[i1 * nsph + a1] - ex2[i1 * nsph + a1];
                factor = 1 - kfactors0[i0 * nsph + a0] * kfactors1[i1 * nsph + a1];
                wt0 = dv[i0] * angw[a0] * mulk * factor;
                wt1 = dv[i1] * angw[a1] * mulp;
                dsubtot += wt0 * wt1 / ediff;
                d0 = (k[1] - p[1] + q);
                d0 = d0 * d0 + k[0] * k[0] + p[0] * p[0];
                d1 = 2 * k[0] * p[0];
                ediff *= sqrt(d0 * d0 - d1 * d1);
                xsubtot += wt0 * wt1 / ediff;
            }}
        }
        ADD_SUBTOTAL;
    }
    free(kfactors0);
    free(kfactors1);
    free(ex0);
    free(ex1);
    free(ex2);
}

