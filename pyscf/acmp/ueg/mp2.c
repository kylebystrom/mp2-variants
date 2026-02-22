#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))
#define PI              4 * atan(1.0)

/*void get_v_inds(int *g_oovx, int *inds_oov, int n_oov, int *vir_gvecs, int nvir)
{
    int maxx = 0, maxy = 0, maxz = 0;
#pragma omp parallel
{
    int _maxx = 0;
    int _maxy = 0;
    int _maxz = 0;
#pragma omp for
    for (int oov = 0; oov < n_oov; oov++) {
        _maxx = MAX(_maxx, abs(g_oovx[3 * oov + 0]));
        _maxy = MAX(_maxy, abs(g_oovx[3 * oov + 1]));
        _maxz = MAX(_maxz, abs(g_oovx[3 * oov + 2]));
    }
#pragma omp critical
{
    maxx = MAX(maxx, _maxx);
    maxy = MAX(maxx, _maxy);
    maxz = MAX(maxx, _maxz);
}
}
    int refsize = (maxx + 1) * (maxy + 1) * (maxz + 1);
    int *ref_xyz = malloc(refsize * sizeof(int));
#pragma omp parallel for
    for (int i = 0; i < refsize; i++) {
        ref_xyz[i] = -1; // nvir;
    }
    for (int v = 0; v < nvir; v++) {
        int indx = abs(vir_gvecs[3 * v + 0]);
        int indy = abs(vir_gvecs[3 * v + 1]);
        int indz = abs(vir_gvecs[3 * v + 2]);
        ref_xyz[indx * (maxy + 1) * (maxz + 1) + indy * (maxz + 1) + indz] = v;
    }
#pragma omp parallel for
    for (int oov = 0; oov < n_oov; oov++) {
        int indx = abs(g_oovx[3 * oov + 0]);
        int indy = abs(g_oovx[3 * oov + 1]);
        int indz = abs(g_oovx[3 * oov + 2]);
        inds_oov[oov] = ref_xyz[indx * (maxy + 1) * (maxz + 1) + indy * (maxz + 1) + indz];
    }
    free(ref_xyz);
}*/

void get_v_inds(int *inds_oov, int nocc, int *occ_gvecs,
                int nvir, int *vir_gvecs)
{
    int maxx = 0;
    int maxy = 0;
    int maxz = 0;
    for (int i = 0; i < nocc; i++) {
        maxx = MAX(maxx, abs(occ_gvecs[3 * i + 0]));
        maxy = MAX(maxy, abs(occ_gvecs[3 * i + 1]));
        maxz = MAX(maxz, abs(occ_gvecs[3 * i + 2]));
    }
    maxx *= 2;
    maxy *= 2;
    maxz *= 2;
    int tmpx = 0;
    int tmpy = 0;
    int tmpz = 0;
    for (int i = 0; i < nvir; i++) {
        tmpx = MAX(tmpx, abs(vir_gvecs[3 * i + 0]));
        tmpy = MAX(tmpy, abs(vir_gvecs[3 * i + 1]));
        tmpz = MAX(tmpz, abs(vir_gvecs[3 * i + 2]));
    }
    maxx += tmpx;
    maxy += tmpy;
    maxz += tmpz;
    int refsize = (maxx + 1) * (maxy + 1) * (maxz + 1);
    int *ref_xyz = malloc(refsize * sizeof(int));
#pragma omp parallel for
    for (int i = 0; i < refsize; i++) {
        ref_xyz[i] = -1; // nvir;
    }
    for (int v = 0; v < nvir; v++) {
        int indx = abs(vir_gvecs[3 * v + 0]);
        int indy = abs(vir_gvecs[3 * v + 1]);
        int indz = abs(vir_gvecs[3 * v + 2]);
        ref_xyz[indx * (maxy + 1) * (maxz + 1) + indy * (maxz + 1) + indz] = v;
    }
#pragma omp parallel for
    for (int i = 0; i < nocc; i++) {
        int ija = i * nocc * nvir;
        int indx, indy, indz;
        for (int j = 0; j < nocc; j++) {
            for (int a = 0; a < nvir; a++, ija++) {
                indx = abs(occ_gvecs[3 * i + 0] + occ_gvecs[3 * j + 0] - vir_gvecs[3 * a + 0]);
                indy = abs(occ_gvecs[3 * i + 1] + occ_gvecs[3 * j + 1] - vir_gvecs[3 * a + 1]);
                indz = abs(occ_gvecs[3 * i + 2] + occ_gvecs[3 * j + 2] - vir_gvecs[3 * a + 2]);
                inds_oov[ija] = ref_xyz[indx * (maxy + 1) * (maxz + 1)
                                        + indy * (maxz + 1) + indz];
            }
        }
    }
    free(ref_xyz);
}

void get_kin(double *kin, int *gvecs, int nvec, double length) {
    double twopdg2 = 2 * PI / length;
    twopdg2 = 0.5 * twopdg2 * twopdg2;
#pragma omp parallel for
    for (int g = 0; g < nvec; g++) {
        kin[g] = gvecs[3 * g + 0] * gvecs[3 * g + 0];
        kin[g] += gvecs[3 * g + 1] * gvecs[3 * g + 1];
        kin[g] += gvecs[3 * g + 2] * gvecs[3 * g + 2];
        kin[g] *= twopdg2;
    }
}

void get_coulomb_ov(double *coulomb_ov, int *occ_gvecs, int *vir_gvecs,
                    int nocc, int nvir, double length, double volume)
{
    double gfac = 2 * PI / length;
    double vfac = 4 * PI / volume;
    double R = 0.5 * length;
#pragma omp parallel for
    for (int o = 0; o < nocc; o++) {
        double G;
        int diff;
        for (int v = 0; v < nvir; v++) {
            G = 0;
            diff = vir_gvecs[3 * v + 0] - occ_gvecs[3 * o + 0];
            G += diff * diff;
            diff = vir_gvecs[3 * v + 1] - occ_gvecs[3 * o + 1];
            G += diff * diff;
            diff = vir_gvecs[3 * v + 2] - occ_gvecs[3 * o + 2];
            G += diff * diff;
            G = gfac * gfac * G;
            G = sqrt(G) + 1e-8;
            coulomb_ov[o * nvir + v] = vfac * ((1 - cos(G * R)) / (G * G));
        }
    }
}

void get_coulomb_ov_cut(double *coulomb_ov, int *occ_gvecs, int *vir_gvecs,
                        int nocc, int nvir, double length, double volume)
{
    double gfac = 2 * PI / length;
    double vfac = 4 * PI / volume;
    double R = 0.5 * length;
    double efac = -0.5 * R * R;
#pragma omp parallel for
    for (int o = 0; o < nocc; o++) {
        double G;
        int diff;
        for (int v = 0; v < nvir; v++) {
            G = 0;
            diff = vir_gvecs[3 * v + 0] - occ_gvecs[3 * o + 0];
            G += diff * diff;
            diff = vir_gvecs[3 * v + 1] - occ_gvecs[3 * o + 1];
            G += diff * diff;
            diff = vir_gvecs[3 * v + 2] - occ_gvecs[3 * o + 2];
            G += diff * diff;
            G = gfac * gfac * G;
            coulomb_ov[o * nvir + v] = vfac / G * (1 - exp(efac * G));
            //G = sqrt(G) + 1e-8;
            //coulomb_ov[o * nvir + v] = vfac * ((1 - cos(G * R)) / (G * G));
        }
    }
}

void get_coulomb_ov_nocut(double *coulomb_ov, int *occ_gvecs, int *vir_gvecs,
                          int nocc, int nvir, double length, double volume)
{
    double gfac = 2 * PI / length;
    double vfac = 4 * PI / volume;
#pragma omp parallel for
    for (int o = 0; o < nocc; o++) {
        double G;
        int diff;
        for (int v = 0; v < nvir; v++) {
            G = 0;
            diff = vir_gvecs[3 * v + 0] - occ_gvecs[3 * o + 0];
            G += diff * diff;
            diff = vir_gvecs[3 * v + 1] - occ_gvecs[3 * o + 1];
            G += diff * diff;
            diff = vir_gvecs[3 * v + 2] - occ_gvecs[3 * o + 2];
            G += diff * diff;
            G = gfac * sqrt(G) + 1e-8;
            coulomb_ov[o * nvir + v] = vfac / (G * G);
        }
    }
}

void get_vk_vcut(double *vk, int *occ_gvecs, int *other_gvecs,
                 int nocc, int nother, double length, double volume)
{
    double gfac = 2 * PI / length;
    double vfac = 4 * PI / volume;
    double R = 0.5 * length;
#pragma omp parallel for
    for (int v = 0; v < nother; v++) {
        vk[v] = 0;
        double G;
        int diff;
        for (int o = 0; o < nocc; o++) {
            G = 0;
            diff = other_gvecs[3 * v + 0] - occ_gvecs[3 * o + 0];
            G += diff * diff;
            diff = other_gvecs[3 * v + 1] - occ_gvecs[3 * o + 1];
            G += diff * diff;
            diff = other_gvecs[3 * v + 2] - occ_gvecs[3 * o + 2];
            G += diff * diff;
            G = gfac * sqrt(G);
            G += 1e-8;
            G = (1 - cos(G * R)) / (G * G);
            // G = ((G == 0) ? (0) : ((1 - cos(G * R)) / (G * G)));
            vk[v] += vfac * G;
            // vk[v] += vfac / (G * G) * (1 - cos(G * R));
        }
    }
}

void get_vk_vcut2(double *vk, int *occ_gvecs, int *other_gvecs,
                  int nocc, int nother, double length, double volume)
{
    double gfac = 2 * PI / length;
    double vfac = 4 * PI / volume;
    double R = 0.5 * length;
    double efac = -0.5 * R * R;
#pragma omp parallel for
    for (int v = 0; v < nother; v++) {
        vk[v] = 0;
        double G;
        int diff;
        for (int o = 0; o < nocc; o++) {
            G = 0;
            diff = other_gvecs[3 * v + 0] - occ_gvecs[3 * o + 0];
            G += diff * diff;
            diff = other_gvecs[3 * v + 1] - occ_gvecs[3 * o + 1];
            G += diff * diff;
            diff = other_gvecs[3 * v + 2] - occ_gvecs[3 * o + 2];
            G += diff * diff;
            G = gfac * gfac * G;
            vk[v] += (G == 0) ? (-vfac * efac) : (vfac / G * (1 - exp(efac * G)));
        }
    }
}

void get_vk_novcut(double *vk, int *occ_gvecs, int *other_gvecs,
                   int nocc, int nother, double length, double volume,
                   double madelung)
{
    double gfac = 2 * PI / length;
    double vfac = 4 * PI / volume;
#pragma omp parallel for
    for (int v = 0; v < nother; v++) {
        vk[v] = 0;
        double G;
        int diff;
        for (int o = 0; o < nocc; o++) {
            G = 0;
            diff = other_gvecs[3 * v + 0] - occ_gvecs[3 * o + 0];
            G += diff * diff;
            diff = other_gvecs[3 * v + 1] - occ_gvecs[3 * o + 1];
            G += diff * diff;
            diff = other_gvecs[3 * v + 2] - occ_gvecs[3 * o + 2];
            G += diff * diff;
            G = gfac * sqrt(G);
            vk[v] += (G < 1e-8) ? madelung : (vfac / (G * G));
        }
    }
}

double get_ecorr_kappa_mp2(int *inds_oov, const int nocc, const int nvir,
                           const double kappa, double *coulomb_ov,
                           double *eig_o, double *eig_v)
{
    double edi = 0;
    double exi = 0;
#pragma omp parallel
{
    double _edi = 0;
    double _exi = 0;
#pragma omp for
    for (int i = 0; i < nocc; i++) {
        int *inds_v = inds_oov + i * nocc * nvir;
        double ediff, wt;
        for (int j = 0; j < nocc; j++) {
            for (int a = 0; a < nvir; a++) {
                if (inds_v[a] >= 0) {
                    ediff = eig_v[a] + eig_v[inds_v[a]] - eig_o[i] - eig_o[j];
                    wt = 1 - exp(-kappa * ediff);
                    wt = wt * wt;
                    wt = wt / ediff;
                    wt = coulomb_ov[j * nvir + a] * wt;
                    _edi += coulomb_ov[j * nvir + a] * wt;
                    _exi += coulomb_ov[i * nvir + a] * wt;
                }
            }
            inds_v += nvir;
        }
    }
#pragma omp critical
{
    edi += 2 * _edi;
    exi += _exi;
}
}
    printf("%lf %lf\n", edi, exi);
    return exi - edi;
}

void setup_mp2_maxs(const int nocc, int *occ_gvecs,
                    const int nvir, int *vir_gvecs,
                    const double kappa, double *eig_o, double *eig_v,
                    const double *gaps, double *exp_ov, int *dat)
{
    int n2_lcut = 0;
    int n2_ucut = 0;
    int n2;
    for (int i = 0; i < nocc; i++) {
        n2 = occ_gvecs[3 * i + 0] * occ_gvecs[3 * i + 0];
        n2 += occ_gvecs[3 * i + 1] * occ_gvecs[3 * i + 1];
        n2 += occ_gvecs[3 * i + 2] * occ_gvecs[3 * i + 2];
        n2_lcut = MAX(n2, n2_lcut);
    }
    for (int i = 0; i < nvir; i++) {
        n2 = vir_gvecs[3 * i + 0] * vir_gvecs[3 * i + 0];
        n2 += vir_gvecs[3 * i + 1] * vir_gvecs[3 * i + 1];
        n2 += vir_gvecs[3 * i + 2] * vir_gvecs[3 * i + 2];
        n2_ucut = MAX(n2, n2_ucut);
    }
    int maxx = 0;
    int maxy = 0;
    int maxz = 0;
    for (int i = 0; i < nocc; i++) {
        maxx = MAX(maxx, abs(occ_gvecs[3 * i + 0]));
        maxy = MAX(maxy, abs(occ_gvecs[3 * i + 1]));
        maxz = MAX(maxz, abs(occ_gvecs[3 * i + 2]));
        if (exp_ov != NULL) {
            for (int a = 0; a < nvir; a++) {
                exp_ov[i * nvir + a] = exp(kappa * (eig_o[i] - eig_v[a]));
            }
        }
    }
    maxx *= 2;
    maxy *= 2;
    maxz *= 2;
    int tmpx = 0;
    int tmpy = 0;
    int tmpz = 0;
    for (int i = 0; i < nvir; i++) {
        tmpx = MAX(tmpx, abs(vir_gvecs[3 * i + 0]));
        tmpy = MAX(tmpy, abs(vir_gvecs[3 * i + 1]));
        tmpz = MAX(tmpz, abs(vir_gvecs[3 * i + 2]));
    }
    maxx += tmpx;
    maxy += tmpy;
    maxz += tmpz;
    dat[0] = maxx;
    dat[1] = maxy;
    dat[2] = maxz;
    dat[3] = n2_lcut;
    dat[4] = n2_ucut;
}

void setup_mp2_refs(int maxx, int maxy, int maxz, const int nvir,
                    const int *vir_gvecs, int *ref_xyz)
{
    int refsize = (maxx + 1) * (maxy + 1) * (maxz + 1);
#pragma omp parallel for
    for (int i = 0; i < refsize; i++) {
        ref_xyz[i] = -1; // nvir;
    }
#pragma omp parallel for
    for (int v = 0; v < nvir; v++) {
        int indx = abs(vir_gvecs[3 * v + 0]);
        int indy = abs(vir_gvecs[3 * v + 1]);
        int indz = abs(vir_gvecs[3 * v + 2]);
        ref_xyz[indx * (maxy + 1) * (maxz + 1) + indy * (maxz + 1) + indz] = v;
    }
}

void ecorr_kappa_mp2(const int nocc, int *occ_gvecs,
                     const int nvir, int *vir_gvecs,
                     const double kappa, double *coulomb_ov,
                     double *eig_o, double *eig_v,
                     double *res, const double *gaps)
{
    double edi = 0;
    double exi = 0;
    // if gaps is provided (not null), use lambda model
    int use_lambda_model = (gaps != NULL);
    double *exp_ov = NULL;
    if (!use_lambda_model) {
        exp_ov = malloc(nocc * nvir * sizeof(double));
    }
    int dat[5];
    setup_mp2_maxs(nocc, occ_gvecs, nvir, vir_gvecs, kappa,
                   eig_o, eig_v, gaps, exp_ov, dat);
    const int maxx = dat[0];
    const int maxy = dat[1];
    const int maxz = dat[2];
    const int n2_lcut = dat[3];
    const int n2_ucut = dat[4];
    const int refsize = (maxx + 1) * (maxy + 1) * (maxz + 1);
    int *ref_xyz = malloc(refsize * sizeof(int));
    setup_mp2_refs(maxx, maxy, maxz, nvir, vir_gvecs, ref_xyz);
#pragma omp parallel
{
    double _edi = 0;
    double _exi = 0;
#pragma omp for
    for (int i = 0; i < nocc; i++) {
        double ediff, wt;
        int n2b;
        int indx;
        int indy;
        int indz;
        int b;
        for (int j = 0; j < nocc; j++) {
            if (use_lambda_model) {
                for (int a = 0; a < nvir; a++) {
                    indx = occ_gvecs[3 * i + 0] + occ_gvecs[3 * j + 0] - vir_gvecs[3 * a + 0];
                    indy = occ_gvecs[3 * i + 1] + occ_gvecs[3 * j + 1] - vir_gvecs[3 * a + 1];
                    indz = occ_gvecs[3 * i + 2] + occ_gvecs[3 * j + 2] - vir_gvecs[3 * a + 2];
                    n2b = indx * indx + indy * indy + indz * indz;
                    if (n2b > n2_lcut && n2b <= n2_ucut) {
                        b = ref_xyz[abs(indx) * (maxy + 1) * (maxz + 1)
                                    + abs(indy) * (maxz + 1) + abs(indz)];
                        ediff = eig_v[a] + eig_v[b] - eig_o[i] - eig_o[j] + gaps[i] + gaps[j];
                        wt = 1.0 / ediff;
                        wt = coulomb_ov[j * nvir + a] * wt;
                        _edi += coulomb_ov[j * nvir + a] * wt;
                        _exi += coulomb_ov[i * nvir + a] * wt;
                    }
                }
            } else {
                for (int a = 0; a < nvir; a++) {
                    indx = occ_gvecs[3 * i + 0] + occ_gvecs[3 * j + 0] - vir_gvecs[3 * a + 0];
                    indy = occ_gvecs[3 * i + 1] + occ_gvecs[3 * j + 1] - vir_gvecs[3 * a + 1];
                    indz = occ_gvecs[3 * i + 2] + occ_gvecs[3 * j + 2] - vir_gvecs[3 * a + 2];
                    n2b = indx * indx + indy * indy + indz * indz;
                    if (n2b > n2_lcut && n2b <= n2_ucut) {
                        b = ref_xyz[abs(indx) * (maxy + 1) * (maxz + 1)
                                    + abs(indy) * (maxz + 1) + abs(indz)];
                        ediff = eig_v[a] + eig_v[b] - eig_o[i] - eig_o[j];
                        wt = 1 - exp_ov[i * nvir + a] * exp_ov[j * nvir + b];
                        wt = wt * wt;
                        wt = wt / ediff;
                        wt = coulomb_ov[j * nvir + a] * wt;
                        _edi += coulomb_ov[j * nvir + a] * wt;
                        _exi += coulomb_ov[i * nvir + a] * wt;
                    }
                }
            }
        }
    }
#pragma omp critical
{
    edi += 2 * _edi;
    exi += _exi;
}
}
    free(ref_xyz);
    if (!use_lambda_model) {
        free(exp_ov);
    }
    printf("%lf %lf\n", edi, exi);
    res[0] = -edi;
    res[1] = exi;
    res[2] = exi - edi;
}


void ecorr_mp2_vec(const int nocc, int *occ_gvecs,
                   const int nvir, int *vir_gvecs,
                   double *coulomb_ov, double *eig_o, double *eig_v,
                   double *res)
{
    int dat[5];
    setup_mp2_maxs(nocc, occ_gvecs, nvir, vir_gvecs, 0.0,
                   eig_o, eig_v, NULL, NULL, dat);
    const int maxx = dat[0];
    const int maxy = dat[1];
    const int maxz = dat[2];
    const int n2_lcut = dat[3];
    const int n2_ucut = dat[4];
    const int refsize = (maxx + 1) * (maxy + 1) * (maxz + 1);
    int *ref_xyz = malloc(refsize * sizeof(int));
    setup_mp2_refs(maxx, maxy, maxz, nvir, vir_gvecs, ref_xyz);
#pragma omp parallel
{
    double edi = 0;
    double exi = 0;
#pragma omp for
    for (int i = 0; i < nocc; i++) {
        double ediff, wt;
        int n2b;
        int indx;
        int indy;
        int indz;
        int b;
        edi = 0;
        exi = 0;
        for (int j = 0; j < nocc; j++) {
            for (int a = 0; a < nvir; a++) {
                indx = occ_gvecs[3 * i + 0] + occ_gvecs[3 * j + 0] - vir_gvecs[3 * a + 0];
                indy = occ_gvecs[3 * i + 1] + occ_gvecs[3 * j + 1] - vir_gvecs[3 * a + 1];
                indz = occ_gvecs[3 * i + 2] + occ_gvecs[3 * j + 2] - vir_gvecs[3 * a + 2];
                n2b = indx * indx + indy * indy + indz * indz;
                if (n2b > n2_lcut && n2b <= n2_ucut) {
                    b = ref_xyz[abs(indx) * (maxy + 1) * (maxz + 1)
                                + abs(indy) * (maxz + 1) + abs(indz)];
                    ediff = eig_v[a] + eig_v[b] - eig_o[i] - eig_o[j];
                    wt = 1.0 / ediff;
                    wt = coulomb_ov[i * nvir + a] * wt;
                    edi += coulomb_ov[i * nvir + a] * wt;
                    exi += coulomb_ov[j * nvir + a] * wt;
                }
            }
        }
        // shouldn't need the critical but debugging OMP
#pragma omp critical
{
        res[i] = exi - 2 * edi;
}
    }
}
    free(ref_xyz);
}

static inline double get_ei_ft(double gap, double beta)
{
    int cond = gap < 0;
    int cond2 = fabs(gap) > 1e-10;
    double expei = exp(-beta * gap);
    double ei = cond2 ? (1.0 / gap) : (-0.5 * beta);
    expei = (cond ? (expei * (1 - expei)) : (expei - 1))
            / (1 + expei * expei);
    if (cond2) {
        ei += (2 + beta * beta * gap * gap) * expei / (beta * gap * gap);
    }
    return ei;
}

void ecorr_mp2_vec_ft(const int nocc, int *occ_gvecs, double *f_occ,
                      const int nvir, int *vir_gvecs, double *fm_vir,
                      double *coulomb_ov, double *eig_o, double *eig_v,
                      double beta, double *res)
{
    // NOTE fm_vir is 1 - f_vir
    int dat[5];
    setup_mp2_maxs(nocc, occ_gvecs, nvir, vir_gvecs, 0.0,
                   eig_o, eig_v, NULL, NULL, dat);
    const int maxx = dat[0];
    const int maxy = dat[1];
    const int maxz = dat[2];
    const int n2_lcut = dat[3];
    const int n2_ucut = dat[4];
    const int refsize = (maxx + 1) * (maxy + 1) * (maxz + 1);
    int *ref_xyz = malloc(refsize * sizeof(int));
    setup_mp2_refs(maxx, maxy, maxz, nvir, vir_gvecs, ref_xyz);
#pragma omp parallel
{
    double edi = 0;
    double exi = 0;
#pragma omp for
    for (int i = 0; i < nocc; i++) {
        double ediff, wt;
        int n2b;
        int indx;
        int indy;
        int indz;
        int b;
        edi = 0;
        exi = 0;
        for (int j = 0; j < nocc; j++) {
            for (int a = 0; a < nvir; a++) {
                indx = occ_gvecs[3 * i + 0] + occ_gvecs[3 * j + 0] - vir_gvecs[3 * a + 0];
                indy = occ_gvecs[3 * i + 1] + occ_gvecs[3 * j + 1] - vir_gvecs[3 * a + 1];
                indz = occ_gvecs[3 * i + 2] + occ_gvecs[3 * j + 2] - vir_gvecs[3 * a + 2];
                n2b = indx * indx + indy * indy + indz * indz;
                if (n2b > n2_lcut && n2b <= n2_ucut) {
                    b = ref_xyz[abs(indx) * (maxy + 1) * (maxz + 1)
                                + abs(indy) * (maxz + 1) + abs(indz)];
                    ediff = eig_v[a] + eig_v[b] - eig_o[i] - eig_o[j];
                    wt = get_ei_ft(ediff, beta);
                    wt = coulomb_ov[i * nvir + a] * wt;
                    wt *= f_occ[j] * fm_vir[a] * fm_vir[b];
                    edi += coulomb_ov[i * nvir + a] * wt;
                    exi += coulomb_ov[j * nvir + a] * wt;
                }
            }
        }
        // shouldn't need the critical but debugging OMP
#pragma omp critical
{
        res[i] = exi - 2 * edi;
}
    }
}
    free(ref_xyz);
}
