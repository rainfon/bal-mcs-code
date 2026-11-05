/* Optional chord kernel. Compile from this directory with:
 *   mex -R2018a har_chord_mex.c
 * Then set cfg.useMexSampler=true. Startup compares a full short chain with the
 * MATLAB sampler and stops if any retained draw differs.
 *
 * [left,right] = har_chord_mex(x,d,S,sb,Phi,y,p,q,epsilon,cap,feasTol,lineIterations)
 */
#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <stddef.h>

#pragma STDC FP_CONTRACT OFF

static int is_real_double(const mxArray *a) {
    return mxIsDouble(a) && !mxIsComplex(a) && !mxIsSparse(a);
}

static double *need_real(const mxArray *a, const char *name) {
    if (!is_real_double(a))
        mexErrMsgIdAndTxt("HAR:MexInput", "%s must be a real full double array.", name);
    return mxGetDoubles(a);
}

static void gemv(const double *A, mwSize m, mwSize n, const double *x, double *y) {
    mwSize i, j;
    for (i = 0; i < m; ++i) y[i] = 0.0;
    for (j = 0; j < n; ++j) {
        double xj = x[j];
        const double *col = A + j * m;
        for (i = 0; i < m; ++i) y[i] += col[i] * xj;
    }
}

static double line_loss(double t, const double *lo, const double *up,
                        const double *lod, const double *upd, mwSize n) {
    double value = 0.0;
    mwSize i;
    for (i = 0; i < n; ++i) {
        double low = lo[i] + t * lod[i];
        double high = up[i] + t * upd[i];
        double branch = (low >= high) ? low : high;
        if (branch > 0.0) value += branch;
    }
    return value;
}

static double loss_at(const double *x, const double *Phi, const double *y,
                      mwSize n, mwSize p, int q, double epsilon, double *U) {
    double value = 0.0;
    mwSize i;
    if (n == 0) return 0.0;
    gemv(Phi, n, p, x, U);
    for (i = 0; i < n; ++i) {
        int cls = (int)y[i];
        double lower = 0.0, upper = 0.0, branch;
        if (cls > 1) lower = x[p + (mwSize)(cls - 2)] + epsilon - U[i];
        if (cls < q) upper = U[i] - x[p + (mwSize)(cls - 1)];
        branch = (lower >= upper) ? lower : upper;
        if (branch > 0.0) value += branch;
    }
    return value;
}

static double bisect_line(double outside, const double *lo, const double *up,
                          const double *lod, const double *upd, double cap,
                          int iterations, mwSize n) {
    double inside = 0.0;
    int k;
    for (k = 0; k < iterations; ++k) {
        double middle = 0.5 * (outside + inside);
        if (middle == outside || middle == inside) break;
        if (line_loss(middle, lo, up, lod, upd, n) <= cap) inside = middle;
        else outside = middle;
    }
    return inside;
}

static double bisect_endpoint(double outside, const double *x, const double *d,
                              const double *Phi, const double *y, mwSize n, mwSize p,
                              mwSize nVar, int q, double epsilon, double cap,
                              int iterations, double *U, double *point) {
    double inside = 0.0;
    mwSize j;
    int k;
    for (j = 0; j < nVar; ++j) point[j] = x[j];
    if (loss_at(point, Phi, y, n, p, q, epsilon, U) > cap)
        mexErrMsgIdAndTxt("HAR:LossStart", "Current point exceeds exact loss cap.");
    for (k = 0; k < iterations; ++k) {
        double middle = 0.5 * (outside + inside);
        if (middle == outside || middle == inside) break;
        for (j = 0; j < nVar; ++j) point[j] = x[j] + middle * d[j];
        if (loss_at(point, Phi, y, n, p, q, epsilon, U) <= cap) inside = middle;
        else outside = middle;
    }
    return inside;
}

static double affine_endpoint(double outside, const double *lo, const double *up,
                              const double *lod, const double *upd, double cap,
                              int iterations, mwSize n, double *low, double *high,
                              double *lorient, double *uorient) {
    double orientation = (outside > 0.0) - (outside < 0.0);
    double t = fabs(outside);
    mwSize i;
    int step, maxStep = 2 * (int)n + 2;
    for (i = 0; i < n; ++i) {
        lorient[i] = orientation * lod[i];
        uorient[i] = orientation * upd[i];
    }
    for (step = 0; step < maxStep; ++step) {
        double value = 0.0, slopeL = 0.0, slopeU = 0.0, slope, candidate;
        for (i = 0; i < n; ++i) {
            low[i] = lo[i] + t * lorient[i];
            high[i] = up[i] + t * uorient[i];
            {
                double branch = (low[i] >= high[i]) ? low[i] : high[i];
                if (branch > 0.0) value += branch;
            }
        }
        if (value <= cap) return orientation * t;
        for (i = 0; i < n; ++i) {
            if (low[i] >= high[i] && low[i] > 0.0) slopeL += lorient[i];
            else if (high[i] > low[i] && high[i] > 0.0) slopeU += uorient[i];
        }
        slope = slopeL + slopeU;
        if (!isfinite(slope) || slope <= 0.0) break;
        candidate = t - (value - cap) / slope;
        if (!isfinite(candidate) || candidate < 0.0 || candidate >= t) break;
        t = candidate;
    }
    return bisect_line(outside, lo, up, lod, upd, cap, iterations, n);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    const double *x, *d, *S, *sb, *Phi, *y;
    double *Sx, *Sd, *slack, *Ux, *Ud, *lo, *up, *lod, *upd;
    double *low, *high, *lorient, *uorient, *point;
    double pScalar, qScalar, epsilon, cap, feasTol, lineScalar;
    double left, right, originalLeft, originalRight;
    mwSize nVar, nS, n, p, i, j;
    int q, lineIterations, outside;
    if (nrhs != 12 || nlhs != 2)
        mexErrMsgIdAndTxt("HAR:MexInput",
            "Usage: [left,right] = har_chord_mex(x,d,S,sb,Phi,y,p,q,epsilon,cap,feasTol,lineIterations).");
    x = need_real(prhs[0], "x");
    d = need_real(prhs[1], "d");
    S = need_real(prhs[2], "S");
    sb = need_real(prhs[3], "sb");
    Phi = need_real(prhs[4], "Phi");
    y = need_real(prhs[5], "y");
    if (!is_real_double(prhs[6]) || !is_real_double(prhs[7]) || !is_real_double(prhs[8]) ||
        !is_real_double(prhs[9]) || !is_real_double(prhs[10]) || !is_real_double(prhs[11]))
        mexErrMsgIdAndTxt("HAR:MexInput", "Scalar chord arguments must be real doubles.");
    pScalar = mxGetScalar(prhs[6]);
    qScalar = mxGetScalar(prhs[7]);
    epsilon = mxGetScalar(prhs[8]);
    cap = mxGetScalar(prhs[9]);
    feasTol = mxGetScalar(prhs[10]);
    lineScalar = mxGetScalar(prhs[11]);
    nVar = mxGetNumberOfElements(prhs[0]);
    nS = mxGetM(prhs[2]);
    n = mxGetNumberOfElements(prhs[5]);
    p = (mwSize)pScalar;
    q = (int)qScalar;
    lineIterations = (int)lineScalar;
    if (mxGetNumberOfElements(prhs[1]) != nVar || mxGetN(prhs[2]) != nVar ||
        mxGetNumberOfElements(prhs[3]) != nS || mxGetM(prhs[4]) != n || mxGetN(prhs[4]) != p ||
        pScalar < 0.0 || qScalar < 1.0 || nVar != p + (mwSize)q - 1)
        mexErrMsgIdAndTxt("HAR:MexInput", "Chord array dimensions do not match the region.");
    Sx = mxCalloc(nS ? nS : 1, sizeof(double));
    Sd = mxCalloc(nS ? nS : 1, sizeof(double));
    slack = mxCalloc(nS ? nS : 1, sizeof(double));
    Ux = mxCalloc(n ? n : 1, sizeof(double));
    Ud = mxCalloc(n ? n : 1, sizeof(double));
    lo = mxCalloc(n ? n : 1, sizeof(double));
    up = mxCalloc(n ? n : 1, sizeof(double));
    lod = mxCalloc(n ? n : 1, sizeof(double));
    upd = mxCalloc(n ? n : 1, sizeof(double));
    low = mxCalloc(n ? n : 1, sizeof(double));
    high = mxCalloc(n ? n : 1, sizeof(double));
    lorient = mxCalloc(n ? n : 1, sizeof(double));
    uorient = mxCalloc(n ? n : 1, sizeof(double));
    point = mxCalloc(nVar ? nVar : 1, sizeof(double));
    gemv(S, nS, nVar, x, Sx);
    outside = 0;
    for (i = 0; i < nS; ++i) {
        double raw = sb[i] - Sx[i];
        if (!(raw >= -feasTol)) outside = 1;
        slack[i] = (raw > 0.0) ? raw : 0.0;
    }
    if (outside) {
        mxFree(Sx); mxFree(Sd); mxFree(slack); mxFree(Ux); mxFree(Ud);
        mxFree(lo); mxFree(up); mxFree(lod); mxFree(upd);
        mxFree(low); mxFree(high); mxFree(lorient); mxFree(uorient); mxFree(point);
        mexErrMsgIdAndTxt("HAR:Outside", "Start outside structure.");
    }
    gemv(S, nS, nVar, d, Sd);
    left = -mxGetInf();
    right = mxGetInf();
    for (i = 0; i < nS; ++i) {
        if (Sd[i] > 0.0) {
            double ratio = slack[i] / Sd[i];
            if (ratio < right) right = ratio;
        } else if (Sd[i] < 0.0) {
            double ratio = slack[i] / Sd[i];
            if (ratio > left) left = ratio;
        }
    }
    if (!(isfinite(left) && isfinite(right) && left <= 0.0 && right >= 0.0)) {
        mxFree(Sx); mxFree(Sd); mxFree(slack); mxFree(Ux); mxFree(Ud);
        mxFree(lo); mxFree(up); mxFree(lod); mxFree(upd);
        mxFree(low); mxFree(high); mxFree(lorient); mxFree(uorient); mxFree(point);
        mexErrMsgIdAndTxt("HAR:Chord", "Unbounded/invalid structural chord.");
    }
    if (n > 0) {
        gemv(Phi, n, p, x, Ux);
        gemv(Phi, n, p, d, Ud);
        for (i = 0; i < n; ++i) {
            int cls = (int)y[i];
            if (cls > 1) {
                lo[i] = x[p + (mwSize)(cls - 2)] + epsilon - Ux[i];
                lod[i] = d[p + (mwSize)(cls - 2)] - Ud[i];
            }
            if (cls < q) {
                up[i] = Ux[i] - x[p + (mwSize)(cls - 1)];
                upd[i] = Ud[i] - d[p + (mwSize)(cls - 1)];
            }
        }
        originalLeft = left;
        originalRight = right;
        if (line_loss(left, lo, up, lod, upd, n) > cap)
            left = affine_endpoint(left, lo, up, lod, upd, cap, lineIterations, n,
                                   low, high, lorient, uorient);
        if (line_loss(right, lo, up, lod, upd, n) > cap)
            right = affine_endpoint(right, lo, up, lod, upd, cap, lineIterations, n,
                                    low, high, lorient, uorient);
        for (j = 0; j < nVar; ++j) point[j] = x[j] + left * d[j];
        if (loss_at(point, Phi, y, n, p, q, epsilon, Ux) > cap)
            left = bisect_endpoint(originalLeft, x, d, Phi, y, n, p, nVar, q,
                                   epsilon, cap, lineIterations, Ux, point);
        for (j = 0; j < nVar; ++j) point[j] = x[j] + right * d[j];
        if (loss_at(point, Phi, y, n, p, q, epsilon, Ux) > cap)
            right = bisect_endpoint(originalRight, x, d, Phi, y, n, p, nVar, q,
                                    epsilon, cap, lineIterations, Ux, point);
    }
    plhs[0] = mxCreateDoubleScalar(left);
    plhs[1] = mxCreateDoubleScalar(right);
    mxFree(Sx); mxFree(Sd); mxFree(slack); mxFree(Ux); mxFree(Ud);
    mxFree(lo); mxFree(up); mxFree(lod); mxFree(upd);
    mxFree(low); mxFree(high); mxFree(lorient); mxFree(uorient); mxFree(point);
}
