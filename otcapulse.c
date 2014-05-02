/*=================================================================
* otcapulse.c
*
* Usage:
* new_a = otcapulse(nsum, rule, K, x, b, a)
*=================================================================*/

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{

    mwSize N;
    mwIndex i, *xi, *bi;
    double *nsum, *rule, *xs, *bs, *a, *new_a, K;

    N = mxGetNumberOfElements(prhs[0]);
    nsum = mxMalloc(N*sizeof(double));
    for(i = 0; i < N; i++) nsum[i] = mxGetPr(prhs[0])[i];
    rule = mxGetPr(prhs[1]);
    K = mxGetPr(prhs[2])[0];
    xi = mxGetIr(prhs[3]);
    xs = mxGetPr(prhs[3]);
    bi = mxGetIr(prhs[4]);
    bs = mxGetPr(prhs[4]);
    a = mxGetPr(prhs[5]);
    
    N = mxGetNzmax(prhs[3]);
    for(i = 0; i < N; i++) nsum[xi[i]] += K*(tanh(xs[i])+1)/2;
    N = mxGetNzmax(prhs[4]);
    for(i = 0; i < N; i++) nsum[bi[i]] += K*(tanh(bs[i])+1)/2;

    N = mxGetNumberOfElements(prhs[0]);
    plhs[0] = mxCreateDoubleMatrix(N, 1, mxREAL);
    new_a = mxGetPr(plhs[0]);
    for(i = 0; i < N; i++)
    {
        if(nsum[i] < 0) nsum[i] = 0;
        else if(nsum[i] > 6*K) nsum[i] = 6*K;
        new_a[i] = rule[(int) (K*nsum[i] + a[i])];
    }
                
    return;
}
