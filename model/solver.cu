#include "cuda_globals.cuh"
#include <cusolver_common.h>
#include <cusolverSp.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

cusolverDnHandle_t dnHandle;
cusolverSpHandle_t cusolverSpH;
cusparseHandle_t cusparseH;
cudaStream_t stream;
cusparseMatDescr_t spdescrA;
csrcholInfo_t sp_chol_info;
// cublasFillMode_t dnUplo;
// cublasHandle_t blasHandle;

void setCublasAndCuSparse()
{
	cusolverDnCreate(&dnHandle);
	// dnUplo = CUBLAS_FILL_MODE_LOWER;
	// cublasCreate(&blasHandle);
	cusolverSpCreate(&cusolverSpH);
	cudaStreamCreate(&stream);
	cusolverSpSetStream(cusolverSpH, stream);
	cusparseCreate(&cusparseH);
	cusparseSetStream(cusparseH, stream);
	cusparseCreateMatDescr(&spdescrA);
	cusolverSpCreateCsrcholInfo(&sp_chol_info);
	cusparseSetMatType(spdescrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(spdescrA, CUSPARSE_INDEX_BASE_ZERO);
}

void freeCublasAndCusparse()
{
	cusolverDnDestroy(dnHandle);
	// cublasDestroy(blasHandle);
	cusolverSpDestroy(cusolverSpH);
	cudaStreamDestroy(stream);
	cusparseDestroy(cusparseH);
	cusolverSpDestroyCsrcholInfo(sp_chol_info);
}

void gpuCholSolver(CsrSparseMatrix& hess, float* x)
{
    // hess must be filled by all nonzero value.
    float tol = 1.e-12f;
    const int reorder = 0; // symrcm
    int singularity = 0;


    auto values = thrust::raw_pointer_cast(hess.values.data());
    auto outer_start = thrust::raw_pointer_cast(hess.outer_start.data());
    auto inner = thrust::raw_pointer_cast(hess.inner.data());
    auto rhs = thrust::raw_pointer_cast(host_cuda_globals.b);
    cusolverStatus_t t = cusolverSpScsrlsvchol(
        cusolverSpH, hess.rows, hess.nnz,
        spdescrA, values, outer_start, inner,
        rhs, tol, reorder, x, &singularity);
    cudaDeviceSynchronize();
    if (0 <= singularity)
    {
        printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
    }

    //checkNumericalPrecisionHost(m_activeDims, x);
}


