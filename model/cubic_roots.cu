#include "cuda_header.cuh"

static const float x_error;

__host__ __device__ float eval(const float coef[4], float x)
{
    return coef[0] + x * (coef[1] + x * (coef[2] + x * coef[3]));
}

__forceinline__ __host__ __device__ float multi_sign(float a, float b)
{
    return a * ((b < 0.0f) ? -1.0f : 1.0f);
}
__forceinline__ __host__ __device__ int is_different_sign(float y0, float yr)
{
    return y0 * yr < 0.0f;
}
__forceinline__ __host__ __device__ void deflate(float defpoly[3], float coef[4], float root)
{
    defpoly[2] = coef[3];
    for (int i = 2; i > 0; i--) {
        defpoly[i - 1] = coef[i] + root * defpoly[i];
    }
}
__forceinline__ __host__ __device__ float find_closed(const float coef[4], const float deriv[4], float x0, float x1, float y0, float y1)
{

    const float x_error = 1e-10f;
    float xr = (x0 + x1) / 2;
    float yr = eval(coef, xr);
    float xb0 = x0, xb1 = x1;
    if (x1 - x0 <= x_error * 2) return xr;
    while (true) {
        int side = is_different_sign(y0, yr);
        if (side)
            xb1 = xr;
        else
            xb0 = xr;
        float dy = eval(deriv, xr);
        float dx = yr / dy;
        float xn = xr - dx;

        if (xn > xb0 && xn < xb1) {
            float stepsize = CUDA_ABS(xr - xn);
            xr = xn;
            if (stepsize > x_error) {
                yr = eval(coef, xr);
            }
            else
                break;
        }
        else {
            xr = (xb0 + xb1) / 2;
            yr = eval(coef, xr);
            if (xb0 == xr || xb1 == xr || xb1 - xb0 <= 2 * x_error) break;
        }
    }
    return xr;
}
__host__ __device__ int quadratic_roots(float roots[2], float coef[3], float x0, float x1)
{
    const float c = coef[0];
    const float b = coef[1];
    const float a = coef[2];
    const float delta = b * b - 4 * a * c;
    if (delta > 0) {
        int ret = 0;
        const float d = CUDA_SQRT(delta);
        const float q = -(b + multi_sign(d, b)) * 0.5f;
        float rv0 = q / a;
        float rv1 = c / q;
        const float xa = CUDA_MIN(rv0, rv1);
        const float xb = CUDA_MAX(rv0, rv1);

        if (xa >= x0 && xa <= x1) {
            roots[ret++] = xa;
        }
        if (xb >= x0 && xb <= x1) {
            roots[ret++] = xb;
        }
        return ret;
    }
    else if (delta < 0)
        return 0;
    const float r0 = -0.5f * b / a;
    roots[0] = r0;
    return r0 >= x0 && r0 <= x1;
}
__host__ __device__ int cubic_roots(float roots[3], float coef[4], float x0, float x1)
{
    float y0 = eval(coef, x0);
    float y1 = eval(coef, x1);

    // coeffs of derivative
    float a = coef[3] * 3;
    float b_2 = coef[2]; // b / 2
    float c = coef[1];
    float deriv[4] = { c, 2 * b_2, a, 0 };
    float delta_4 = b_2 * b_2 - a * c;
    if (delta_4 > 0.0f) {
        const float d_2 = CUDA_SQRT(delta_4);
        const float q = -(b_2 + multi_sign(d_2, b_2));
        float rv0 = q / a;
        float rv1 = c / q;
        const float xa = CUDA_MIN(rv0, rv1);
        const float xb = CUDA_MAX(rv0, rv1);

        if (is_different_sign(y0, y1)) {
            if (xa >= x1 || xb <= x0 || (xa <= x0 && xb >= x1)) {
                roots[0] = find_closed(coef, deriv, x0, x1, y0, y1);
                return 1;
            }
        }
        else {
            if ((xa >= x1 || xb <= x0) || (xa <= x0 && xb >= x1)) {
                return 0;
            }
        }
        if (xa > x0) {
            const auto ya = eval(coef, xa);
            if (is_different_sign(y0, ya)) {
                roots[0] = find_closed(coef, deriv, x0, xa, y0, ya);
                if (is_different_sign(ya, y1) || (xb < x1 && is_different_sign(ya, eval(coef, xb)))) {
                    float defpoly[3];
                    deflate(defpoly, coef, roots[0]);
                    return quadratic_roots(roots + 1, defpoly, xa, x1) + 1;
                }
                else
                    return 1;
            }

            if (xb < x1) {
                const float yb = eval(coef, xb);
                if (is_different_sign(ya, yb)) {
                    roots[0] = find_closed(coef, deriv, xa, xb, ya, yb);
                    if (is_different_sign(yb, y1)) {
                        float defpoly[3];
                        deflate(defpoly, coef, roots[0]);
                        return quadratic_roots(roots + 1, defpoly, xb, x1) + 1;
                    }
                    else
                        return 1;
                }
                if (is_different_sign(yb, y1)) {
                    roots[0] = find_closed(coef, deriv, xb, x1, yb, y1);
                    return 1;
                } 
            } else {
                if (is_different_sign(ya, y1)) {
                    roots[0] = find_closed(coef, deriv, xa, x1, ya, y1);
                    return 1;
                }
            }
        } else {
            const float yb = eval(coef, xb);
            if (is_different_sign(y0, yb)) {
                roots[0] = find_closed(coef, deriv, x0, xb, y0, yb);
                if (is_different_sign(yb, y1)) {
                    float defpoly[3];
                    deflate(defpoly, coef, roots[0]);
                    return quadratic_roots(roots + 1, defpoly, xb, x1) + 1;
                } 
                else return 1;
            } 
            if (is_different_sign(yb, y1)){
                roots[0] = find_closed(coef, deriv, xb, x1, yb, y1);
                return 1;
            }
        }
    }
    else {
        if (is_different_sign(y0, y1)) {
            roots[0] = find_closed(coef, deriv, x0, x1, y0, y1);
            return 1;
        }
        return 0;
    }
}