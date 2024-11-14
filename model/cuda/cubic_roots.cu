#define CUDA_SOURCE
#include "common.cuh"

using namespace std;
namespace cuda {
    using namespace cuda;

    func scalar eval(const scalar coef[4], scalar x)
    {
        return coef[0] + x * (coef[1] + x * (coef[2] + x * coef[3]));
    }
    
    __forceinline__ func scalar multi_sign(scalar a, scalar b)
    {
        return a * ((b < 0.0f) ? -1.0f : 1.0f);
    }
    __forceinline__ func int is_different_sign(scalar y0, scalar yr)
    {
        return y0 * yr < 0.0f;
    }
    __forceinline__ func void deflate(scalar defpoly[3], scalar coef[4], scalar root)
    {
        defpoly[2] = coef[3];
        for(int i = 2; i > 0; i--) {
            defpoly[i - 1] = coef[i] + root * defpoly[i];
        }
    }
    __forceinline__ func scalar find_closed(const scalar coef[4], const scalar deriv[4], scalar x0, scalar x1, scalar y0, scalar y1)
    {
    
        const scalar x_error = 1e-15f;
        scalar xr = (x0 + x1) / 2;
        scalar yr = eval(coef, xr);
        scalar xb0 = x0, xb1 = x1;
        if(x1 - x0 <= x_error * 2) return xr;
        while(true) {
            int side = is_different_sign(y0, yr);
            if(side)
                xb1 = xr;
            else
                xb0 = xr;
            scalar dy = eval(deriv, xr);
            scalar dx = yr / dy;
            scalar xn = xr - dx;
    
            if(xn > xb0 && xn < xb1) {
                scalar stepsize = abs(xr - xn);
                xr = xn;
                if(stepsize > x_error) {
                    yr = eval(coef, xr);
                }
                else
                    break;
            }
            else {
                xr = (xb0 + xb1) / 2;
                yr = eval(coef, xr);
                if(xb0 == xr || xb1 == xr || xb1 - xb0 <= 2 * x_error) break;
            }
        }
        return xr;
    }
    func int quadratic_roots(scalar roots[2], scalar coef[3], scalar x0, scalar x1)
    {
        const scalar c = coef[0];
        const scalar b = coef[1];
        const scalar a = coef[2];
        const scalar delta = b * b - 4 * a * c;
        if(delta > 0) {
            int ret = 0;
            const scalar d = sqrt(delta);
            const scalar q = -(b + multi_sign(d, b)) * 0.5f;
            scalar rv0 = q / a;
            scalar rv1 = c / q;
            const scalar xa = fmin(rv0, rv1);
            const scalar xb = fmax(rv0, rv1);
    
            if(xa >= x0 && xa <= x1) {
                roots[ret++] = xa;
            }
            if(xb >= x0 && xb <= x1) {
                roots[ret++] = xb;
            }
            return ret;
        }
        else if(delta < 0)
            return 0;
        const scalar r0 = -0.5f * b / a;
        roots[0] = r0;
        return r0 >= x0 && r0 <= x1;
    }
    func int cubic_roots(scalar roots[3], scalar coef[4], scalar x0, scalar x1)
    {
        scalar y0 = eval(coef, x0);
        scalar y1 = eval(coef, x1);
    
        // coeffs of derivative
        scalar a = coef[3] * 3;
        scalar b_2 = coef[2]; // b / 2
        scalar c = coef[1];
        scalar deriv[4] = { c, 2 * b_2, a, 0 };
        scalar delta_4 = b_2 * b_2 - a * c;
        if(delta_4 > 0.0f) {
            const scalar d_2 = sqrt(delta_4);
            const scalar q = -(b_2 + multi_sign(d_2, b_2));
            scalar rv0 = q / a;
            scalar rv1 = c / q;
            const scalar xa = fmin(rv0, rv1);
            const scalar xb = fmax(rv0, rv1);
    
            if(is_different_sign(y0, y1)) {
                if(xa >= x1 || xb <= x0 || (xa <= x0 && xb >= x1)) {
                    roots[0] = find_closed(coef, deriv, x0, x1, y0, y1);
                    return 1;
                }
            }
            else {
                if((xa >= x1 || xb <= x0) || (xa <= x0 && xb >= x1)) {
                    return 0;
                }
            }
            if(xa > x0) {
                const auto ya = eval(coef, xa);
                if(is_different_sign(y0, ya)) {
                    roots[0] = find_closed(coef, deriv, x0, xa, y0, ya);
                    if(is_different_sign(ya, y1) || (xb < x1 && is_different_sign(ya, eval(coef, xb)))) {
                        scalar defpoly[3];
                        deflate(defpoly, coef, roots[0]);
                        return quadratic_roots(roots + 1, defpoly, xa, x1) + 1;
                    }
                    else
                        return 1;
                }
    
                if(xb < x1) {
                    const scalar yb = eval(coef, xb);
                    if(is_different_sign(ya, yb)) {
                        roots[0] = find_closed(coef, deriv, xa, xb, ya, yb);
                        if(is_different_sign(yb, y1)) {
                            scalar defpoly[3];
                            deflate(defpoly, coef, roots[0]);
                            return quadratic_roots(roots + 1, defpoly, xb, x1) + 1;
                        }
                        else
                            return 1;
                    }
                    if(is_different_sign(yb, y1)) {
                        roots[0] = find_closed(coef, deriv, xb, x1, yb, y1);
                        return 1;
                    }
                }
                else {
                    if(is_different_sign(ya, y1)) {
                        roots[0] = find_closed(coef, deriv, xa, x1, ya, y1);
                        return 1;
                    }
                }
            }
            else {
                const scalar yb = eval(coef, xb);
                if(is_different_sign(y0, yb)) {
                    roots[0] = find_closed(coef, deriv, x0, xb, y0, yb);
                    if(is_different_sign(yb, y1)) {
                        scalar defpoly[3];
                        deflate(defpoly, coef, roots[0]);
                        return quadratic_roots(roots + 1, defpoly, xb, x1) + 1;
                    }
                    else
                        return 1;
                }
                if(is_different_sign(yb, y1)) {
                    roots[0] = find_closed(coef, deriv, xb, x1, yb, y1);
                    return 1;
                }
            }
        }
        else {
            if(is_different_sign(y0, y1)) {
                roots[0] = find_closed(coef, deriv, x0, x1, y0, y1);
                return 1;
            }
            return 0;
        }
    }
    }