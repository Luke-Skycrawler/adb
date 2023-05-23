#include "../model/affine_body.h"
using namespace Eigen;
// template <
//     typename _Scalar,
//     int _Rows,
//     int _Cols>
// Eigen::Matrix<_Scalar, _Rows, _Cols> project_to_psd(
//     const Eigen::Matrix<_Scalar, _Rows, _Cols>& A)
// {   
//     // https://math.stackexchange.com/q/2776803
//     Eigen::SelfAdjointEigenSolver<
//         Eigen::Matrix<_Scalar, _Rows, _Cols>>
//         eigensolver(A);
//     // Check if all eigen values are zero or positive.
//     // The eigenvalues are sorted in increasing order.
//     if (eigensolver.eigenvalues()[0] >= 0.0) {
//         return A;
//     }
//     Eigen::DiagonalMatrix<double, Eigen::Dynamic> D(eigensolver.eigenvalues());
//     // Save a little time and only project the negative values
//     for (int i = 0; i < A.rows(); i++) {
//         if (D.diagonal()[i] < 0.0) {
//             D.diagonal()[i] = 0.0;
//         }
//         else {
//             break;
//         }
//     }
//     return eigensolver.eigenvectors() * D
//         * eigensolver.eigenvectors().transpose();
// }

Matrix2d project_to_psd(
    const Matrix2d& A)
{   
    // https://math.stackexchange.com/q/2776803
    Eigen::SelfAdjointEigenSolver<
        Matrix2d>
        eigensolver(A);
    // Check if all eigen values are zero or positive.
    // The eigenvalues are sorted in increasing order.
    if (eigensolver.eigenvalues()[0] >= 0.0) {
        return A;
    }
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> D(eigensolver.eigenvalues());
    // Save a little time and only project the negative values
    for (int i = 0; i < A.rows(); i++) {
        if (D.diagonal()[i] < 0.0) {
            D.diagonal()[i] = 0.0;
        }
        else {
            break;
        }
    }
    return eigensolver.eigenvectors() * D
        * eigensolver.eigenvectors().transpose();
}



