#include "math.h"
#include "IpcCollisionConstraint.h"
#include <ipc/distance/edge_edge_mollifier.hpp>
namespace AIPC {
T edge_edge_cross_norm2(const Vector<T, 3>& ea0, const Vector<T, 3>& ea1, const Vector<T, 3>& eb0, const Vector<T, 3>& eb1)
{
    return (ea1 - ea0).cross(eb1 - eb0).squaredNorm();
}

void mollifier_info(const int ei0, const int ei1, const int ej0, const int ej1, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, T& eps, T& mollifier)
{
    const Vector<T, dim>& ea0_rest = surface_X[ei0];
    const Vector<T, dim>& ea1_rest = surface_X[ei1];
    const Vector<T, dim>& eb0_rest = surface_X[ej0];
    const Vector<T, dim>& eb1_rest = surface_X[ej1];
    eps = ::ipc::edge_edge_mollifier_threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest);
    mollifier = ::ipc::edge_edge_mollifier(surface_x[ei0], surface_x[ei1], surface_x[ej0], surface_x[ej1], eps);
}

void mollifier_gradient(const int ei0, const int ei1, const int ej0, const int ej1, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, T& eps, Vector<T, dim * 4>& gm)
{
    const Vector<T, dim>& ea0_rest = surface_X[ei0];
    const Vector<T, dim>& ea1_rest = surface_X[ei1];
    const Vector<T, dim>& eb0_rest = surface_X[ej0];
    const Vector<T, dim>& eb1_rest = surface_X[ej1];
    ::ipc::edge_edge_mollifier_gradient(surface_x[ei0], surface_x[ei1], surface_x[ej0], surface_x[ej1], eps, gm);
}

void mollifier_hessian(const int ei0, const int ei1, const int ej0, const int ej1, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, T& eps, Matrix<T, dim * 4, dim * 4>& hm)
{
    const Vector<T, dim>& ea0_rest = surface_X[ei0];
    const Vector<T, dim>& ea1_rest = surface_X[ei1];
    const Vector<T, dim>& eb0_rest = surface_X[ej0];
    const Vector<T, dim>& eb1_rest = surface_X[ej1];
    ::ipc::edge_edge_mollifier_hessian(surface_x[ei0], surface_x[ei1], surface_x[ej0], surface_x[ej1], eps, hm);
}

bool is_mollifier(const int ei0, const int ei1, const int ej0, const int ej1, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, T& eps)
{
    T EECrossSqNorm = edge_edge_cross_norm2(surface_x[ei0], surface_x[ei1], surface_x[ej0], surface_x[ej1]);
    return EECrossSqNorm < eps;
}

void mollifier_info(const int ei0, const int ei1, const int ej0, const int ej1, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, T& m, Vector<T, dim * 4>& gm, Matrix<T, dim * 4, dim * 4>& hm)
{
    const Vector<T, dim>& ea0_rest = surface_X[ei0];
    const Vector<T, dim>& ea1_rest = surface_X[ei1];
    const Vector<T, dim>& eb0_rest = surface_X[ej0];
    const Vector<T, dim>& eb1_rest = surface_X[ej1];
    T eps_x = ::ipc::edge_edge_mollifier_threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest);
    m = ::ipc::edge_edge_mollifier(surface_x[ei0], surface_x[ei1], surface_x[ej0], surface_x[ej1], eps_x);
    ::ipc::edge_edge_mollifier_gradient(surface_x[ei0], surface_x[ei1], surface_x[ej0], surface_x[ej1], eps_x, gm);
    ::ipc::edge_edge_mollifier_hessian(surface_x[ei0], surface_x[ei1], surface_x[ej0], surface_x[ej1], eps_x, hm);
}

T IpcPPConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
{
    return scale * barrier_dist2;
}

void IpcPPConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
{
    T dE = scale * barrier_gradient;
    ::ipc::point_point_distance_gradient(surface_x[c_nodes[0]], surface_x[c_nodes[1]], PP_grad);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            ga.data()[3 * i + j] = w[i] * dE * PP_grad.data()[j];
            gb.data()[3 * i + j] = w[4 + i] * dE * PP_grad.data()[3 + j];
        }
    }
}

void IpcPPConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd)
{
    Matrix<T, 6, 6> PP_hess;
    ::ipc::point_point_distance_hessian(surface_x[c_nodes[0]], surface_x[c_nodes[1]], PP_hess);

    T dE = scale * barrier_gradient;
    T dE2 = scale * _barrier_hessian(dist2, dHat2, kappa);

    for (int r = 0; r < 6; r++)
        for (int c = r; c < 6; c++) {
            T val = dE * PP_hess.data()[6 * c + r] + dE2 * PP_grad.data()[r] * PP_grad.data()[c];
            PP_hess.data()[6 * c + r] = val;
            PP_hess.data()[6 * r + c] = val;
        }

    if (project_pd)
        make_pd(PP_hess);

    T aa_wr, bb_wr, ab_wr;
    T aa_wc, bb_wc, ab_wc;
    T aa_val, bb_val, ab_val;

    int rowId, colId;
    for (int r = 0; r < 4; r++) {
        aa_wr = w[r];
        bb_wr = w[4 + r];
        for (int c = r; c < 4; c++) {
            aa_wc = w[c];
            bb_wc = w[4 + c];
            for (int rr = 0; rr < 3; rr++)
                for (int cc = (r == c ? rr : 0); cc < 3; cc++) {
                    rowId = 3 * r + rr;
                    colId = 3 * c + cc;
                    int i = 12 * colId + rowId;
                    aa_val = aa_wr * aa_wc * PP_hess.data()[6 * cc + rr];
                    bb_val = bb_wr * bb_wc * PP_hess.data()[6 * (3 + cc) + 3 + rr];
                    aa_hess.data()[i] = aa_val;
                    bb_hess.data()[i] = bb_val;
                    if (rowId != colId) {
                        i = colId + 12 * rowId;
                        aa_hess.data()[i] = aa_val;
                        bb_hess.data()[i] = bb_val;
                    }
                }
        }
    }

    if (obj_Id[0] < obj_Id[1]) {
        for (int r = 0; r < 4; r++) {
            ab_wr = w[r];
            for (int c = 0; c < 4; c++) {
                ab_wc = w[4 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        int i = 12 * colId + rowId;
                        ab_val = ab_wr * ab_wc * PP_hess.data()[6 * (3 + cc) + rr];
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
    else {
        for (int r = 0; r < 4; r++) {
            ab_wr = w[r];
            for (int c = 0; c < 4; c++) {
                ab_wc = w[4 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        int i = 12 * rowId + colId;
                        ab_val = ab_wr * ab_wc * PP_hess.data()[6 * (3 + cc) + rr];
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
}

T IpcPEConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
{
    return scale * barrier_dist2;
}

void IpcPEConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
{
    T dE = scale * barrier_gradient;
    ::ipc::point_edge_distance_gradient(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], PE_grad);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            ga.data()[3 * i + j] = w[i] * dE * PE_grad.data()[j];
            gb.data()[3 * i + j] = (w[4 + i] * PE_grad.data()[3 + j] + w[8 + i] * PE_grad.data()[6 + j]) * dE;
        }
    }
}

void IpcPEConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd)
{
    Eigen::Matrix<T, 9, 9> PE_hess;
    ::ipc::point_edge_distance_hessian(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], PE_hess);

    T dE = scale * barrier_gradient;
    T dE2 = scale * _barrier_hessian(dist2, dHat2, kappa);

    // PE_hess = dE2 * PE_grad * PE_grad.transpose() + dE * PE_hess;

    for (int r = 0; r < 9; r++)
        for (int c = r; c < 9; c++) {
            T val = dE * PE_hess.data()[9 * c + r] + dE2 * PE_grad.data()[r] * PE_grad.data()[c];
            PE_hess.data()[9 * c + r] = val;
            PE_hess.data()[9 * r + c] = val;
        }

    if (project_pd)
        make_pd(PE_hess);

    T aa_wr, bb_wr0, bb_wr1, ab_wr;
    T aa_wc, bb_wc0, bb_wc1, ab_wc0, ab_wc1;
    T aa_val, bb_val, ab_val;
    int rowId, colId;

    for (int r = 0; r < 4; r++) {
        aa_wr = w[r];
        bb_wr0 = w[4 + r];
        bb_wr1 = w[8 + r];
        for (int c = r; c < 4; c++) {
            aa_wc = w[c];
            bb_wc0 = w[4 + c];
            bb_wc1 = w[8 + c];
            for (int rr = 0; rr < 3; rr++)
                for (int cc = (r == c ? rr : 0); cc < 3; cc++) {
                    rowId = 3 * r + rr;
                    colId = 3 * c + cc;
                    aa_val = aa_wr * aa_wc * PE_hess.data()[9 * cc + rr];
                    bb_val = bb_wc0 * (bb_wr0 * PE_hess.data()[9 * (3 + cc) + 3 + rr] + bb_wr1 * PE_hess.data()[9 * (3 + cc) + 6 + rr]) + bb_wc1 * (bb_wr0 * PE_hess.data()[9 * (6 + cc) + 3 + rr] + bb_wr1 * PE_hess.data()[9 * (6 + cc) + 6 + rr]);

                    int i = 12 * colId + rowId;
                    aa_hess.data()[i] = aa_val;
                    bb_hess.data()[i] = bb_val;

                    if (rowId != colId) {
                        i = colId + 12 * rowId;
                        aa_hess.data()[i] = aa_val;
                        bb_hess.data()[i] = bb_val;
                    }
                }
        }
    }

    if (obj_Id[0] < obj_Id[1]) {

        for (int r = 0; r < 4; r++) {
            ab_wr = w[r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[4 + c];
                ab_wc1 = w[8 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ab_val = ab_wr * (ab_wc0 * PE_hess.data()[9 * (3 + cc) + rr] + ab_wc1 * PE_hess.data()[9 * (6 + cc) + rr]);
                        int i = 12 * colId + rowId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
    else {

        for (int r = 0; r < 4; r++) {
            ab_wr = w[r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[4 + c];
                ab_wc1 = w[8 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ab_val = ab_wr * (ab_wc0 * PE_hess.data()[9 * (3 + cc) + rr] + ab_wc1 * PE_hess.data()[9 * (6 + cc) + rr]);
                        int i = 12 * rowId + colId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
}

T IpcPTConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
{
    return scale * barrier_dist2;
}

void IpcPTConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
{
    T dE = scale * barrier_gradient;
    ::ipc::point_triangle_distance_gradient(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], PT_grad);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            ga.data()[3 * i + j] = w[i] * dE * PT_grad.data()[j];
            gb.data()[3 * i + j] = (w[4 + i] * PT_grad.data()[dim + j] + w[8 + i] * PT_grad.data()[6 + j] + w[12 + i] * PT_grad.data()[9 + j]) * dE;
        }
    }
}

void IpcPTConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd)
{
    Eigen::Matrix<T, 12, 12> PT_hess;
    ::ipc::point_triangle_distance_hessian(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], PT_hess);
    // Vector<T, 12> PT_grad;
    //::ipc::point_triangle_distance_gradient(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], PT_grad);
    // T dist2 = ::ipc::point_triangle_distance(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]]);
    // const auto area = snode_area.find(c_nodes[0]);
    // BOW_ASSERT_INFO(area != snode_area.end(), "cannot find surface node area for IPC weighting");

    T dE = scale * barrier_gradient;
    T dE2 = scale * _barrier_hessian(dist2, dHat2, kappa);

    // PT_hess = dE2 * PT_grad * PT_grad.transpose() + dE * PT_hess;
    for (int r = 0; r < 12; r++)
        for (int c = r; c < 12; c++) {
            T val = dE * PT_hess.data()[12 * c + r] + dE2 * PT_grad.data()[r] * PT_grad.data()[c];
            PT_hess.data()[12 * c + r] = val;
            PT_hess.data()[12 * r + c] = val;
        }

    if (project_pd)
        make_pd(PT_hess);

    T aa_wr, bb_wr0, bb_wr1, bb_wr2, ab_wr;
    T aa_wc, bb_wc0, bb_wc1, bb_wc2, ab_wc0, ab_wc1, ab_wc2;
    T aa_val, bb_val, ab_val;
    int rowId, colId;

    for (int r = 0; r < 4; r++) {
        aa_wr = w[r];
        bb_wr0 = w[4 + r];
        bb_wr1 = w[8 + r];
        bb_wr2 = w[12 + r];
        for (int c = r; c < 4; c++) {
            aa_wc = w[c];
            bb_wc0 = w[4 + c];
            bb_wc1 = w[8 + c];
            bb_wc2 = w[12 + c];
            for (int rr = 0; rr < 3; rr++)
                for (int cc = (r == c ? rr : 0); cc < 3; cc++) {
                    rowId = 3 * r + rr;
                    colId = 3 * c + cc;

                    aa_val = aa_wr * aa_wc * PT_hess.data()[12 * cc + rr];
                    bb_val = bb_wc0 * (bb_wr0 * PT_hess.data()[12 * (3 + cc) + 3 + rr] + bb_wr1 * PT_hess.data()[12 * (3 + cc) + 6 + rr] + bb_wr2 * PT_hess.data()[12 * (3 + cc) + 9 + rr]) + bb_wc1 * (bb_wr0 * PT_hess.data()[12 * (6 + cc) + 3 + rr] + bb_wr1 * PT_hess.data()[12 * (6 + cc) + 6 + rr] + bb_wr2 * PT_hess.data()[12 * (6 + cc) + 9 + rr]) + bb_wc2 * (bb_wr0 * PT_hess.data()[12 * (9 + cc) + 3 + rr] + bb_wr1 * PT_hess.data()[12 * (9 + cc) + 6 + rr] + bb_wr2 * PT_hess.data()[12 * (9 + cc) + 9 + rr]);

                    int i = 12 * colId + rowId;
                    aa_hess.data()[i] = aa_val;
                    bb_hess.data()[i] = bb_val;

                    if (rowId != colId) {
                        i = colId + 12 * rowId;
                        aa_hess.data()[i] = aa_val;
                        bb_hess.data()[i] = bb_val;
                    }
                }
        }
    }

    if (obj_Id[0] < obj_Id[1]) {
        for (int r = 0; r < 4; r++) {
            ab_wr = w[r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[4 + c];
                ab_wc1 = w[8 + c];
                ab_wc2 = w[12 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ;
                        ab_val = ab_wr * (ab_wc0 * PT_hess.data()[12 * (3 + cc) + rr] + ab_wc1 * PT_hess.data()[12 * (6 + cc) + rr] + ab_wc2 * PT_hess.data()[12 * (9 + cc) + rr]);
                        int i = 12 * colId + rowId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
    else {
        for (int r = 0; r < 4; r++) {
            ab_wr = w[r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[4 + c];
                ab_wc1 = w[8 + c];
                ab_wc2 = w[12 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ;
                        ab_val = ab_wr * (ab_wc0 * PT_hess.data()[12 * (3 + cc) + rr] + ab_wc1 * PT_hess.data()[12 * (6 + cc) + rr] + ab_wc2 * PT_hess.data()[12 * (9 + cc) + rr]);
                        int i = 12 * rowId + colId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
}

T IpcEEConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
{
    return scale * barrier_dist2;
}

void IpcEEConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
{
    ::ipc::edge_edge_distance_gradient(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], EE_grad);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            ga.data()[3 * i + j] = scale * barrier_gradient * (w[i] * EE_grad.data()[j] + w[4 + i] * EE_grad.data()[3 + j]);
            gb.data()[3 * i + j] = scale * barrier_gradient * (w[8 + i] * EE_grad.data()[6 + j] + w[12 + i] * EE_grad.data()[9 + j]);
        }
    }
}

void IpcEEConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd)
{
    Matrix<T, 12, 12> EE_hess;
    T barrier_hessian = _barrier_hessian(dist2, dHat2, kappa);
    ::ipc::edge_edge_distance_hessian(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], EE_hess);
    EE_hess = scale * (barrier_hessian * EE_grad * EE_grad.transpose() + barrier_gradient * EE_hess);

    if (project_pd)
        make_pd(EE_hess);

    T aa_wr0, aa_wr1, bb_wr0, bb_wr1, ab_wr0, ab_wr1;
    T aa_wc0, aa_wc1, bb_wc0, bb_wc1, ab_wc0, ab_wc1;
    T aa_val, bb_val, ab_val;
    int rowId, colId;

    for (int r = 0; r < 4; r++) {
        aa_wr0 = w[r];
        aa_wr1 = w[4 + r];
        bb_wr0 = w[8 + r];
        bb_wr1 = w[12 + r];
        for (int c = r; c < 4; c++) {
            aa_wc0 = w[c];
            aa_wc1 = w[4 + c];
            bb_wc0 = w[8 + c];
            bb_wc1 = w[12 + c];
            for (int rr = 0; rr < 3; rr++)
                for (int cc = (r == c ? rr : 0); cc < 3; cc++) {
                    rowId = 3 * r + rr;
                    colId = 3 * c + cc;

                    aa_val = aa_wc0 * (aa_wr0 * EE_hess.data()[12 * cc + rr] + aa_wr1 * EE_hess.data()[12 * cc + 3 + rr]) + aa_wc1 * (aa_wr0 * EE_hess.data()[12 * (3 + cc) + rr] + aa_wr1 * EE_hess.data()[12 * (3 + cc) + 3 + rr]);
                    bb_val = bb_wc0 * (bb_wr0 * EE_hess.data()[12 * (6 + cc) + 6 + rr] + bb_wr1 * EE_hess.data()[12 * (6 + cc) + 9 + rr]) + bb_wc1 * (bb_wr0 * EE_hess.data()[12 * (9 + cc) + 6 + rr] + bb_wr1 * EE_hess.data()[12 * (9 + cc) + 9 + rr]);

                    int i = 12 * colId + rowId;
                    aa_hess.data()[i] = aa_val;
                    bb_hess.data()[i] = bb_val;

                    if (rowId != colId) {
                        i = 12 * rowId + colId;
                        aa_hess.data()[i] = aa_val;
                        bb_hess.data()[i] = bb_val;
                    }
                }
        }
    }

    if (obj_Id[0] < obj_Id[1]) {
        for (int r = 0; r < 4; r++) {
            ab_wr0 = w[r];
            ab_wr1 = w[4 + r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[8 + c];
                ab_wc1 = w[12 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ab_val = ab_wc0 * (ab_wr0 * EE_hess.data()[12 * (6 + cc) + rr] + ab_wr1 * EE_hess.data()[12 * (6 + cc) + 3 + rr]) + ab_wc1 * (ab_wr0 * EE_hess.data()[12 * (9 + cc) + rr] + ab_wr1 * EE_hess.data()[12 * (9 + cc) + 3 + rr]);

                        int i = 12 * colId + rowId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
    else {
        for (int r = 0; r < 4; r++) {
            ab_wr0 = w[r];
            ab_wr1 = w[4 + r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[8 + c];
                ab_wc1 = w[12 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ab_val = ab_wc0 * (ab_wr0 * EE_hess.data()[12 * (6 + cc) + rr] + ab_wr1 * EE_hess.data()[12 * (6 + cc) + 3 + rr]) + ab_wc1 * (ab_wr0 * EE_hess.data()[12 * (9 + cc) + rr] + ab_wr1 * EE_hess.data()[12 * (9 + cc) + 3 + rr]);

                        int i = 12 * rowId + colId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
}
double IpcPPMConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
{
    return scale * mollifier * barrier_dist2;
}

void IpcPPMConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
{
    ::ipc::point_point_distance_gradient(surface_x[c_nodes[0]], surface_x[c_nodes[2]], PP_grad);
    mollifier_gradient(c_nodes[0], c_nodes[1], c_nodes[2], c_nodes[3], surface_x, surface_X, eps_x, mollifier_grad);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            ga.data()[3 * i + j] = scale * (w[i] * (barrier_dist2 * mollifier_grad.data()[j] + mollifier * barrier_gradient * PP_grad.data()[j]) + w[4 + i] * (barrier_dist2 * mollifier_grad.data()[3 + j]));
            gb.data()[3 * i + j] = scale * (w[8 + i] * (barrier_dist2 * mollifier_grad.data()[6 + j] + mollifier * barrier_gradient * PP_grad.data()[3 + j]) + w[12 + i] * (barrier_dist2 * mollifier_grad.data()[9 + j]));
        }
    }
}

void IpcPPMConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd)
{
    Matrix<T, 6, 6> barrier_dist2_hess;
    T barrier_hessian = _barrier_hessian(dist2, dHat2, kappa);
    ::ipc::point_point_distance_hessian(surface_x[c_nodes[0]], surface_x[c_nodes[2]], barrier_dist2_hess);
    for (int r = 0; r < 6; r++) {
        for (int c = r; c < 6; c++) {
            T val = barrier_gradient * barrier_dist2_hess.data()[6 * c + r] + barrier_hessian * PP_grad.data()[r] * PP_grad.data()[c];
            barrier_dist2_hess.data()[6 * c + r] = val;
            barrier_dist2_hess.data()[6 * r + c] = val;
        }
    }

    Matrix<T, 12, 12> PPM_hess;
    mollifier_hessian(c_nodes[0], c_nodes[1], c_nodes[2], c_nodes[3], surface_x, surface_X, eps_x, PPM_hess);

    Vector<T, 12> barrier_dist2_grad_extended;
    barrier_dist2_grad_extended.setZero();
    barrier_dist2_grad_extended.template segment<dim>(0) = barrier_gradient * PP_grad.template segment<dim>(0);
    barrier_dist2_grad_extended.template segment<dim>(6) = barrier_gradient * PP_grad.template segment<dim>(3);

    for (int r = 0; r < 12; r++) {
        for (int c = r; c < 12; c++) {
            T val = barrier_dist2 * PPM_hess.data()[12 * c + r] + mollifier_grad.data()[r] * barrier_dist2_grad_extended.data()[c] + barrier_dist2_grad_extended.data()[r] * mollifier_grad.data()[c];
            PPM_hess.data()[12 * c + r] = val;
            PPM_hess.data()[12 * r + c] = val;
        }
    }

    PPM_hess.template block<dim, dim>(0, 0) += mollifier * barrier_dist2_hess.template block<dim, dim>(0, 0);
    PPM_hess.template block<dim, dim>(0, 6) += mollifier * barrier_dist2_hess.template block<dim, dim>(0, 3);
    PPM_hess.template block<dim, dim>(6, 0) += mollifier * barrier_dist2_hess.template block<dim, dim>(3, 0);
    PPM_hess.template block<dim, dim>(6, 6) += mollifier * barrier_dist2_hess.template block<dim, dim>(3, 3);
    PPM_hess = scale * PPM_hess;

    if (project_pd)
        make_pd(PPM_hess);

    T aa_wr0, aa_wr1, bb_wr0, bb_wr1, ab_wr0, ab_wr1;
    T aa_wc0, aa_wc1, bb_wc0, bb_wc1, ab_wc0, ab_wc1;
    T aa_val, bb_val, ab_val;
    int rowId, colId;

    for (int r = 0; r < 4; r++) {
        aa_wr0 = w[r];
        aa_wr1 = w[4 + r];
        bb_wr0 = w[8 + r];
        bb_wr1 = w[12 + r];
        for (int c = r; c < 4; c++) {
            aa_wc0 = w[c];
            aa_wc1 = w[4 + c];
            bb_wc0 = w[8 + c];
            bb_wc1 = w[12 + c];
            for (int rr = 0; rr < 3; rr++)
                for (int cc = (r == c ? rr : 0); cc < 3; cc++) {
                    rowId = 3 * r + rr;
                    colId = 3 * c + cc;

                    aa_val = aa_wc0 * (aa_wr0 * PPM_hess.data()[12 * cc + rr] + aa_wr1 * PPM_hess.data()[12 * cc + 3 + rr]) + aa_wc1 * (aa_wr0 * PPM_hess.data()[12 * (3 + cc) + rr] + aa_wr1 * PPM_hess.data()[12 * (3 + cc) + 3 + rr]);
                    bb_val = bb_wc0 * (bb_wr0 * PPM_hess.data()[12 * (6 + cc) + 6 + rr] + bb_wr1 * PPM_hess.data()[12 * (6 + cc) + 9 + rr]) + bb_wc1 * (bb_wr0 * PPM_hess.data()[12 * (9 + cc) + 6 + rr] + bb_wr1 * PPM_hess.data()[12 * (9 + cc) + 9 + rr]);

                    int i = 12 * colId + rowId;
                    aa_hess.data()[i] = aa_val;
                    bb_hess.data()[i] = bb_val;

                    if (rowId != colId) {
                        i = 12 * rowId + colId;
                        aa_hess.data()[i] = aa_val;
                        bb_hess.data()[i] = bb_val;
                    }
                }
        }
    }

    if (obj_Id[0] < obj_Id[1]) {
        for (int r = 0; r < 4; r++) {
            ab_wr0 = w[r];
            ab_wr1 = w[4 + r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[8 + c];
                ab_wc1 = w[12 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ab_val = ab_wc0 * (ab_wr0 * PPM_hess.data()[12 * (6 + cc) + rr] + ab_wr1 * PPM_hess.data()[12 * (6 + cc) + 3 + rr]) + ab_wc1 * (ab_wr0 * PPM_hess.data()[12 * (9 + cc) + rr] + ab_wr1 * PPM_hess.data()[12 * (9 + cc) + 3 + rr]);

                        int i = 12 * colId + rowId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
    else {
        for (int r = 0; r < 4; r++) {
            ab_wr0 = w[r];
            ab_wr1 = w[4 + r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[8 + c];
                ab_wc1 = w[12 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ab_val = ab_wc0 * (ab_wr0 * PPM_hess.data()[12 * (6 + cc) + rr] + ab_wr1 * PPM_hess.data()[12 * (6 + cc) + 3 + rr]) + ab_wc1 * (ab_wr0 * PPM_hess.data()[12 * (9 + cc) + rr] + ab_wr1 * PPM_hess.data()[12 * (9 + cc) + 3 + rr]);

                        int i = 12 * rowId + colId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
}

T IpcPEMConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
{
    return scale * mollifier * barrier_dist2;
}

void IpcPEMConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
{
    ::ipc::point_edge_distance_gradient(surface_x[c_nodes[0]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], PE_grad);
    mollifier_gradient(c_nodes[0], c_nodes[1], c_nodes[2], c_nodes[3], surface_x, surface_X, eps_x, mollifier_grad);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            ga.data()[3 * i + j] = scale * (w[i] * (barrier_dist2 * mollifier_grad.data()[j] + mollifier * barrier_gradient * PE_grad.data()[j]) + w[4 + i] * (barrier_dist2 * mollifier_grad.data()[3 + j]));
            gb.data()[3 * i + j] = scale * (w[8 + i] * (barrier_dist2 * mollifier_grad.data()[6 + j] + mollifier * barrier_gradient * PE_grad.data()[3 + j]) + w[12 + i] * (barrier_dist2 * mollifier_grad.data()[9 + j] + mollifier * barrier_gradient * PE_grad.data()[6 + j]));
        }
    }
}

void IpcPEMConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd)
{
    Matrix<T, 12, 12> PEM_hess;
    Matrix<T, 9, 9> barrier_dist2_hess;
    T barrier_hessian = _barrier_hessian(dist2, dHat2, kappa);
    ::ipc::point_edge_distance_hessian(surface_x[c_nodes[0]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], barrier_dist2_hess);
    for (int r = 0; r < 9; r++) {
        for (int c = r; c < 9; c++) {
            T val = barrier_gradient * barrier_dist2_hess.data()[9 * c + r] + barrier_hessian * PE_grad.data()[r] * PE_grad.data()[c];
            barrier_dist2_hess.data()[9 * c + r] = val;
            barrier_dist2_hess.data()[9 * r + c] = val;
        }
    }
    mollifier_hessian(c_nodes[0], c_nodes[1], c_nodes[2], c_nodes[3], surface_x, surface_X, eps_x, PEM_hess);

    Vector<T, 12> barrier_dist2_grad_extended;
    barrier_dist2_grad_extended.setZero();
    barrier_dist2_grad_extended.template segment<dim>(0) = barrier_gradient * PE_grad.template segment<dim>(0);
    barrier_dist2_grad_extended.template segment<2 * dim>(6) = barrier_gradient * PE_grad.template segment<2 * dim>(3);

    for (int r = 0; r < 12; r++) {
        for (int c = r; c < 12; c++) {
            T val = barrier_dist2 * PEM_hess.data()[12 * c + r] + mollifier_grad.data()[r] * barrier_dist2_grad_extended.data()[c] + barrier_dist2_grad_extended.data()[r] * mollifier_grad.data()[c];
            PEM_hess.data()[12 * c + r] = val;
            PEM_hess.data()[12 * r + c] = val;
        }
    }

    PEM_hess.template block<dim, dim>(0, 0) += mollifier * barrier_dist2_hess.template block<dim, dim>(0, 0);
    PEM_hess.template block<dim, 2 * dim>(0, 6) += mollifier * barrier_dist2_hess.template block<dim, 2 * dim>(0, 3);
    PEM_hess.template block<2 * dim, dim>(6, 0) += mollifier * barrier_dist2_hess.template block<2 * dim, dim>(3, 0);
    PEM_hess.template block<2 * dim, 2 * dim>(6, 6) += mollifier * barrier_dist2_hess.template block<2 * dim, 2 * dim>(3, 3);
    PEM_hess *= scale;

    if (project_pd)
        make_pd(PEM_hess);

    T aa_wr0, aa_wr1, bb_wr0, bb_wr1, ab_wr0, ab_wr1;
    T aa_wc0, aa_wc1, bb_wc0, bb_wc1, ab_wc0, ab_wc1;
    T aa_val, bb_val, ab_val;
    int rowId, colId;

    for (int r = 0; r < 4; r++) {
        aa_wr0 = w[r];
        aa_wr1 = w[4 + r];
        bb_wr0 = w[8 + r];
        bb_wr1 = w[12 + r];
        for (int c = r; c < 4; c++) {
            aa_wc0 = w[c];
            aa_wc1 = w[4 + c];
            bb_wc0 = w[8 + c];
            bb_wc1 = w[12 + c];
            for (int rr = 0; rr < 3; rr++)
                for (int cc = (r == c ? rr : 0); cc < 3; cc++) {
                    rowId = 3 * r + rr;
                    colId = 3 * c + cc;

                    aa_val = aa_wc0 * (aa_wr0 * PEM_hess.data()[12 * cc + rr] + aa_wr1 * PEM_hess.data()[12 * cc + 3 + rr]) + aa_wc1 * (aa_wr0 * PEM_hess.data()[12 * (3 + cc) + rr] + aa_wr1 * PEM_hess.data()[12 * (3 + cc) + 3 + rr]);
                    bb_val = bb_wc0 * (bb_wr0 * PEM_hess.data()[12 * (6 + cc) + 6 + rr] + bb_wr1 * PEM_hess.data()[12 * (6 + cc) + 9 + rr]) + bb_wc1 * (bb_wr0 * PEM_hess.data()[12 * (9 + cc) + 6 + rr] + bb_wr1 * PEM_hess.data()[12 * (9 + cc) + 9 + rr]);

                    int i = 12 * colId + rowId;
                    aa_hess.data()[i] = aa_val;
                    bb_hess.data()[i] = bb_val;

                    if (rowId != colId) {
                        i = 12 * rowId + colId;
                        aa_hess.data()[i] = aa_val;
                        bb_hess.data()[i] = bb_val;
                    }
                }
        }
    }

    if (obj_Id[0] < obj_Id[1]) {
        for (int r = 0; r < 4; r++) {
            ab_wr0 = w[r];
            ab_wr1 = w[4 + r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[8 + c];
                ab_wc1 = w[12 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ab_val = ab_wc0 * (ab_wr0 * PEM_hess.data()[12 * (6 + cc) + rr] + ab_wr1 * PEM_hess.data()[12 * (6 + cc) + 3 + rr]) + ab_wc1 * (ab_wr0 * PEM_hess.data()[12 * (9 + cc) + rr] + ab_wr1 * PEM_hess.data()[12 * (9 + cc) + 3 + rr]);

                        int i = 12 * colId + rowId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
    else {
        for (int r = 0; r < 4; r++) {
            ab_wr0 = w[r];
            ab_wr1 = w[4 + r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[8 + c];
                ab_wc1 = w[12 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ab_val = ab_wc0 * (ab_wr0 * PEM_hess.data()[12 * (6 + cc) + rr] + ab_wr1 * PEM_hess.data()[12 * (6 + cc) + 3 + rr]) + ab_wc1 * (ab_wr0 * PEM_hess.data()[12 * (9 + cc) + rr] + ab_wr1 * PEM_hess.data()[12 * (9 + cc) + 3 + rr]);

                        int i = 12 * rowId + colId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
}

T IpcEEMConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
{
    return scale * mollifier * barrier_dist2;
}

void IpcEEMConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
{
    ::ipc::edge_edge_distance_gradient(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], EE_grad);
    mollifier_gradient(c_nodes[0], c_nodes[1], c_nodes[2], c_nodes[3], surface_x, surface_X, eps_x, mollifier_grad);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            ga.data()[3 * i + j] = scale * (w[i] * (barrier_dist2 * mollifier_grad.data()[j] + mollifier * barrier_gradient * EE_grad.data()[j]) + w[4 + i] * (barrier_dist2 * mollifier_grad.data()[3 + j] + mollifier * barrier_gradient * EE_grad.data()[3 + j]));
            gb.data()[3 * i + j] = scale * (w[8 + i] * (barrier_dist2 * mollifier_grad.data()[6 + j] + mollifier * barrier_gradient * EE_grad.data()[6 + j]) + w[12 + i] * (barrier_dist2 * mollifier_grad.data()[9 + j] + mollifier * barrier_gradient * EE_grad.data()[9 + j]));
        }
    }
}

void IpcEEMConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd)
{
    Matrix<T, 12, 12> barrier_dist2_hess;
    T barrier_hessian = _barrier_hessian(dist2, dHat2, kappa);
    ::ipc::edge_edge_distance_hessian(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], barrier_dist2_hess);
    for (int r = 0; r < 12; r++) {
        for (int c = r; c < 12; c++) {
            T val = barrier_gradient * barrier_dist2_hess.data()[12 * c + r] + barrier_hessian * EE_grad.data()[r] * EE_grad.data()[c];
            barrier_dist2_hess.data()[12 * c + r] = val;
            barrier_dist2_hess.data()[12 * r + c] = val;
        }
    }

    Matrix<T, 12, 12> EEM_hess;
    mollifier_hessian(c_nodes[0], c_nodes[1], c_nodes[2], c_nodes[3], surface_x, surface_X, eps_x, EEM_hess);
    for (int r = 0; r < 12; r++) {
        for (int c = r; c < 12; c++) {
            T val = barrier_dist2 * EEM_hess.data()[12 * c + r] + barrier_gradient * (mollifier_grad.data()[r] * EE_grad.data()[c] + EE_grad.data()[r] * mollifier_grad.data()[c]) + mollifier * barrier_dist2_hess.data()[12 * c + r];
            EEM_hess.data()[12 * c + r] = val;
            EEM_hess.data()[12 * r + c] = val;
        }
    }
    EEM_hess *= scale;

    if (project_pd)
        make_pd(EEM_hess);

    T aa_wr0, aa_wr1, bb_wr0, bb_wr1, ab_wr0, ab_wr1;
    T aa_wc0, aa_wc1, bb_wc0, bb_wc1, ab_wc0, ab_wc1;
    T aa_val, bb_val, ab_val;
    int rowId, colId;

    for (int r = 0; r < 4; r++) {
        aa_wr0 = w[r];
        aa_wr1 = w[4 + r];
        bb_wr0 = w[8 + r];
        bb_wr1 = w[12 + r];
        for (int c = r; c < 4; c++) {
            aa_wc0 = w[c];
            aa_wc1 = w[4 + c];
            bb_wc0 = w[8 + c];
            bb_wc1 = w[12 + c];
            for (int rr = 0; rr < 3; rr++)
                for (int cc = (r == c ? rr : 0); cc < 3; cc++) {
                    rowId = 3 * r + rr;
                    colId = 3 * c + cc;

                    aa_val = aa_wc0 * (aa_wr0 * EEM_hess.data()[12 * cc + rr] + aa_wr1 * EEM_hess.data()[12 * cc + 3 + rr]) + aa_wc1 * (aa_wr0 * EEM_hess.data()[12 * (3 + cc) + rr] + aa_wr1 * EEM_hess.data()[12 * (3 + cc) + 3 + rr]);
                    bb_val = bb_wc0 * (bb_wr0 * EEM_hess.data()[12 * (6 + cc) + 6 + rr] + bb_wr1 * EEM_hess.data()[12 * (6 + cc) + 9 + rr]) + bb_wc1 * (bb_wr0 * EEM_hess.data()[12 * (9 + cc) + 6 + rr] + bb_wr1 * EEM_hess.data()[12 * (9 + cc) + 9 + rr]);

                    int i = 12 * colId + rowId;
                    aa_hess.data()[i] = aa_val;
                    bb_hess.data()[i] = bb_val;

                    if (rowId != colId) {
                        i = 12 * rowId + colId;
                        aa_hess.data()[i] = aa_val;
                        bb_hess.data()[i] = bb_val;
                    }
                }
        }
    }

    if (obj_Id[0] < obj_Id[1]) {
        for (int r = 0; r < 4; r++) {
            ab_wr0 = w[r];
            ab_wr1 = w[4 + r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[8 + c];
                ab_wc1 = w[12 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ab_val = ab_wc0 * (ab_wr0 * EEM_hess.data()[12 * (6 + cc) + rr] + ab_wr1 * EEM_hess.data()[12 * (6 + cc) + 3 + rr]) + ab_wc1 * (ab_wr0 * EEM_hess.data()[12 * (9 + cc) + rr] + ab_wr1 * EEM_hess.data()[12 * (9 + cc) + 3 + rr]);

                        int i = 12 * colId + rowId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
    else {
        for (int r = 0; r < 4; r++) {
            ab_wr0 = w[r];
            ab_wr1 = w[4 + r];
            for (int c = 0; c < 4; c++) {
                ab_wc0 = w[8 + c];
                ab_wc1 = w[12 + c];
                for (int rr = 0; rr < 3; rr++)
                    for (int cc = 0; cc < 3; cc++) {
                        rowId = 3 * r + rr;
                        colId = 3 * c + cc;
                        ab_val = ab_wc0 * (ab_wr0 * EEM_hess.data()[12 * (6 + cc) + rr] + ab_wr1 * EEM_hess.data()[12 * (6 + cc) + 3 + rr]) + ab_wc1 * (ab_wr0 * EEM_hess.data()[12 * (9 + cc) + rr] + ab_wr1 * EEM_hess.data()[12 * (9 + cc) + 3 + rr]);

                        int i = 12 * rowId + colId;
                        ab_hess.data()[i] = ab_val;
                    }
            }
        }
    }
}
}