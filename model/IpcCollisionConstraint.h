#pragma once
#ifndef IPC_COLLISION_CONSTRAINT_H
#define IPC_COLLISION_CONSTRAINT_H
// #include <Bow/Types.h>
// #include <Bow/Simulation/FEM/IpcEnergy.h>
#include "affine_body.h"
#include "barrier.h"
#include <map>

#include <ipc/distance/distance_type.hpp>
#include <ipc/distance/point_point.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_triangle.hpp>

namespace AIPC {
static const int dim = 3;
using T = double;
using _StorageIndex = int;
using StorageIndex = int;
template <typename DerivedV>
using Field = std::vector<DerivedV>;
using namespace Eigen;
void mollifier_info(const int ei0, const int ei1, const int ej0, const int ej1, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, T& eps, T& mollifier);

void mollifier_gradient(const int ei0, const int ei1, const int ej0, const int ej1, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, T& eps, Vector<T, dim * 4>& gm);

void mollifier_hessian(const int ei0, const int ei1, const int ej0, const int ej1, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, T& eps, Matrix<T, dim * 4, dim * 4>& hm);

bool is_mollifier(const int ei0, const int ei1, const int ej0, const int ej1, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, T& eps);

void mollifier_info(const int ei0, const int ei1, const int ej0, const int ej1, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, T& m, Vector<T, dim * 4>& gm, Matrix<T, dim * 4, dim * 4>& hm);

inline T barrier(T dist2, T dHat2, T kappa)
{
    return barrier::barrier_function(dist2);
}
inline T _barrier_gradient(T dist2, T dHat2, T kappa)
{
    return barrier::barrier_derivative_d(dist2);
}

inline T _barrier_hessian(T dist2, T dHat2, T kappa)
{
    return barrier::barrier_second_derivative(dist2);
}
class IpcConstraintOp3D {
public:
    using TV = Vector<T, dim>;

    IpcConstraintOp3D(const int index, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat_, const T& kappa_, const T& dt_, const T w, const T es_ = 1.0)
        : objPairId(index), dHat(dHat_), kappa(kappa_), dt(dt_), area(w), energy_scale(es_)
    {
        c_nodes.reserve(4);
        dHat2 = dHat * dHat;
    }

    int objPairId;
    int obj_Id[2];

    Field<int> c_nodes;
    T energy_scale = 1.0;

    const T &dHat, kappa, dt;
    T dHat2, dist2, area;

    T scale;
    T barrier_dist2, barrier_gradient;

    virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area) = 0;

    virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb) = 0;

    virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true) = 0;
};

class IpcPPConstraint : public IpcConstraintOp3D {
public:
    IpcPPConstraint(const int index, const int p0, const int p1, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& dt, const T area, const T es = 1.0)
        : IpcConstraintOp3D(index, surface_x, surface_X, dHat, kappa, dt, area, es)
    {
        c_nodes = { p0, p1 };

        for (int n = 0; n < 2; n++)
            for (int j = 0; j < 4; j++) {
                cId[4 * n + j] = dpdx[c_nodes[n]].first.data()[j];
                w[4 * n + j] = dpdx[c_nodes[n]].second.data()[j];
            }
        obj_Id[0] = cId[0] / 4;
        obj_Id[1] = cId[4] / 4;

        dist2 = ::ipc::point_point_distance(surface_x[c_nodes[0]], surface_x[c_nodes[1]]);
        scale = this->energy_scale * area * dHat;
        barrier_dist2 = barrier(dist2, dHat2, kappa);
        barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);
    }

    virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

    virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

    virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true);

    int cId[8];
    T w[8];

    Vector<T, 6> PP_grad;
};

class IpcPEConstraint : public IpcConstraintOp3D {
public:
    IpcPEConstraint(const int index, const int p, const int ei0, const int ei1, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& dt, const T area, const T es = 1.0)
        : IpcConstraintOp3D(index, surface_x, surface_X, dHat, kappa, dt, area, es)
    {
        c_nodes = { p, ei0, ei1 };
        for (int n = 0; n < 3; n++)
            for (int j = 0; j < 4; j++) {
                cId[4 * n + j] = dpdx[c_nodes[n]].first.data()[j];
                w[4 * n + j] = dpdx[c_nodes[n]].second.data()[j];
            }
        obj_Id[0] = cId[0] / 4;
        obj_Id[1] = cId[4] / 4;

        dist2 = ::ipc::point_edge_distance(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]]);

        scale = this->energy_scale * area * dHat;
        barrier_dist2 = barrier(dist2, dHat2, kappa);
        barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);
    }

    virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

    virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

    virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true);

    int cId[12];
    T w[12];

    Vector<T, 9> PE_grad;
};

class IpcPTConstraint : public IpcConstraintOp3D {
public:
    IpcPTConstraint(const int index, const int p, const int t0, const int t1, const int t2, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& dt_, const T area, const T es = 1.0)
        : IpcConstraintOp3D(index, surface_x, surface_X, dHat, kappa, dt, area, es)
    {
        c_nodes = { p, t0, t1, t2 };
        for (int n = 0; n < 4; n++)
            for (int j = 0; j < 4; j++) {
                cId[4 * n + j] = dpdx[c_nodes[n]].first.data()[j];
                w[4 * n + j] = dpdx[c_nodes[n]].second.data()[j];
            }
        obj_Id[0] = cId[0] / 4;
        obj_Id[1] = cId[4] / 4;

        dist2 = ::ipc::point_triangle_distance(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]]);
        scale = this->energy_scale * area * dHat;
        barrier_dist2 = barrier(dist2, dHat2, kappa);
        barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);
    }

    virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

    virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

    virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true);

    int cId[16];
    T w[16];

    Vector<T, 12> PT_grad;
};

class IpcEEConstraint : public IpcConstraintOp3D {
public:
    IpcEEConstraint(const int index, const int ei0, const int ei1, const int ej0, const int ej1, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& dt, const T area, const T es = 1.0)
        : IpcConstraintOp3D(index, surface_x, surface_X, dHat, kappa, dt, area, es)
    {
        c_nodes = { ei0, ei1, ej0, ej1 };
        for (int n = 0; n < 4; n++)
            for (int j = 0; j < 4; j++) {
                cId[4 * n + j] = dpdx[c_nodes[n]].first.data()[j];
                w[4 * n + j] = dpdx[c_nodes[n]].second.data()[j];
            }
        obj_Id[0] = cId[0] / 4;
        obj_Id[1] = cId[8] / 4;

        dist2 = ::ipc::edge_edge_distance(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]]);
        scale = this->energy_scale * area * dHat;
        barrier_dist2 = barrier(dist2, dHat2, kappa);
        barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);
    }

    virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

    virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

    virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true);

    int cId[16];
    T w[16];
    Vector<T, 12> EE_grad;
};

class IpcPPMConstraint : public IpcConstraintOp3D
	{
	public:
		IpcPPMConstraint(const int index, const int ei0, const int ei1, const int ej0, const int ej1, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& dt, const T area, const T m, const T eps, const T es = 1.0) :
			IpcConstraintOp3D(index, surface_x, surface_X, dHat, kappa, dt, area, es)
		{
			c_nodes = { ei0, ei1, ej0, ej1 };
			for (int n = 0; n < 4; n++)
				for (int j = 0; j < 4; j++)
				{
					cId[4 * n + j] = dpdx[c_nodes[n]].first.data()[j];
					w[4 * n + j] = dpdx[c_nodes[n]].second.data()[j];
				}
			obj_Id[0] = cId[0] / 4;
			obj_Id[1] = cId[8] / 4;

			dist2 = ::ipc::point_point_distance(surface_x[c_nodes[0]], surface_x[c_nodes[2]]);
			mollifier = m;
			eps_x = eps;

			scale = this->energy_scale * area * dHat;
			barrier_dist2 = barrier(dist2, dHat2, kappa);
			barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);
		}

		virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

		virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

		virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true);

		int cId[16];
		T w[16];
		T mollifier, eps_x;
		Vector<T, 6> PP_grad;
		Vector<T, 12> mollifier_grad;
	};

	class IpcPEMConstraint : public IpcConstraintOp3D
	{
	public:
		IpcPEMConstraint(const int index, const int ei0, const int ei1, const int ej0, const int ej1, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& dt, const T area, const T m, const T eps, const T es = 1.0) :
			IpcConstraintOp3D(index, surface_x, surface_X, dHat, kappa, dt, area, es)
		{
			c_nodes = { ei0, ei1, ej0, ej1 };
			for (int n = 0; n < 4; n++)
				for (int j = 0; j < 4; j++)
				{
					cId[4 * n + j] = dpdx[c_nodes[n]].first.data()[j];
					w[4 * n + j] = dpdx[c_nodes[n]].second.data()[j];
				}
			obj_Id[0] = cId[0] / 4;
			obj_Id[1] = cId[8] / 4;

			dist2 = ::ipc::point_edge_distance(surface_x[c_nodes[0]], surface_x[c_nodes[2]], surface_x[c_nodes[3]]);
			mollifier = m;
			eps_x = eps;

			scale = this->energy_scale * area * dHat;
			barrier_dist2 = barrier(dist2, dHat2, kappa);
			barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);
		}

		virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

		virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

		virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true);

		int cId[16];
		T w[16];
		T mollifier, eps_x;
		Vector<T, 9> PE_grad;
		Vector<T, 12> mollifier_grad;
	};

	class IpcEEMConstraint : public IpcConstraintOp3D
	{
	public:
		IpcEEMConstraint(const int index, const int ei0, const int ei1, const int ej0, const int ej1, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& dt, const T area, const T m, const T eps, const T es = 1.0) :
			IpcConstraintOp3D(index, surface_x, surface_X, dHat, kappa, dt, area, es)
		{
			c_nodes = { ei0, ei1, ej0, ej1 };
			for (int n = 0; n < 4; n++)
				for (int j = 0; j < 4; j++)
				{
					cId[4 * n + j] = dpdx[c_nodes[n]].first.data()[j];
					w[4 * n + j] = dpdx[c_nodes[n]].second.data()[j];
				}
			obj_Id[0] = cId[0] / 4;
			obj_Id[1] = cId[8] / 4;

			dist2 = ::ipc::edge_edge_distance(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]]);

			mollifier = m;
			eps_x = eps;
			
			scale = this->energy_scale * area * dHat;
			barrier_dist2 = barrier(dist2, dHat2, kappa);
			barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);
		}

		virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

		virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

		virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true);

		int cId[16];
		T w[16];
		T mollifier, eps_x;
		Vector<T, 12> EE_grad;
		Vector<T, 12> mollifier_grad;
	};
}
#endif