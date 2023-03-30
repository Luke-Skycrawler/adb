#pragma once
#ifndef IPC_FRICTION_CONSTRAINT_H
#define IPC_FRICTION_CONSTRAINT_H
#include "IpcCollisionConstraint.h"
#include "FrictionUtils.h"
#include <ipc/friction/tangent_basis.hpp>

namespace AIPC
{
	class IpcFrictionConstraintOp3D : public IpcConstraintOp3D
	{
	public:
		IpcFrictionConstraintOp3D(const int index, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat_, const T& kappa_, const T& mu_, const T& dt_, const T& epsv_, const T& x_weight_, const T w, const T es_ = 1.0) :IpcConstraintOp3D(index, surface_x, surface_X, dHat_, kappa_, dt_, w, es_),
			mu(mu_),
			epsv(epsv_),
			x_weight(x_weight_)
		{
			epsvh = epsv_ * dt_;
		}

		T mu, epsv, epsvh, x_weight;
		T normalForce;
		Matrix<T, dim, dim - 1> tanBasis;
		Vector<T, 2> yita;
	};

	class IpcPPFConstraint : public IpcFrictionConstraintOp3D
	{
	public:
		IpcPPFConstraint(const int index, const int p0, const int p1, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& mu, const T& dt, const T& epsv, const T& x_weight, const T area, const T es = 1.0) : IpcFrictionConstraintOp3D(index, surface_x, surface_X, dHat, kappa, mu, dt, epsv, x_weight, area, es)
		{
			c_nodes = { p0, p1 };

			for (int n = 0; n < 2; n++)
				for (int j = 0; j < 4; j++)
				{
					cId[4 * n + j] = dpdx[c_nodes[n]].first.data()[j];
					w[4 * n + j] = dpdx[c_nodes[n]].second.data()[j];
				}
			obj_Id[0] = cId[0] / 4;
			obj_Id[1] = cId[4] / 4;

			dist2 = ::ipc::point_point_distance(surface_x[c_nodes[0]], surface_x[c_nodes[1]]);
			barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);

			normalForce = -barrier_gradient * 2 * area * dHat * std::sqrt(dist2);
			tanBasis = ::ipc::point_point_tangent_basis(surface_x[c_nodes[0]], surface_x[c_nodes[1]]);
		}

		virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

		virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

		virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true);

		int cId[8];
		T w[8];

		Vector<T, 6> friction_grad;
	};

	class IpcPEFConstraint : public IpcFrictionConstraintOp3D
	{
	public:
		IpcPEFConstraint(const int index, const int p, const int ei0, const int ei1, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& mu, const T& dt, const T& epsv, const T& x_weight, const T area, const T es = 1.0) : IpcFrictionConstraintOp3D(index, surface_x, surface_X, dHat, kappa, mu, dt, epsv, x_weight, area, es)
		{
			c_nodes = { p, ei0, ei1 };
			for (int n = 0; n < 3; n++)
				for (int j = 0; j < 4; j++)
				{
					cId[4 * n + j] = dpdx[c_nodes[n]].first.data()[j];
					w[4 * n + j] = dpdx[c_nodes[n]].second.data()[j];
				}
			obj_Id[0] = cId[0] / 4;
			obj_Id[1] = cId[4] / 4;


			dist2 = ::ipc::point_edge_distance(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]]);
			barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);

			normalForce = -barrier_gradient * 2 * area * dHat * std::sqrt(dist2);
			tanBasis = ::ipc::point_edge_tangent_basis(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]]);
			compute_PE_yita_3D(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], yita[0]);
		}

		virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

		virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

		virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd & aa_hess, MatrixXd & bb_hess, MatrixXd & ab_hess, const bool project_pd = true);

		int cId[12];
		T w[12];

		Vector<T, 9> friction_grad;
	};

	class IpcPTFConstraint : public IpcFrictionConstraintOp3D
	{
	public:
		IpcPTFConstraint(const int index, const int p, const int t0, const int t1, const int t2, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& mu, const T& dt, const T& epsv, const T& x_weight, const T area, const T es = 1.0) : IpcFrictionConstraintOp3D(index, surface_x, surface_X, dHat, kappa, mu, dt, epsv, x_weight, area, es)
		{
			c_nodes = { p, t0, t1, t2 };
			for (int n = 0; n < 4; n++)
				for (int j = 0; j < 4; j++)
				{
					cId[4 * n + j] = dpdx[c_nodes[n]].first.data()[j];
					w[4 * n + j] = dpdx[c_nodes[n]].second.data()[j];
				}
			obj_Id[0] = cId[0] / 4;
			obj_Id[1] = cId[4] / 4;

			dist2 = ::ipc::point_triangle_distance(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]]);
			barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);

			normalForce = -barrier_gradient * 2 * area * dHat * std::sqrt(dist2);
			tanBasis = ::ipc::point_triangle_tangent_basis(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]]);
			compute_PT_yita_3D(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], yita);
		}

		virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

		virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

		virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true);

		int cId[16];
		T w[16];

		Vector<T, 12> friction_grad;
	};

	class IpcEEFConstraint : public IpcFrictionConstraintOp3D
	{
	public:
		IpcEEFConstraint(const int index, const int ei0, const int ei1, const int ej0, const int ej1, const Field<std::pair<Vector<int, dim + 1>, Vector<T, dim + 1>>>& dpdx, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const T& dHat, const T& kappa, const T& mu, const T& dt, const T& epsv, const T& x_weight, const T area, const T es = 1.0) : IpcFrictionConstraintOp3D(index, surface_x, surface_X, dHat, kappa, mu, dt, epsv, x_weight, area, es)
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
			barrier_gradient = _barrier_gradient(dist2, dHat2, kappa);

			normalForce = -barrier_gradient * 2 * area * dHat * std::sqrt(dist2);
			tanBasis = ::ipc::edge_edge_tangent_basis(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]]);
			compute_EE_yita_3D(surface_x[c_nodes[0]], surface_x[c_nodes[1]], surface_x[c_nodes[2]], surface_x[c_nodes[3]], yita);
		}

		virtual T energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area);

		virtual void gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb);

		virtual void hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd = true);

		int cId[16];
		T w[16];
		Vector<T, 12> friction_grad;
	};

}



#endif
