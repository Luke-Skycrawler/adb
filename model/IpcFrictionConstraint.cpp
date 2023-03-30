#include "math.h"
#include "IpcFrictionConstraint.h"

namespace AIPC
{
	T IpcPPFConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
	{
		TV p0 = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV p1 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];
		TV relDX3D;
		Point_Point_RelDX_3D(p0, p1, relDX3D);
		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;
		return this->energy_scale * f0_SF(relDX.squaredNorm(), epsvh) * mu * normalForce / x_weight;

	}

	void IpcPPFConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
	{
		TV p0 = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV p1 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];

		TV relDX3D;
		Point_Point_RelDX_3D(p0, p1, relDX3D);
		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;
		T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
		relDX *= f1_div_relDXNorm * mu * normalForce;

		Point_Point_RelDXTan_To_Mesh_3D(relDX, tanBasis, friction_grad);
		friction_grad *= this->energy_scale;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				ga.data()[3 * i + j] = w[i] * friction_grad.data()[j];
				gb.data()[3 * i + j] = w[4 + i] * friction_grad.data()[3 + j];
			}
		}

	}

	void IpcPPFConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd)
	{
		Eigen::Matrix<T, 6, 6> PP_hess;
		PP_hess.setZero();

		TV p0 = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV p1 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];

		TV relDX3D;
		Point_Point_RelDX_3D(p0, p1, relDX3D);
		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;

		T relDXSqNorm = relDX.squaredNorm();
		T relDXNorm = std::sqrt(relDXSqNorm);

		T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDXSqNorm, epsvh);
		T f2_term = f2_SF_Term(relDXSqNorm, epsvh);

		Eigen::Matrix<T, 2, 6> TT;
		Point_Point_TT_3D(tanBasis, TT);

		if (relDXSqNorm >= epsvh * epsvh) {
			Vector<T, dim - 1> ubar(-relDX(1), relDX(0));
			PP_hess = (TT.transpose() * ((mu * normalForce * f1_div_relDXNorm / relDXSqNorm) * ubar)) * (ubar.transpose() * TT);
		}
		else {
			if (relDXNorm == 0) {
				if (normalForce > 0)
				{
					PP_hess = ((mu * normalForce * f1_div_relDXNorm) * TT.transpose()) * TT;
				}else PP_hess.setZero();
			}
			else {
				// only need to project the inner 2x2 matrix to SPD
				Eigen::Matrix<T, 2, 2> innerMtr = ((f2_term / relDXNorm) * relDX) * relDX.transpose();
				innerMtr.diagonal().array() += f1_div_relDXNorm;
				innerMtr *= mu * normalForce;
				if (project_pd)
					make_pd(innerMtr);
				// tensor product:
				PP_hess = TT.transpose() * innerMtr * TT;
			}
		}
		PP_hess *= this-> energy_scale * x_weight;

		T aa_wr, bb_wr, ab_wr;
		T aa_wc, bb_wc, ab_wc;
		T aa_val, bb_val, ab_val;

		int rowId, colId;
		for (int r = 0; r < 4; r++)
		{
			aa_wr = w[r];
			bb_wr = w[4 + r];
			for (int c = r; c < 4; c++)
			{
				aa_wc = w[c];
				bb_wc = w[4 + c];
				for (int rr = 0; rr < 3; rr++)
					for (int cc = (r == c ? rr : 0); cc < 3; cc++)
					{
						rowId = 3 * r + rr;
						colId = 3 * c + cc;
						int i = 12 * colId + rowId;
						aa_val = aa_wr * aa_wc * PP_hess.data()[6 * cc + rr];
						bb_val = bb_wr * bb_wc * PP_hess.data()[6 * (3 + cc) + 3 + rr];
						aa_hess.data()[i] = aa_val;
						bb_hess.data()[i] = bb_val;
						if (rowId != colId)
						{
							i = colId + 12 * rowId;
							aa_hess.data()[i] = aa_val;
							bb_hess.data()[i] = bb_val;
						}
					}
			}
		}

		if (obj_Id[0] < obj_Id[1])
		{
			for (int r = 0; r < 4; r++)
			{
				ab_wr = w[r];
				for (int c = 0; c < 4; c++)
				{
					ab_wc = w[4 + c];
					for (int rr = 0; rr < 3; rr++)
						for (int cc = 0; cc < 3; cc++)
						{
							rowId = 3 * r + rr;
							colId = 3 * c + cc;
							int i = 12 * colId + rowId;
							ab_val = ab_wr * ab_wc * PP_hess.data()[6 * (3 + cc) + rr];
							ab_hess.data()[i] = ab_val;
						}
				}
			}
		}
		else
		{
			for (int r = 0; r < 4; r++)
			{
				ab_wr = w[r];
				for (int c = 0; c < 4; c++)
				{
					ab_wc = w[4 + c];
					for (int rr = 0; rr < 3; rr++)
						for (int cc = 0; cc < 3; cc++)
						{
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

	T IpcPEFConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
	{
		TV p = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV e0 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];
		TV e1 = x_weight * surface_x[c_nodes[2]] - surface_xhat[c_nodes[2]];
		TV relDX3D;
		Point_Edge_RelDX(p, e0, e1, yita[0], relDX3D);
		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;
		return this->energy_scale * f0_SF(relDX.squaredNorm(), epsvh) * mu * normalForce / x_weight;
	}

	void IpcPEFConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
	{
		TV p = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV e0 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];
		TV e1 = x_weight * surface_x[c_nodes[2]] - surface_xhat[c_nodes[2]];
		TV relDX3D;
		Point_Edge_RelDX(p, e0, e1, yita[0], relDX3D);

		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;

		T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
		relDX *= f1_div_relDXNorm * mu * normalForce;

		Point_Edge_RelDXTan_To_Mesh_3D(relDX, tanBasis, yita[0], friction_grad);
		friction_grad *= this->energy_scale;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				ga.data()[3 * i + j] = w[i] * friction_grad.data()[j];
				gb.data()[3 * i + j] = (w[4 + i] * friction_grad.data()[3 + j] + w[8 + i] * friction_grad.data()[6 + j]);
			}
		}

	}

	void IpcPEFConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd & aa_hess, MatrixXd & bb_hess, MatrixXd & ab_hess, const bool project_pd)
	{
		Eigen::Matrix<T, 9, 9> PE_hess;
		PE_hess.setZero();
		TV p = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV e0 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];
		TV e1 = x_weight * surface_x[c_nodes[2]] - surface_xhat[c_nodes[2]];
		TV relDX3D;
		Point_Edge_RelDX(p, e0, e1, yita[0], relDX3D);

		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;

		T relDXSqNorm = relDX.squaredNorm();
		T relDXNorm = std::sqrt(relDXSqNorm);

		Eigen::Matrix<T, 2, 9> TT;
		Point_Edge_TT_3D(tanBasis, yita[0], TT);

		T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
		T f2_term = f2_SF_Term(relDXSqNorm, epsvh);

		if (relDXSqNorm >= epsvh * epsvh) {
			Vector<T, dim - 1> ubar(-relDX(1), relDX(0));
			PE_hess = (TT.transpose() * ((mu * normalForce * f1_div_relDXNorm / relDXSqNorm) * ubar)) * (ubar.transpose() * TT);
		}
		else {
			if (relDXNorm == 0) {
				if (normalForce > 0)
				{
					PE_hess = ((mu * normalForce * f1_div_relDXNorm) * TT.transpose()) * TT;
				}else PE_hess.setZero();
			}
			else {
				Eigen::Matrix<T, 2, 2> innerMtr = ((f2_term / relDXNorm) * relDX) * relDX.transpose();
				innerMtr.diagonal().array() += f1_div_relDXNorm;
				innerMtr *= mu * normalForce;
				if (project_pd)
					make_pd(innerMtr);
				PE_hess = TT.transpose() * innerMtr * TT;
			}
		}
		PE_hess *= this->energy_scale * x_weight;

		T aa_wr, bb_wr0, bb_wr1, ab_wr;
		T aa_wc, bb_wc0, bb_wc1, ab_wc0, ab_wc1;
		T aa_val, bb_val, ab_val;
		int rowId, colId;

		for (int r = 0; r < 4; r++)
		{
			aa_wr = w[r];
			bb_wr0 = w[4 + r];
			bb_wr1 = w[8 + r];
			for (int c = r; c < 4; c++)
			{
				aa_wc = w[c];
				bb_wc0 = w[4 + c];
				bb_wc1 = w[8 + c];
				for (int rr = 0; rr < 3; rr++)
					for (int cc = (r == c ? rr : 0); cc < 3; cc++)
					{
						rowId = 3 * r + rr;
						colId = 3 * c + cc;
						aa_val = aa_wr * aa_wc * PE_hess.data()[9 * cc + rr];
						bb_val = bb_wc0 * (bb_wr0 * PE_hess.data()[9 * (3 + cc) + 3 + rr] + bb_wr1 * PE_hess.data()[9 * (3 + cc) + 6 + rr]) +
							bb_wc1 * (bb_wr0 *  PE_hess.data()[9 * (6 + cc) + 3 + rr] + bb_wr1 * PE_hess.data()[9 * (6 + cc) + 6 + rr]);

						int i = 12 * colId + rowId;
						aa_hess.data()[i] = aa_val;
						bb_hess.data()[i] = bb_val;

						if (rowId != colId)
						{
							i = colId + 12 * rowId;
							aa_hess.data()[i] = aa_val;
							bb_hess.data()[i] = bb_val;
						}
					}
			}
		}

		if (obj_Id[0] < obj_Id[1])
		{

			for (int r = 0; r < 4; r++)
			{
				ab_wr = w[r];
				for (int c = 0; c < 4; c++)
				{
					ab_wc0 = w[4 + c];
					ab_wc1 = w[8 + c];
					for (int rr = 0; rr < 3; rr++)
						for (int cc = 0; cc < 3; cc++)
						{
							rowId = 3 * r + rr;
							colId = 3 * c + cc;
							ab_val = ab_wr * (ab_wc0 * PE_hess.data()[9 * (3 + cc) + rr] + ab_wc1 * PE_hess.data()[9 * (6 + cc) + rr]);
							int i = 12 * colId + rowId;
							ab_hess.data()[i] = ab_val;
						}
				}
			}
		}
		else
		{

			for (int r = 0; r < 4; r++)
			{
				ab_wr = w[r];
				for (int c = 0; c < 4; c++)
				{
					ab_wc0 = w[4 + c];
					ab_wc1 = w[8 + c];
					for (int rr = 0; rr < 3; rr++)
						for (int cc = 0; cc < 3; cc++)
						{
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

	T IpcPTFConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
	{
		TV p = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV t0 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];
		TV t1 = x_weight * surface_x[c_nodes[2]] - surface_xhat[c_nodes[2]];
		TV t2 = x_weight * surface_x[c_nodes[3]] - surface_xhat[c_nodes[3]];
		TV relDX3D;
		Point_Triangle_RelDX(p, t0, t1, t2, yita[0], yita[1], relDX3D);
		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;
		return this->energy_scale * f0_SF(relDX.squaredNorm(), epsvh) * mu * normalForce / x_weight;
	}

	void IpcPTFConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
	{
		TV p = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV t0 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];
		TV t1 = x_weight * surface_x[c_nodes[2]] - surface_xhat[c_nodes[2]];
		TV t2 = x_weight * surface_x[c_nodes[3]] - surface_xhat[c_nodes[3]];
		TV relDX3D;
		Point_Triangle_RelDX(p, t0, t1, t2, yita[0], yita[1], relDX3D);

		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;
		T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
		relDX *= f1_div_relDXNorm * mu * normalForce;

		Point_Triangle_RelDXTan_To_Mesh(relDX, tanBasis, yita[0], yita[1], friction_grad);
		friction_grad *= this->energy_scale;
		
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				ga.data()[3 * i + j] = w[i] * friction_grad.data()[j];
				gb.data()[3 * i + j] = (w[4 + i] * friction_grad.data()[dim + j] + w[8 + i] * friction_grad.data()[6 + j] + w[12 + i] * friction_grad.data()[9 + j]);
			}
		}

	}

	void IpcPTFConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd)
	{
		Eigen::Matrix<T, 12, 12> PT_hess;
		PT_hess.setZero();
		TV p = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV t0 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];
		TV t1 = x_weight * surface_x[c_nodes[2]] - surface_xhat[c_nodes[2]];
		TV t2 = x_weight * surface_x[c_nodes[3]] - surface_xhat[c_nodes[3]];
		TV relDX3D;
		Point_Triangle_RelDX(p, t0, t1, t2, yita[0], yita[1], relDX3D);

		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;
		T relDXSqNorm = relDX.squaredNorm();
		T relDXNorm = std::sqrt(relDXSqNorm);

		Eigen::Matrix<T, 2, 12> TT;
		Point_Triangle_TT(tanBasis, yita[0], yita[1], TT);

		T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
		T f2_term = f2_SF_Term(relDXSqNorm, epsvh);

		if (relDXSqNorm >= epsvh * epsvh) {
			Vector<T, dim - 1> ubar(-relDX(1), relDX(0));
			PT_hess = (TT.transpose() * ((mu * normalForce * f1_div_relDXNorm / relDXSqNorm) * ubar)) * (ubar.transpose() * TT);
		}
		else {
			if (relDXNorm == 0) {
				if (normalForce > 0)
				{
					PT_hess = ((mu * normalForce * f1_div_relDXNorm) * TT.transpose()) * TT;
				}else PT_hess.setZero();
			}
			else {
				Eigen::Matrix<T, 2, 2> innerMtr = ((f2_term / relDXNorm) * relDX) * relDX.transpose();
				innerMtr.diagonal().array() += f1_div_relDXNorm;
				innerMtr *= mu * normalForce;
				if (project_pd)
					make_pd(innerMtr);
				PT_hess = TT.transpose() * innerMtr * TT;
			}
		}
		PT_hess *= this->energy_scale * x_weight;

		T aa_wr, bb_wr0, bb_wr1, bb_wr2, ab_wr;
		T aa_wc, bb_wc0, bb_wc1, bb_wc2, ab_wc0, ab_wc1, ab_wc2;
		T aa_val, bb_val, ab_val;
		int rowId, colId;

		for (int r = 0; r < 4; r++)
		{
			aa_wr = w[r];
			bb_wr0 = w[4 + r];
			bb_wr1 = w[8 + r];
			bb_wr2 = w[12 + r];
			for (int c = r; c < 4; c++)
			{
				aa_wc = w[c];
				bb_wc0 = w[4 + c];
				bb_wc1 = w[8 + c];
				bb_wc2 = w[12 + c];
				for (int rr = 0; rr < 3; rr++)
					for (int cc = (r == c ? rr : 0); cc < 3; cc++)
					{
						rowId = 3 * r + rr;
						colId = 3 * c + cc;

						aa_val = aa_wr * aa_wc * PT_hess.data()[12 * cc + rr];
						bb_val = bb_wc0 * (bb_wr0 * PT_hess.data()[12 * (3 + cc) + 3 + rr] + bb_wr1 * PT_hess.data()[12 * (3 + cc) + 6 + rr] + bb_wr2 * PT_hess.data()[12 * (3 + cc) + 9 + rr]) +
							bb_wc1 * (bb_wr0 *PT_hess.data()[12 * (6 + cc) + 3 + rr] + bb_wr1 * PT_hess.data()[12 * (6 + cc) + 6 + rr] + bb_wr2 * PT_hess.data()[12 * (6 + cc) + 9 + rr]) +
							bb_wc2 * (bb_wr0 *PT_hess.data()[12 * (9 + cc) + 3 + rr] + bb_wr1 * PT_hess.data()[12 * (9 + cc) + 6 + rr] + bb_wr2 * PT_hess.data()[12 * (9 + cc) + 9 + rr]);

						int i = 12 * colId + rowId;
						aa_hess.data()[i] = aa_val;
						bb_hess.data()[i] = bb_val;

						if (rowId != colId)
						{
							i = colId + 12 * rowId;
							aa_hess.data()[i] = aa_val;
							bb_hess.data()[i] = bb_val;
						}
					}
			}
		}

		if (obj_Id[0] < obj_Id[1])
		{
			for (int r = 0; r < 4; r++)
			{
				ab_wr = w[r];
				for (int c = 0; c < 4; c++)
				{
					ab_wc0 = w[4 + c];
					ab_wc1 = w[8 + c];
					ab_wc2 = w[12 + c];
					for (int rr = 0; rr < 3; rr++)
						for (int cc = 0; cc < 3; cc++)
						{
							rowId = 3 * r + rr;
							colId = 3 * c + cc;;
							ab_val = ab_wr * (ab_wc0 * PT_hess.data()[12 * (3 + cc) + rr] + ab_wc1 * PT_hess.data()[12 * (6 + cc) + rr] + ab_wc2 * PT_hess.data()[12 * (9 + cc) + rr]);
							int i = 12 * colId + rowId;
							ab_hess.data()[i] = ab_val;
						}
				}
			}
		}
		else
		{
			for (int r = 0; r < 4; r++)
			{
				ab_wr = w[r];
				for (int c = 0; c < 4; c++)
				{
					ab_wc0 = w[4 + c];
					ab_wc1 = w[8 + c];
					ab_wc2 = w[12 + c];
					for (int rr = 0; rr < 3; rr++)
						for (int cc = 0; cc < 3; cc++)
						{
							rowId = 3 * r + rr;
							colId = 3 * c + cc;;
							ab_val = ab_wr * (ab_wc0 * PT_hess.data()[12 * (3 + cc) + rr] + ab_wc1 * PT_hess.data()[12 * (6 + cc) + rr] + ab_wc2 * PT_hess.data()[12 * (9 + cc) + rr]);
							int i = 12 * rowId + colId;
							ab_hess.data()[i] = ab_val;
						}
				}
			}
		}
	}

	T IpcEEFConstraint::energy(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area)
	{
		TV ea0 = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV ea1 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];
		TV eb0 = x_weight * surface_x[c_nodes[2]] - surface_xhat[c_nodes[2]];
		TV eb1 = x_weight * surface_x[c_nodes[3]] - surface_xhat[c_nodes[3]];

		TV relDX3D;
		Edge_Edge_RelDX(ea0, ea1, eb0, eb1, yita[0], yita[1], relDX3D);
		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;
		return this->energy_scale * f0_SF(relDX.squaredNorm(), epsvh) * mu * normalForce / x_weight;
	}

	void IpcEEFConstraint::gradient(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, VectorXd& ga, VectorXd& gb)
	{
		TV ea0 = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV ea1 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];
		TV eb0 = x_weight * surface_x[c_nodes[2]] - surface_xhat[c_nodes[2]];
		TV eb1 = x_weight * surface_x[c_nodes[3]] - surface_xhat[c_nodes[3]];

		TV relDX3D;
		Edge_Edge_RelDX(ea0, ea1, eb0, eb1, yita[0], yita[1], relDX3D);
		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;
		T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
		relDX *= f1_div_relDXNorm * mu * normalForce;

		Edge_Edge_RelDXTan_To_Mesh(relDX, tanBasis, yita[0], yita[1], friction_grad);
		friction_grad *= this->energy_scale;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				ga.data()[3 * i + j] = (w[i] * friction_grad.data()[j] + w[4 + i] * friction_grad.data()[3 + j]);
				gb.data()[3 * i + j] = (w[8 + i] * friction_grad.data()[6 + j] + w[12 + i] * friction_grad.data()[9 + j]);
			}
		}

	}

	void IpcEEFConstraint::hessian(const Field<Vector<T, dim>>& dof_x, const Field<Vector<T, dim>>& surface_x, const Field<Vector<T, dim>>& surface_X, const Field<Vector<T, dim>>& surface_xhat, const std::map<int, T>& snode_area, MatrixXd& aa_hess, MatrixXd& bb_hess, MatrixXd& ab_hess, const bool project_pd)
	{
		Eigen::Matrix<T, 12, 12> EE_hess;
		EE_hess.setZero();
		TV ea0 = x_weight * surface_x[c_nodes[0]] - surface_xhat[c_nodes[0]];
		TV ea1 = x_weight * surface_x[c_nodes[1]] - surface_xhat[c_nodes[1]];
		TV eb0 = x_weight * surface_x[c_nodes[2]] - surface_xhat[c_nodes[2]];
		TV eb1 = x_weight * surface_x[c_nodes[3]] - surface_xhat[c_nodes[3]];
		TV relDX3D;
		Edge_Edge_RelDX(ea0, ea1, eb0, eb1, yita[0], yita[1], relDX3D);

		Eigen::Matrix<T, 2, 1> relDX = tanBasis.transpose() * relDX3D;
		T relDXSqNorm = relDX.squaredNorm();
		T relDXNorm = std::sqrt(relDXSqNorm);

		Eigen::Matrix<T, 2, 12> TT;
		Edge_Edge_TT(tanBasis, yita[0], yita[1], TT);

		T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
		T f2_term = f2_SF_Term(relDXSqNorm, epsvh);

		if (relDXSqNorm >= epsvh * epsvh) {
			Vector<T, dim - 1> ubar(-relDX(1), relDX(0));
			EE_hess = (TT.transpose() * ((mu * normalForce * f1_div_relDXNorm / relDXSqNorm) * ubar)) * (ubar.transpose() * TT);
		}
		else {
			if (relDXNorm == 0) {
				if (normalForce > 0)
				{
					EE_hess = ((mu * normalForce * f1_div_relDXNorm) * TT.transpose()) * TT;
				}
			}
			else {
				Eigen::Matrix<T, 2, 2> innerMtr = ((f2_term / relDXNorm) * relDX) * relDX.transpose();
				innerMtr.diagonal().array() += f1_div_relDXNorm;
				innerMtr *= mu * normalForce;
				if (project_pd)
					make_pd(innerMtr);
				EE_hess = TT.transpose() * innerMtr * TT;
			}
		}
		EE_hess *= this->energy_scale * x_weight;

		T aa_wr0, aa_wr1, bb_wr0, bb_wr1, ab_wr0, ab_wr1;
		T aa_wc0, aa_wc1, bb_wc0, bb_wc1, ab_wc0, ab_wc1;
		T aa_val, bb_val, ab_val;
		int rowId, colId;

		for (int r = 0; r < 4; r++)
		{
			aa_wr0 = w[r];
			aa_wr1 = w[4 + r];
			bb_wr0 = w[8 + r];
			bb_wr1 = w[12 + r];
			for (int c = r; c < 4; c++)
			{
				aa_wc0 = w[c];
				aa_wc1 = w[4 + c];
				bb_wc0 = w[8 + c];
				bb_wc1 = w[12 + c];
				for (int rr = 0; rr < 3; rr++)
					for (int cc = (r == c ? rr : 0); cc < 3; cc++)
					{
						rowId = 3 * r + rr;
						colId = 3 * c + cc;

						aa_val = aa_wc0 * (aa_wr0 * EE_hess.data()[12 * cc + rr] + aa_wr1 * EE_hess.data()[12 * cc + 3 + rr]) +
							aa_wc1 * (aa_wr0 *  EE_hess.data()[12 * (3 + cc) + rr] + aa_wr1 * EE_hess.data()[12 * (3 + cc) + 3 + rr]);
						bb_val = bb_wc0 * (bb_wr0 * EE_hess.data()[12 * (6 + cc) + 6 + rr] + bb_wr1 * EE_hess.data()[12 * (6 + cc) + 9 + rr]) +
							bb_wc1 * (bb_wr0 *  EE_hess.data()[12 * (9 + cc) + 6 + rr] + bb_wr1 * EE_hess.data()[12 * (9 + cc) + 9 + rr]);

						int i = 12 * colId + rowId;
						aa_hess.data()[i] = aa_val;
						bb_hess.data()[i] = bb_val;

						if (rowId != colId)
						{
							i = 12 * rowId + colId;
							aa_hess.data()[i] = aa_val;
							bb_hess.data()[i] = bb_val;
						}
					}

			}
		}

		if (obj_Id[0] < obj_Id[1])
		{
			for (int r = 0; r < 4; r++)
			{
				ab_wr0 = w[r];
				ab_wr1 = w[4 + r];
				for (int c = 0; c < 4; c++)
				{
					ab_wc0 = w[8 + c];
					ab_wc1 = w[12 + c];
					for (int rr = 0; rr < 3; rr++)
						for (int cc = 0; cc < 3; cc++)
						{
							rowId = 3 * r + rr;
							colId = 3 * c + cc;
							ab_val = ab_wc0 * (ab_wr0 * EE_hess.data()[12 * (6 + cc) + rr] + ab_wr1 * EE_hess.data()[12 * (6 + cc) + 3 + rr]) +
								ab_wc1 * (ab_wr0 * EE_hess.data()[12 * (9 + cc) + rr] + ab_wr1 * EE_hess.data()[12 * (9 + cc) + 3 + rr]);

							int i = 12 * colId + rowId;
							ab_hess.data()[i] = ab_val;
						}

				}
			}
		}
		else
		{
			for (int r = 0; r < 4; r++)
			{
				ab_wr0 = w[r];
				ab_wr1 = w[4 + r];
				for (int c = 0; c < 4; c++)
				{
					ab_wc0 = w[8 + c];
					ab_wc1 = w[12 + c];
					for (int rr = 0; rr < 3; rr++)
						for (int cc = 0; cc < 3; cc++)
						{
							rowId = 3 * r + rr;
							colId = 3 * c + cc;
							ab_val = ab_wc0 * (ab_wr0 * EE_hess.data()[12 * (6 + cc) + rr] + ab_wr1 * EE_hess.data()[12 * (6 + cc) + 3 + rr]) +
								ab_wc1 * (ab_wr0 * EE_hess.data()[12 * (9 + cc) + rr] + ab_wr1 * EE_hess.data()[12 * (9 + cc) + 3 + rr]);

							int i = 12 * rowId + colId;
							ab_hess.data()[i] = ab_val;
						}

				}
			}
		}
	}
}