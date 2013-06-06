// Copyright (C) 2011, 2012 Zeno Gantner
// Copyright (C) 2010 Steffen Rendle, Zeno Gantner
//
// This file is part of MyMediaLite.
//
// MyMediaLite is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MyMediaLite is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with MyMediaLite.  If not, see <http://www.gnu.org/licenses/>.
using System;
using System.Globalization;
using System.Threading.Tasks;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Double;
using MyMediaLite.DataType;

namespace MyMediaLite.ItemRecommendation
{
	/// <summary>Weighted matrix factorization method proposed by Hu et al. and Pan et al.</summary>
	/// <remarks>
	///   <para>
	///     We use the fast learning method proposed by Hu et al. (alternating least squares, ALS),
	///     and we use a global parameter to give observed values higher weights.
	///   </para>
	///   <para>
	///     Literature:
	///     <list type="bullet">
	///       <item><description>
	///         Y. Hu, Y. Koren, C. Volinsky: Collaborative filtering for implicit feedback datasets.
	///         ICDM 2008.
	///         http://research.yahoo.net/files/HuKorenVolinsky-ICDM08.pdf
	///       </description></item>
	///       <item><description>
	///         R. Pan, Y. Zhou, B. Cao, N. N. Liu, R. M. Lukose, M. Scholz, Q. Yang:
	///         One-class collaborative filtering,
	///         ICDM 2008.
	///         http://www.hpl.hp.com/techreports/2008/HPL-2008-48R1.pdf
	///       </description></item>
	///     </list>
	///   </para>
	///   <para>
	///     This recommender does NOT support incremental updates.
	///   </para>
	/// </remarks>
	public class WRMF : MF
	{
		/// <summary>parameter for the weight/confidence that is put on positive observations</summary>
		public double Alpha { get { return alpha; } set { alpha = value; } }
		double alpha = 1;

		/// <summary>Regularization parameter</summary>
		public double Regularization { get { return regularization; } set { regularization = value; } }
		double regularization = 0.015;

		///
		public WRMF()
		{
			NumIter = 15;
			//UpdateUsers = true;
			//UpdateItems = true;
		}

		///
		public override void Iterate()
		{
			// perform alternating parameter fitting
			Optimize(Feedback.UserMatrix, user_factors, item_factors);
			Optimize(Feedback.ItemMatrix, item_factors, user_factors);
		}

		/// <summary>Optimizes the specified data</summary>
		/// <param name="data">data</param>
		/// <param name="W">W</param>
		/// <param name="H">H</param>
		protected virtual void Optimize(IBooleanMatrix data, Matrix<float> W, Matrix<float> H)
		{
			var HH = new Matrix<double>(num_factors, num_factors);

			// comments are in terms of computing the user factors
			// ... works the same with users and items exchanged

			// (1) create HH in O(f^2|Items|)
			// HH is symmetric
			for (int f_1 = 0; f_1 < num_factors; f_1++)
				for (int f_2 = 0; f_2 < num_factors; f_2++)
				{
					double d = 0;
					for (int i = 0; i < H.dim1; i++)
						d += H[i, f_1] * H[i, f_2];
					HH[f_1, f_2] = d;
				}
			// (2) optimize all U
			// HC_minus_IH is symmetric
			Parallel.For(0, W.dim1, u =>
			{
				var row = data.GetEntriesByRow(u);
				// create HC_minus_IH in O(f^2|S_u|)
				var HC_minus_IH = new Matrix<double>(num_factors, num_factors);
				for (int f_1 = 0; f_1 < num_factors; f_1++)
					for (int f_2 = 0; f_2 < num_factors; f_2++)
					{
						double d = 0;
						foreach (int i in row)
							d += H[i, f_1] * H[i, f_2] * alpha;
						HC_minus_IH[f_1, f_2] = d;
					}
				// create HCp in O(f|S_u|)
				var HCp = new double[num_factors];
				for (int f = 0; f < num_factors; f++)
				{
					double d = 0;
					foreach (int i in row)
						d += H[i, f] * (1 + alpha);
					HCp[f] = d;
				}
				// create m = HH + HC_minus_IH + reg*I
				// m is symmetric
				// the inverse m_inv is symmetric
				var m = new DenseMatrix(num_factors, num_factors);
				for (int f_1 = 0; f_1 < num_factors; f_1++)
					for (int f_2 = 0; f_2 < num_factors; f_2++)
					{
						double d = HH[f_1, f_2] + HC_minus_IH[f_1, f_2];
						if (f_1 == f_2)
							d += regularization;
						m[f_1, f_2] = d;
					}
				var m_inv = m.Inverse();
				// write back optimal W
				for (int f = 0; f < num_factors; f++)
				{
					double d = 0;
					for (int f_2 = 0; f_2 < num_factors; f_2++)
						d += m_inv[f, f_2] * HCp[f_2];
					W[u, f] = (float) d;
				}
			});
		}

		///
		public override void AddFeedback(ICollection<Tuple<int, int>> feedback)
		{
			base.AddFeedback(feedback);
			//FoldIn(feedback);
		}
		
		///
		public override void RemoveFeedback(ICollection<Tuple<int, int>> feedback)
		{
			base.RemoveFeedback(feedback);
			// TODO
			//FoldOut(feedback);
		}

		/*
		void FoldIn(ICollection<Tuple<int, int>> feedback)
		{
			foreach (var entry in feedback)
				UpdateFactors(entry.Item1, entry.Item2, UpdateUsers, UpdateItems);
		}
		*/
		
		///
		protected override void AddUser(int user_id)
		{
			base.AddUser(user_id);
			
			user_factors.AddRows(user_id + 1);
			user_factors.RowInitNormal(user_id, InitMean, InitStdDev);
		}
		
		///
		protected override void AddItem(int item_id)
		{
			base.AddItem(item_id);
			
			item_factors.AddRows(item_id + 1);
			item_factors.RowInitNormal(item_id, InitMean, InitStdDev);
		}
		
		
		///
		public override void RemoveUser(int user_id)
		{
			base.RemoveUser(user_id);
			
			// set user latent factors to zero
			user_factors.SetRowToOneValue(user_id, 0);
		}
		
		///
		public override void RemoveItem(int item_id)
		{
			base.RemoveItem(item_id);
			
			// set item latent factors to zero
			item_factors.SetRowToOneValue(item_id, 0);
		}
		
		/* TODO
		/// <summary>
		/// Performs factor updates for a user and item pair.
		/// </summary>
		/// <param name="user_id">User_id.</param>
		/// <param name="item_id">Item_id.</param>
		protected virtual void UpdateFactors(int user_id, int item_id, bool update_user, bool update_item)
		{
			//Console.WriteLine(float.MinValue);
			float err = 1 - Predict(user_id, item_id);
			
			// adjust factors
			if(update_user)
			{
				// Get user's items
				var user_items = Feedback.UserMatrix.GetEntriesByRow(user_id);
				var if_t_if = new Matrix<float>(MaxItemID,MaxItemID);
				for (int i_ft = 0; i_ft < num_factors; i_ft++)
					for (int i_f = 0; i_f < num_factors; i_f++)
					{
						if_t_if[i_ft, i_f] = 0;
						for (int i = 0; i <= MaxItemID; i++)
							if_t_if[if_t, i_f] += item_factors[i, i_ft] * item_factors[i, i_f];
					}

				for (int f = 0; f < NumFactors; f++)
				{
					float u_f = user_factors[user_id, f];
				
				// if necessary, compute and apply updates
				if (update_user)
				{
					// user_factors[user_id] += 
					user_factors.Inc(user_id, f, current_learnrate * delta_u);
				}
				if (update_item)
				{
					double delta_i = err * u_f - Regularization * i_f;
					item_factors.Inc(item_id, f, current_learnrate * delta_i);
				}
			}
			
		}
		*/
		
		///
		public override float ComputeObjective()
		{
			return -1;
		}

		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"WRMF num_factors={0} regularization={1} alpha={2} num_iter={3}",
				NumFactors, Regularization, Alpha, NumIter);
		}
	}
}