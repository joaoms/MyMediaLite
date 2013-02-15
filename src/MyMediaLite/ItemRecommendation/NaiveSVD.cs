// Copyright (C) 2013 Zeno Gantner
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
//
using System;
using System.Globalization;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;
using MyMediaLite.DataType;
using MyMediaLite.Data;

namespace MyMediaLite.ItemRecommendation
{
	public class NaiveSVD : MF
	{
		/// <summary>Regularization parameter</summary>
		public double Regularization { get { return regularization; } set { regularization = value; } }
		double regularization = 0.01;

		/// <summary>Learn rate (update step size)</summary>
		public float LearnRate { get { return learn_rate; } set { learn_rate = value; } }
		float learn_rate = 0.001f;

		public NaiveSVD ()
		{
			NumFactors = 200;
		}

		///
		public override void Iterate()
		{
			Console.WriteLine("Iteration");
			Iterate(true, true);
		}

		public override float ComputeObjective()
		{
			return -1;
		}

		/// <summary>Iterate once over feedback data and adjust corresponding factors (stochastic gradient descent)</summary>
		/// <param name="rating_indices">a list of indices pointing to the feedback to iterate over</param>
		/// <param name="update_user">true if user factors to be updated</param>
		/// <param name="update_item">true if item factors to be updated</param>
		protected virtual void Iterate(bool update_user, bool update_item)
		{
			var indexes = Feedback.RandomIndex;
			foreach (int index in indexes)
			{
				int u = Feedback.Users[index];
				int i = Feedback.Items[index];
				//Console.WriteLine("User " + u + " Item " + i);


				float err = 1 - Predict(u, i, false);
				
				// adjust factors
				for (int f = 0; f < NumFactors; f++)
				{
					float u_f = user_factors[u, f];
					float i_f = item_factors[i, f];
					
					// if necessary, compute and apply updates
					if (update_user)
					{
						double delta_u = err * i_f - Regularization * u_f;
						user_factors.Inc(u, f, learn_rate * delta_u);
					}
					if (update_item)
					{
						double delta_i = err * u_f - Regularization * i_f;
						item_factors.Inc(i, f, learn_rate * delta_i);
					}
				}
			}
			
		}

		///
		protected float Predict(int user_id, int item_id, bool bound)
		{
			float result = DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);
			
			if (bound)
			{
				if (result > 1)
					return 1;
				if (result < 0)
					return 0;
			}
			return result;
		}
		


	}
}

