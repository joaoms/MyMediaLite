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
using System.Collections.Generic;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;
using MyMediaLite.DataType;
using MyMediaLite.Data;

namespace MyMediaLite.ItemRecommendation
{
	public class ItemSimMF : IncrementalItemRecommender, IIterativeModel
	{
		protected Matrix<float> row_factors;
		protected Matrix<float> col_factors;

		/// <summary>Mean of the normal distribution used to initialize the latent factors</summary>
		public double InitMean { get; set; }

		/// <summary>Standard deviation of the normal distribution used to initialize the latent factors</summary>
		public double InitStdDev { get; set; }

		/// <summary>Number of latent factors per user/item</summary>
		public uint NumFactors { get { return (uint) num_factors; } set { num_factors = (int) value; } }
		/// <summary>Number of latent factors per user/item</summary>
		protected int num_factors = 10;

		/// <summary>Number of iterations over the training data</summary>
		public uint NumIter { get; set; }

		/// <summary>Regularization parameter</summary>
		public double Regularization { get { return regularization; } set { regularization = value; } }
		double regularization = 0.032;

		/// <summary>Learn rate (update step size)</summary>
		public float LearnRate { get { return learn_rate; } set { learn_rate = value; } }
		float learn_rate = 0.31f;



		public ItemSimMF ()
		{
			NumIter = 30;
			InitMean = 0;
			InitStdDev = 0.1;
		}

		protected override void InitModel()
		{
			row_factors = new Matrix<float>(MaxItemID + 1, NumFactors);
			col_factors = new Matrix<float>(MaxItemID + 1, NumFactors);

			row_factors.InitNormal(InitMean, InitStdDev);
			col_factors.InitNormal(InitMean, InitStdDev);

		}

		public override float ComputeObjective()
		{
			return -1;
		}

		/// <summary>Iterate once over feedback data and adjust corresponding factors (stochastic gradient descent)</summary>
		protected virtual void Iterate()
		{
			//TODO: respeitar sequencia original
			// ou não: o objetivo é ter uma sequência diferente por cada iteração.
			// só se aplica ao treino inicial
			var indexes = Feedback.RandomIndex;
			foreach (int index in indexes)
			{
				int u = Feedback.Users[index];
				int i = Feedback.Items[index];
				//Console.WriteLine("User " + u + " Item " + i);

				UpdateFactors(u, i, update_user, update_item);
			}

		}

		private float GetCorrelation(int item_i, int item_j, bool bound)
		{
			if(item_i >= row_factors.dim1 || item_j >= col_factors.dim1)
				return float.MinValue;
		
			float result = DataType.MatrixExtensions.RowScalarProduct(row_factors, item_i, col_factors, item_j);

			if (bound)
			{
				if (result > 1)
					return 1;
				if (result < 0)
					return 0;
			}

			return result;
		}

		///
		protected float Predict(int user_id, int item_id, bool bound)
		{
			if (user_id > MaxUserID)
				return float.MinValue;
			if (item_id > MaxItemID)
				return float.MinValue;

			double corr;
			double sum = 0;
			double normalization = 0;
			foreach (int item in Feedback.AllItems)
			{
				corr = Math.Pow(GetCorrelation(item_id, item, true), Q);
				normalization += corr;
				if (Feedback.ItemMatrix[neighbor, user_id])
					sum += corr;
			}
			if (sum == 0) return 0;
			return (float) (sum / normalization);
		}
		
		///
		public override void AddFeedback(ICollection<Tuple<int, int>> feedback)
		{
			base.AddFeedback(feedback);
			Retrain(feedback);
		}
		
		///
		public override void RemoveFeedback(ICollection<Tuple<int, int>> feedback)
		{
			base.RemoveFeedback(feedback);
			Retrain(feedback);
		}
		
		void Retrain(ICollection<Tuple<int, int>> feedback)
		{
			foreach (var entry in feedback)
				UpdateFactors(entry.Item1, entry.Item2, UpdateUsers, UpdateItems);
		}
		
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
		
		/// <summary>
		/// Performs factor updates for a user and item pair.
		/// </summary>
		/// <param name="user_id">User_id.</param>
		/// <param name="item_id">Item_id.</param>
		protected virtual void UpdateFactors(int user_id, int item_id, bool update_user, bool update_item)
		{
			//Console.WriteLine(float.MinValue);
			float err = 1 - Predict(user_id, item_id, false);
			
			// adjust factors
			for (int f = 0; f < NumFactors; f++)
			{
				float u_f = user_factors[user_id, f];
				float i_f = item_factors[item_id, f];
				
				// if necessary, compute and apply updates
				if (update_user)
				{
					double delta_u = err * i_f - Regularization * u_f;
					user_factors.Inc(user_id, f, current_learnrate * delta_u);
				}
				if (update_item)
				{
					double delta_i = err * u_f - Regularization * i_f;
					item_factors.Inc(item_id, f, current_learnrate * delta_i);
				}
			}

		}

		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"NaiveSVD num_factors={0} regularization={1} learn_rate={2} num_iter={3} decay={4}",
				NumFactors, Regularization, LearnRate, NumIter, Decay);
		}


	}
}

