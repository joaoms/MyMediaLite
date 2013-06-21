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
	/// <summary>
	/// Item sim M.
	/// </summary>
	public class ItemSimMF : IncrementalItemRecommender, IIterativeModel
	{
		/// <summary>
		/// The row_factors.
		/// </summary>
		protected Matrix<float> row_factors;

		/// <summary>
		/// The col_factors.
		/// </summary>
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
		
		/// <summary>Alpha parameter for </summary>
		public float Alpha { get { return alpha; } set { alpha = value; } }
		float alpha = 0.4f;
		


		/// <summary>
		/// Initializes a new instance of the <see cref="MyMediaLite.ItemRecommendation.ItemSimMF"/> class.
		/// </summary>
		public ItemSimMF ()
		{
			NumIter = 30;
			InitMean = 0;
			InitStdDev = 0.1;
		}

		/// <summary>
		/// Inits the model.
		/// </summary>
		protected virtual void InitModel()
		{
			row_factors = new Matrix<float>(MaxItemID + 1, NumFactors);
			col_factors = new Matrix<float>(MaxItemID + 1, NumFactors);

			row_factors.InitNormal(InitMean, InitStdDev);
			col_factors.InitNormal(InitMean, InitStdDev);

		}

		public override void Train()
		{
			InitModel();

			for (uint i = 0; i < NumIter; i++)
				Iterate();

		}

		/// <summary>
		/// Compute the current optimization objective (usually loss plus regularization term) of the model
		/// </summary>
		/// <returns>
		/// the current objective; -1 if not implemented
		/// </returns>
		public virtual float ComputeObjective()
		{
			return -1;
		}

		/// <summary>Iterate once over feedback data and adjust corresponding factors (stochastic gradient descent)</summary>
		public void Iterate()
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

				UpdateFactors(u, i);
			}

		}

		private float GetCorrelation(int item_i, int item_j, bool bound)
		{
			if (item_i >= row_factors.dim1 || item_j >= col_factors.dim1)
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
		public override float Predict(int user_id, int item_id)
		{
			if (user_id > MaxUserID)
				return float.MinValue;
			if (item_id > MaxItemID)
				return float.MinValue;

			double sum = 0;
			var user_items = Feedback.UserMatrix.GetEntriesByRow(user_id);
			var user_items_count = user_items.Count;
			if (user_items_count == 0)
				return float.MinValue;

			foreach (int item in user_items) 
			{
				if (item != item_id) 
					sum += GetCorrelation (item_id, item, false);
				else {
					user_items_count--;
					if (user_items_count == 0)
						return float.MinValue;
				}
			}

			var normalization = Math.Pow(user_items_count, Alpha);
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
				UpdateFactors(entry.Item1, entry.Item2);
		}
		
		///
		protected override void AddItem(int item_id)
		{
			base.AddItem(item_id);
			
			row_factors.AddRows(item_id + 1);
			col_factors.AddRows(item_id + 1);
			row_factors.RowInitNormal(item_id, InitMean, InitStdDev);
			col_factors.RowInitNormal(item_id, InitMean, InitStdDev);
		}

		///
		public override void RemoveItem(int item_id)
		{
			base.RemoveItem(item_id);
			
			// set item latent factors to zero
			row_factors.SetRowToOneValue(item_id, 0);
			col_factors.SetRowToOneValue(item_id, 0);
		}
		
		/// <summary>
		/// Performs factor updates for a user and item pair.
		/// </summary>
		/// <param name="user_id">User_id.</param>
		/// <param name="item_id">Item_id.</param>
		protected virtual void UpdateFactors(int user_id, int item_id)
		{
			//Console.WriteLine(float.MinValue);
			float err = 1 - Predict(user_id, item_id);
			
			// adjust factors
			for (int f = 0; f < NumFactors; f++)
			{
				float r_f = row_factors[item_id, f];
				float c_f = col_factors[item_id, f];
				
				// if necessary, compute and apply updates
				double delta_r = err * c_f - Regularization * r_f;
				row_factors.Inc(item_id, f, LearnRate * delta_r);

				double delta_c = err * r_f - Regularization * c_f;
				col_factors.Inc(item_id, f, LearnRate * delta_c);
			}

		}

		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"ItemSimMF num_factors={0} regularization={1} learn_rate={2} num_iter={3} alpha={4}",
				NumFactors, Regularization, LearnRate, NumIter, Alpha);
		}


	}
}

