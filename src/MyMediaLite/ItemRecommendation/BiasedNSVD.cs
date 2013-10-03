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
	public class BiasedNSVD : NaiveSVD
	{
		/// <summary>The bias (global average)</summary>
		protected float global_bias;

		/// <summary>rating biases of the users</summary>
		protected internal float[] user_bias;

		/// <summary>rating biases of the items</summary>
		protected internal float[] item_bias;

		public BiasedNSVD ()
		{
		}

		protected override void InitModel()
		{
			base.InitModel();
			user_bias = new float[MaxUserID + 1];
			item_bias = new float[MaxItemID + 1];
		}

		///
		public override void Train()
		{
			InitModel();

			global_bias = Feedback.Count / ((MaxUserID + 1) * (MaxItemID + 1));

			for (uint i = 0; i < NumIter; i++)
				Iterate();
		}


		///
		protected float Predict(int user_id, int item_id, bool bound)
		{
			float result = global_bias;

			if (user_id < user_bias.Length)
				result += user_bias[user_id];
			if (item_id < item_bias.Length)
				result += item_bias[item_id];
			if (user_id >= user_factors.dim1 || item_id >= item_factors.dim1)
				result += DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);

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
		protected override void AddUser(int user_id)
		{
			base.AddUser(user_id);
			Array.Resize(ref user_bias, MaxUserID + 1);
		}

		///
		protected override void AddItem(int item_id)
		{
			base.AddItem(item_id);
			Array.Resize(ref item_bias, MaxItemID + 1);
		}


		///
		public override void RemoveUser(int user_id)
		{
			user_bias[user_id] = 0;
			base.RemoveUser(user_id);
		}

		///
		public override void RemoveItem(int item_id)
		{
			item_bias[item_id] = 0;
			base.RemoveItem(item_id);
		}

		/// <summary>
		/// Performs factor updates for a user and item pair.
		/// </summary>
		/// <param name="user_id">User_id.</param>
		/// <param name="item_id">Item_id.</param>
		protected override void UpdateFactors(int user_id, int item_id, bool update_user, bool update_item)
		{
			//Console.WriteLine(float.MinValue);
			float err = 1 - Predict(user_id, item_id, false);

			// adjust biases
			if (update_user)
				user_bias[user_id] += current_learnrate * (err - (float) Regularization * user_bias[user_id]);
			if (update_item)
				item_bias[item_id] += current_learnrate * (err - (float) Regularization * item_bias[item_id]);

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
				"BiasedNSVD num_factors={0} regularization={1} learn_rate={2} num_iter={3} decay={4}",
				NumFactors, Regularization, LearnRate, NumIter, Decay);
		}


	}
}

