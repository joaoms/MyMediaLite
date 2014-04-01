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
using C5;
using System.Linq;
using System.Globalization;
using System.Collections.Generic;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;
using MyMediaLite.DataType;
using MyMediaLite.Data;

namespace MyMediaLite.ItemRecommendation
{
	public class NaiveSVDZI : NaiveSVD
	{
		protected HashedLinkedList<int> item_queue;

		// float max_score = 1.0f;

		protected override void InitModel()
		{
			base.InitModel();

			item_queue = new HashedLinkedList<int>();
			item_queue.AddAll(Feedback.Items);
		}

		/// <summary>Iterate once over feedback data and adjust corresponding factors (stochastic gradient descent)</summary>
		/// <param name="rating_indices">a list of indices pointing to the feedback to iterate over</param>
		/// <param name="update_user">true if user factors to be updated</param>
		/// <param name="update_item">true if item factors to be updated</param>
		protected override void Iterate(bool update_user, bool update_item)
		{
			//TODO: respeitar sequencia original
			// ou não: o objetivo é ter uma sequência diferente por cada iteração.
			// só se aplica ao treino inicial
			/*
			 * var indexes = Feedback.RandomIndex;
			 * foreach (int index in indexes)
			 */
			for (int index = 0; index < Feedback.Count; index++)
			{
				int u = Feedback.Users[index];
				int i = Feedback.Items[index];
				//Console.WriteLine("User " + u + " Item " + i);

				UpdateFactors(u, i, update_user, update_item);
			}

			UpdateLearnRate();
			
		}

		void Retrain(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			foreach (var entry in feedback)
			{
				int qi;
				do 
					qi = item_queue.RemoveFirst();
				while (qi == entry.Item2);
				for (uint i = 0; i < IncrIter; i++)
				{
					UpdateFactors(entry.Item1, qi, UpdateUsers, UpdateItems, 0);
					UpdateFactors(entry.Item1, entry.Item2, UpdateUsers, UpdateItems, 1);
				}
				item_queue.Remove(entry.Item2);
				item_queue.InsertLast(entry.Item2);
				item_queue.InsertLast(qi);
			}
		}

		///
		public override void RemoveItem(int item_id)
		{
			base.RemoveItem(item_id);
			
			item_queue.Remove(item_id);
		}
		
		/// <summary>
		/// Performs factor updates for a user and item pair.
		/// </summary>
		/// <param name="user_id">User_id.</param>
		/// <param name="item_id">Item_id.</param>
		protected virtual void UpdateFactors(int user_id, int item_id, bool update_user, bool update_item, float base_val = 1)
		{
			//Console.WriteLine(float.MinValue);
			float err = base_val - Predict(user_id, item_id, false);

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
				"NaiveSVDZI num_factors={0} regularization={1} learn_rate={2} num_iter={3} incr_iter={4} decay={5}",
				NumFactors, Regularization, LearnRate, NumIter, IncrIter, Decay);
		}


	}
}

