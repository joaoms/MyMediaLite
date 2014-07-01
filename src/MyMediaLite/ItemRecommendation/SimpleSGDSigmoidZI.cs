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
	public class SimpleSGDSigmoidZI : SimpleSGDSigmoid
	{

		public bool ForgetUsersInUsers { get; set; }
		public bool ForgetItemsInUsers { get; set; }
		public bool ForgetUsersInItems { get; set; }
		public bool ForgetItemsInItems { get; set; }

		private bool forget_users;
		private bool forget_items;

		protected HashedLinkedList<int> item_queue;
		protected HashedLinkedList<int> user_queue;

		// float max_score = 1.0f;

		public SimpleSGDSigmoidZI()
		{
			ForgetUsersInUsers = false;
			ForgetUsersInItems = false;
			ForgetItemsInUsers = true;
			ForgetItemsInItems = false;
		}

		protected override void InitModel()
		{
			base.InitModel();

			forget_users = ForgetUsersInUsers || ForgetUsersInItems;
			forget_items = ForgetItemsInUsers || ForgetItemsInItems;

			if (forget_users)
			{
				user_queue = new HashedLinkedList<int>();
				user_queue.AddAll(Feedback.Users);
			}
			if (forget_items)
			{
				item_queue = new HashedLinkedList<int>();
				item_queue.AddAll(Feedback.Items);
			}
		}

		protected override void Retrain(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			foreach (var entry in feedback)
			{
				int qu = -1;
				int qi = -1;
				if (forget_users)
				{
					do 
						qu = user_queue.RemoveFirst();
					while (qu == entry.Item1);
				}
				if (forget_items)
				{
					do 
						qi = item_queue.RemoveFirst();
					while (qi == entry.Item2);
				}
				//Console.WriteLine("Forgetting item "+qi);
				for (uint i = 0; i < IncrIter; i++)
				{
					if (forget_users && qu >= 0)
						UpdateFactors(qu, entry.Item2, ForgetUsersInUsers, ForgetUsersInItems, 0);
					if (forget_items && qi >= 0)
						UpdateFactors(entry.Item1, qi, ForgetItemsInUsers, ForgetItemsInItems, 0);
					UpdateFactors(entry.Item1, entry.Item2, UpdateUsers, UpdateItems, 1);
				}
				if (forget_items)
				{
					item_queue.Remove(entry.Item2);
					item_queue.InsertLast(entry.Item2);
					if (qi >= 0)
						item_queue.InsertLast(qi);
				}
				if (forget_users)
				{
					user_queue.Remove(entry.Item1);
					user_queue.InsertLast(entry.Item1);
					if (qu >= 0)
						user_queue.InsertLast(qu);
				}
			}
		}

		///
		public override void RemoveItem(int item_id)
		{
			base.RemoveItem(item_id);

			if (forget_items)
				item_queue.Remove(item_id);
		}

		///
		public override void RemoveUser(int user_id)
		{
			base.RemoveUser(user_id);

			if (forget_users)
				user_queue.Remove(user_id);
		}

		/// <summary>
		/// Performs factor updates for a user and item pair.
		/// </summary>
		/// <param name="user_id">User_id.</param>
		/// <param name="item_id">Item_id.</param>
		protected virtual void UpdateFactors(int user_id, int item_id, bool update_user, bool update_item, float base_val = 1)
		{
			//Console.WriteLine(float.MinValue);
			float score = Predict(user_id, item_id);
			float err = base_val - score;
			float gradient = (float) (err * score * (1 - score));

			// adjust factors
			for (int f = 0; f < NumFactors; f++)
			{
				float u_f = user_factors[user_id, f];
				float i_f = item_factors[item_id, f];
				
				// if necessary, compute and apply updates
				if (update_user)
				{
					double delta_u = gradient * i_f - Regularization * u_f;
					user_factors.Inc(user_id, f, current_learnrate * delta_u);
				}
				if (update_item)
				{
					double delta_i = gradient * u_f - Regularization * i_f;
					item_factors.Inc(item_id, f, current_learnrate * delta_i);
				}
			}

		}


		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"NaiveSVDSigmoidZI num_factors={0} regularization={1} learn_rate={2} num_iter={3} incr_iter={4} decay={5}",
				NumFactors, Regularization, LearnRate, NumIter, IncrIter, Decay);
		}


	}
}

