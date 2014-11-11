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
using Uhlme; 

namespace MyMediaLite.ItemRecommendation
{
	public class SimpleSGDPoisson : SimpleSGD
	{

		public bool ForgetUsersInUsers { get; set; }
		public bool ForgetItemsInUsers { get; set; }
		public bool ForgetUsersInItems { get; set; }
		public bool ForgetItemsInItems { get; set; }
		public int ForgetHorizon { get { return forget_horizon; } set { forget_horizon = value; } }
		public float Alpha { get { return alpha; } set { alpha = value; } }
		int forget_horizon = 1;
		float alpha = 0.001f;
		MyMediaLite.Random rand = MyMediaLite.Random.GetInstance();

		private bool forget_users;
		private bool forget_items;

		HeapPriorityQueue<int> item_queue;
		HeapPriorityQueue<int> user_queue;

		// float max_score = 1.0f;

		public SimpleSGDPoisson()
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
				user_queue = new HeapPriorityQueue<int>(Feedback.Users.Count);
				foreach (int usr in Feedback.Users)
					user_queue.AddOrUpdate(usr, Alpha);
			}
			if (forget_items)
			{
				item_queue = new HeapPriorityQueue<int>(Feedback.Items.Count);
				foreach (int item in Feedback.Items)
				{
					//Console.WriteLine("Adding item "+item);
					item_queue.AddOrUpdate(item, Alpha);
				}
				//DEBUG stuff
				/*
				int cnt;
				Console.WriteLine("Heap has "+ (cnt = item_queue.Count) +" elements");

				for (int i = 0; i < cnt; i++)
					Console.WriteLine("Prio: "+item_queue.PeekPriority()+" item "+item_queue.Dequeue());
				*/
			}
		}

		protected override void Retrain(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			int user_horizon = ForgetHorizon;
			int item_horizon = ForgetHorizon;

			if (forget_users)
				user_horizon = Math.Min(user_horizon, user_queue.Count - 1);
			int[] qu = new int[user_horizon];
			double[] qu_prio = new double[user_horizon];

			if (forget_items)
				item_horizon = Math.Min(item_horizon, item_queue.Count - 1);

			int[] qi = new int[item_horizon];
			double[] qi_prio = new double[item_horizon];

			foreach (var entry in feedback)
			{
				if (forget_users)
				{
					for (uint i = 0; i < qu.Length; i++)
					{
						qu_prio[i] = user_queue.PeekPriority();
						qu[i] = user_queue.Dequeue();
					}
				}
				if (forget_items)
				{
					for (uint i = 0; i < qi.Length; i++)
					{
						qi_prio[i] = item_queue.PeekPriority();
						qi[i] = item_queue.Dequeue();
					}
				}
				//Console.WriteLine("Forgetting item "+qi);
				if (forget_users)
					foreach (var usr in qu.Reverse())
						if (usr >= 0 && usr != entry.Item1) 
							for (uint i = 0; i < IncrIter; i++)
								UpdateFactors(usr, entry.Item2, ForgetUsersInUsers, ForgetUsersInItems, 0);
				if (forget_items)
					foreach (var itm in qi.Reverse())
						if (itm >= 0 && itm != entry.Item2)
							for (uint i = 0; i < IncrIter; i++)
								UpdateFactors(entry.Item1, itm, ForgetItemsInUsers, ForgetItemsInItems, 0);
				for (uint i = 0; i < IncrIter; i++)
					UpdateFactors(entry.Item1, entry.Item2, UpdateUsers, UpdateItems, 1);

				if (forget_items)
				{
					item_queue.AddOrUpdate(entry.Item2, Alpha);

					for (int i = qi.Length - 1; i >= 0; i--)
						if (qi[i] != entry.Item2)
						{
							var lambda = alpha + ((1-alpha)*qi_prio[i]);
							item_queue.Enqueue(qi[i], lambda);
						}
				}
				if (forget_users)
				{
					user_queue.AddOrUpdate(entry.Item1, Alpha);

					for (int i = qu.Length - 1; i >= 0; i--)
						if (qu[i] != entry.Item1)
						{
							var lambda = alpha + ((1-alpha)*qu_prio[i]);
							user_queue.Enqueue(qu[i], lambda);
						}
				}
			}
			if(rand.NextDouble() <= 0.001)
				dumpItemQueue();
		}

		private void dumpItemQueue()
		{
			Console.WriteLine("\nupdate " + Feedback.Count);
			double prio;
			int item;
			int cnt = item_queue.Count;
			HeapPriorityQueue<int> newHeap = new HeapPriorityQueue<int>(cnt);
			for (int i = 0; i < cnt; i++)
			{
				prio = item_queue.PeekPriority();
				item = item_queue.Dequeue();
				Console.WriteLine(prio + "\t" + item);
				newHeap.Enqueue(item, prio);
			}
			item_queue = newHeap;
		}

		///
		public override void RemoveItem(int item_id)
		{
			base.RemoveItem(item_id);

			if (forget_items)
			{
				// TODO: Implement queue remove
				//item_queue.Remove(item_id);
			}
		}

		///
		public override void RemoveUser(int user_id)
		{
			base.RemoveUser(user_id);

			if (forget_users)
			{
				// TODO: Implement queue remove
				//user_queue.Remove(user_id);
			}
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
				"SimpleSGDPoisson num_factors={0} regularization={1} learn_rate={2} num_iter={3} incr_iter={4} decay={5}, forget_horizon={6}, alpha={7}",
				NumFactors, Regularization, LearnRate, NumIter, IncrIter, Decay, ForgetHorizon, Alpha);
		}


	}

}

