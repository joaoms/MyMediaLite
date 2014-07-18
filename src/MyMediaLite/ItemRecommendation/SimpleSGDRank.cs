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
	public class SimpleSGDRank : SimpleSGD
	{
		/// <summary>Maximum list size</summary>
		public int MaxListSize { get { return max_list_size; } set { max_list_size = value; } }
		int max_list_size = -1;

		///
		protected override float Predict(int user_id, int item_id, bool bound)
		{
			if (user_id >= user_factors.dim1 || item_id >= item_factors.dim1)
				return float.MinValue;

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
		
		/// <summary>
		/// Performs factor updates for a user and item pair.
		/// </summary>
		/// <param name="user_id">User_id.</param>
		/// <param name="item_id">Item_id.</param>
		protected override void UpdateFactors(int user_id, int item_id, bool update_user, bool update_item)
		{
			var ignore_items = new System.Collections.Generic.HashSet<int>(Feedback.UserMatrix[user_id]);
			ignore_items.Remove(item_id);
			var reclist = Recommend(user_id, MaxListSize, ignore_items);
			float rel_position = 1;
			if (reclist.Count > 0)
			{
				int i = 0;
				while (reclist[i].Item1 != item_id && i < reclist.Count-1) i++;
				rel_position = (float) i / reclist.Count;
			}
			
			//Console.WriteLine(rel_position);
			float err = rel_position;

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
		public override System.Collections.Generic.IList<Tuple<int, float>> Recommend(
			int user_id, int n = -1,
			System.Collections.Generic.ICollection<int> ignore_items = null,
			System.Collections.Generic.ICollection<int> candidate_items = null)
		{
			if (candidate_items == null)
				candidate_items = Enumerable.Range(0, MaxItemID - 1).ToList();
			if (ignore_items == null)
				ignore_items = new int[0];

			System.Collections.Generic.IList<Tuple<int, float>> ordered_items;

			if (n == -1)
			{
				var scored_items = new List<Tuple<int, float>>();
				foreach (int item_id in candidate_items)
					if (!ignore_items.Contains(item_id))
					{
						float error = Math.Abs(Predict(user_id, item_id));
						if (error > float.MaxValue)
							error = float.MaxValue;
						scored_items.Add(Tuple.Create(item_id, error));
					}

				ordered_items = scored_items.OrderBy(x => x.Item2).ToArray();
			}
			else
			{
				var comparer = new DelegateComparer<Tuple<int, float>>( (a, b) => a.Item2.CompareTo(b.Item2) );
				var heap = new IntervalHeap<Tuple<int, float>>(n, comparer);
				float max_error = float.MaxValue;

				foreach (int item_id in candidate_items)
					if (!ignore_items.Contains(item_id))
					{
						float error = Math.Abs(Predict(user_id, item_id));
						if (error < max_error)
						{
							heap.Add(Tuple.Create(item_id, error));
							if (heap.Count > n)
							{
								heap.DeleteMax();
								max_error = heap.FindMax().Item2;
							}
						}
					}

				ordered_items = new Tuple<int, float>[heap.Count];
				for (int i = 0; i < ordered_items.Count; i++)
					ordered_items[i] = heap.DeleteMin();
			}

			return ordered_items;
		}


		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"SimpleSGDRank num_factors={0} regularization={1} learn_rate={2} num_iter={3} incr_iter={4} decay={5}",
				NumFactors, Regularization, LearnRate, NumIter, IncrIter, Decay);
		}


	}
}

