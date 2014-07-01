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
	public class SimpleSGDSigmoid : SimpleSGD
	{

		protected double SigmoidFunc(double x)
		{
			return (1 / (1 + Math.Exp(-x)));
		}

		///
		public override float Predict(int user_id, int item_id)
		{
			if (user_id >= user_factors.dim1 || item_id >= item_factors.dim1)
				return float.MinValue;

			float result = DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);

			return (float) SigmoidFunc(result);
		}
		
		/// <summary>
		/// Performs factor updates for a user and item pair.
		/// </summary>
		/// <param name="user_id">User_id.</param>
		/// <param name="item_id">Item_id.</param>
		protected override void UpdateFactors(int user_id, int item_id, bool update_user, bool update_item)
		{
			//Console.WriteLine(float.MinValue);
			float score = Predict(user_id, item_id);
			float gradient = (float) Math.Pow((1 - score), 2) * score;

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
				if (UseMulticore)
				{
					Parallel.ForEach(candidate_items, item_id => {
						if (!ignore_items.Contains(item_id))
						{
							float score = Predict(user_id, item_id);
							if (score < float.MinValue)
								score = float.MinValue;
							lock(scored_items)
								scored_items.Add(Tuple.Create(item_id, score));
						}
					});
				} else {
					foreach (int item_id in candidate_items)
						if (!ignore_items.Contains(item_id))
						{
							float score = Predict(user_id, item_id);
							if (score < float.MinValue)
								score = float.MinValue;
							scored_items.Add(Tuple.Create(item_id, score));
						}
				}

				ordered_items = scored_items.OrderByDescending(x => x.Item2).ToArray();
			}
			else
			{
				var comparer = new DelegateComparer<Tuple<int, float>>( (a, b) => a.Item2.CompareTo(b.Item2) );
				var heap = new IntervalHeap<Tuple<int, float>>(n, comparer);
				float min_score = float.MinValue;

				if (UseMulticore)
				{
					Parallel.ForEach(candidate_items, item_id => {
						if (!ignore_items.Contains(item_id))
						{
							float score = Predict(user_id, item_id);
							if (score > min_score)
							{
								lock (heap)
								{
									heap.Add(Tuple.Create(item_id, score));
									if (heap.Count > n)
									{
										heap.DeleteMin();
										min_score = heap.FindMin().Item2;
									}
								}
							}
						}
					});
				} else {
					foreach (int item_id in candidate_items)
						if (!ignore_items.Contains(item_id))
						{
							float score = Predict(user_id, item_id);
							if (score > min_score)
							{
								heap.Add(Tuple.Create(item_id, score));
								if (heap.Count > n)
								{
									heap.DeleteMin();
									min_score = heap.FindMin().Item2;
								}
							}
						}
				}

				ordered_items = new Tuple<int, float>[heap.Count];
				for (int i = 0; i < ordered_items.Count; i++)
					ordered_items[i] = heap.DeleteMax();
			}

			return ordered_items;
		}


		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"NaiveSVDSigmoid num_factors={0} regularization={1} learn_rate={2} num_iter={3} incr_iter={4} decay={5}",
				NumFactors, Regularization, LearnRate, NumIter, IncrIter, Decay);
		}


	}
}

