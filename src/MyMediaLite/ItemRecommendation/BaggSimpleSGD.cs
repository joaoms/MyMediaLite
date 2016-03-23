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
using MathNet.Numerics.Distributions;

namespace MyMediaLite.ItemRecommendation
{
	public class BaggSimpleSGD : MF
	{
		/// <summary>Regularization parameter</summary>
		public double Regularization { get { return regularization; } set { regularization = value; } }
		double regularization = 0.032;

		/// <summary>Learn rate (update step size)</summary>
		public float LearnRate { get { return learn_rate; } set { learn_rate = value; } }
		float learn_rate = 0.31f;

		/// <summary>Multiplicative learn rate decay</summary>
		/// <remarks>Applied after each epoch (= pass over the whole dataset)</remarks>
		public float Decay { get { return decay; } set { decay = value; } }
		float decay = 1.0f;

		/// <summary>Incremental iteration number</summary>
		public uint IncrIter { get; set; }

		/// <summary>Incremental iteration number</summary>
		public bool UseMulticore { get { return use_multicore; } set { use_multicore = value; } }
		bool use_multicore = false;

		public int NumNodes { get { return num_nodes; } set { num_nodes = value; } }
		int num_nodes = 4;

		protected MyMediaLite.Random rand;

		protected List<SimpleSGD> recommender_nodes;

		//readonly double BAG_PROB = 1 - 1 / Math.E;


		// float max_score = 1.0f;

		public BaggSimpleSGD ()
		{
			UpdateUsers = true;
			UpdateItems = true;
			rand = MyMediaLite.Random.GetInstance();
		}

		protected override void InitModel()
		{
			recommender_nodes = new List<SimpleSGD>(num_nodes);
			SimpleSGD recommender_node;
			for (int i = 0; i < num_nodes; i++) {
				recommender_node = new SimpleSGD();
				recommender_node.UpdateUsers = true;
				recommender_node.UpdateItems = true;
				recommender_node.Regularization = this.Regularization;
				recommender_node.NumFactors = this.NumFactors;
				recommender_node.LearnRate = this.LearnRate;
				recommender_node.IncrIter = this.IncrIter;
				recommender_node.NumIter = this.NumIter;
				recommender_node.Decay = this.Decay;
				recommender_node.Feedback = this.Feedback;
				recommender_nodes.Add(recommender_node);
			}
		}

		public override void Train()
		{
			InitModel();
			Parallel.ForEach(recommender_nodes, rnode => { rnode.Train(); });
		}

		///
		public override void Iterate()
		{
			Parallel.ForEach(recommender_nodes, rnode => { rnode.Iterate(); });
		}

		public override float ComputeObjective()
		{
			return -1;
		}

		public override float Predict(int user_id, int item_id)
		{
			return Predict(user_id, item_id, false);
		}

		///
		protected virtual float Predict(int user_id, int item_id, bool bound)
		{
			if (user_id > recommender_nodes[0].MaxUserID || item_id >= recommender_nodes[0].MaxItemID)
				return float.MinValue;

			float result = 0f;
			foreach (var rnode in recommender_nodes)
				result += rnode.Predict(user_id, item_id);
			
			result /= num_nodes;

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
		public override void AddFeedback(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			base.AddFeedback(feedback);
			recommender_nodes.Shuffle();
			if (UseMulticore)
			{
				Parallel.For(0, num_nodes, i => {
					int npoisson = Poisson.Sample(rand,1);
					recommender_nodes[i].AddFeedbackRetrainN(feedback, npoisson);
				});
			}
			else
			{
				foreach (var rnode in recommender_nodes)
				{
					int npoisson = Poisson.Sample(rand,1);
					rnode.AddFeedbackRetrainN(feedback, npoisson);
				}
			}
		}

		///
		public override void RemoveFeedback(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			base.RemoveFeedback(feedback);
			foreach (var rnode in recommender_nodes)
				rnode.RemoveFeedback(feedback);
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
						float score = Predict(user_id, item_id);
						if (score < float.MinValue)
							score = float.MinValue;
						scored_items.Add(Tuple.Create(item_id, score));
					}

				ordered_items = scored_items.OrderByDescending(x => x.Item2).ToArray();
			}
			else
			{
				var comparer = new DelegateComparer<Tuple<int, float>>( (a, b) => a.Item2.CompareTo(b.Item2) );
				var heap = new IntervalHeap<Tuple<int, float>>(n, comparer);
				float min_score = float.MinValue;

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
				"BaggSimpleSGD num_factors={0} regularization={1} learn_rate={2} num_iter={3} incr_iter={4} decay={5} num_nodes={6}",
				NumFactors, Regularization, LearnRate, NumIter, IncrIter, Decay, NumNodes);
		}


	}
}

