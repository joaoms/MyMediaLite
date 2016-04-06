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
	public class BaggedBPRMF : MF
	{
		/// <summary>Sample positive observations with (true) or without (false) replacement</summary>
		public bool WithReplacement { get; set; }

		/// <summary>Sample uniformly from users</summary>
		public bool UniformUserSampling { get; set; }

		/// <summary>Regularization parameter for the bias term</summary>
		public float BiasReg { get; set; }

		/// <summary>Learning rate alpha</summary>
		public float LearnRate { get { return learn_rate; } set { learn_rate = value; } }
		/// <summary>Learning rate alpha</summary>
		protected float learn_rate = 0.05f;

		/// <summary>Regularization parameter for user factors</summary>
		public float RegU { get { return reg_u; } set { reg_u = value; } }
		/// <summary>Regularization parameter for user factors</summary>
		protected float reg_u = 0.0025f;

		/// <summary>Regularization parameter for positive item factors</summary>
		public float RegI { get { return reg_i; } set { reg_i = value;	} }
		/// <summary>Regularization parameter for positive item factors</summary>
		protected float reg_i = 0.0025f;

		/// <summary>Regularization parameter for negative item factors</summary>
		public float RegJ { get { return reg_j; } set { reg_j = value; } }
		/// <summary>Regularization parameter for negative item factors</summary>
		protected float reg_j = 0.00025f;

		/// <summary>If set (default), update factors for negative sampled items during learning</summary>
		public bool UpdateJ { get { return update_j; } set { update_j = value; } }
		/// <summary>If set (default), update factors for negative sampled items during learning</summary>
		protected bool update_j = true;

		/// <summary>Incremental iteration number</summary>
		public bool UseMulticore { get { return use_multicore; } set { use_multicore = value; } }
		bool use_multicore = true;

		public int NumNodes { get { return num_nodes; } set { num_nodes = value; } }
		int num_nodes = 4;

		public string AggregationStrategy { get { return aggregation_strategy; } set { aggregation_strategy = value; } }
		string aggregation_strategy = "average";

		protected MyMediaLite.Random rand;

		protected List<BPRMF> recommender_nodes;

		//readonly double BAG_PROB = 1 - 1 / Math.E;


		// float max_score = 1.0f;

		public BaggedBPRMF ()
		{
			UniformUserSampling = true;
			UpdateUsers = true;
			UpdateItems = false;
			rand = MyMediaLite.Random.GetInstance();
		}

		protected override void InitModel()
		{
			recommender_nodes = new List<BPRMF>(num_nodes);
			BPRMF recommender_node;
			for (int i = 0; i < num_nodes; i++) {
				recommender_node = new BPRMF();
				recommender_node.UniformUserSampling = true;
				recommender_node.UpdateUsers = true;
				recommender_node.UpdateItems = true;
				recommender_node.RegU = this.RegU;
				recommender_node.RegI = this.RegI;
				recommender_node.RegJ = this.RegJ;
				recommender_node.UpdateJ = this.UpdateJ;
				recommender_node.NumFactors = this.NumFactors;
				recommender_node.LearnRate = this.LearnRate;
				recommender_node.NumIter = this.NumIter;
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

		///
		public override float Predict(int user_id, int item_id)
		{
			if (user_id > recommender_nodes[0].MaxUserID || item_id >= recommender_nodes[0].MaxItemID)
				return float.MinValue;

			List<float> results = new List<float>(num_nodes);
			foreach (var rnode in recommender_nodes)
				results.Add(rnode.Predict(user_id, item_id));
			
			return results.Average();

		}

		///
		public override void AddFeedback(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			base.AddFeedback(feedback);
			recommender_nodes.Shuffle();
			foreach (var rnode in recommender_nodes)
			{
				int npoisson = Poisson.Sample(rand,1);
				rnode.AddFeedbackRetrainN(feedback,npoisson);
			}
			/*
			Parallel.For(0, num_nodes, i => {
				int npoisson = Poisson.Sample(rand,1);
				recommender_nodes[i].AddFeedbackRetrainN(feedback, npoisson);
			});
			*/
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
			var resultsLock = new Object();
			var results = new List<System.Collections.Generic.IList<Tuple<int,float>>>(num_nodes);

			Parallel.ForEach(recommender_nodes, rnode => {
				var res = rnode.Recommend(user_id, n, ignore_items, candidate_items);
				lock(resultsLock) results.Add(res);
			});

			switch(aggregation_strategy) {
			case "average":
				return AvgResults(results);
			case "rank_average":
				return RankAvgResults(results);
			case "cooccurrence":
				return CooResults(results);
			default:	
				return BestResults(results);
			}

		}

		System.Collections.Generic.IList<Tuple<int, float>> AvgResults(System.Collections.Generic.IList<System.Collections.Generic.IList<Tuple<int, float>>> results)
		{
			int n = results[0].Count;
			var items = new Dictionary<int,float>();
			foreach (var recs in results)
			{
				foreach(var tup in recs)
				{
					if(items.ContainsKey(tup.Item1))
						items[tup.Item1] += tup.Item2;
					else
						items.Add(tup.Item1,tup.Item2);
				}
			}

			var comparer = new DelegateComparer<Tuple<int, float>>( (a, b) => a.Item2.CompareTo(b.Item2) );
			var heap = new IntervalHeap<Tuple<int, float>>(n, comparer);
			float min_score = float.MinValue;

			foreach (var item in items)
				if (item.Value/num_nodes > min_score)
				{
					heap.Add(Tuple.Create(item.Key, item.Value/num_nodes));
					if (heap.Count > n)
					{
						heap.DeleteMin();
						min_score = heap.FindMin().Item2;
					}
				}

			System.Collections.Generic.IList<Tuple<int, float>> ordered_items = new Tuple<int, float>[heap.Count];
			for (int i = 0; i < ordered_items.Count; i++)
				ordered_items[i] = heap.DeleteMax();
			return ordered_items;
		}

		System.Collections.Generic.IList<Tuple<int, float>> RankAvgResults(System.Collections.Generic.IList<System.Collections.Generic.IList<Tuple<int, float>>> results)
		{
			int n = results[0].Count;
			var items = new Dictionary<int,float>();
			foreach (var recs in results)
			{
				for (int i = 0; i < recs.Count; i++)
				{
					var tup = recs[i];
					if(items.ContainsKey(tup.Item1))
						items[tup.Item1] += (n-i);
					else
						items.Add(tup.Item1, n-i);
				}
			}

			var comparer = new DelegateComparer<Tuple<int, float>>( (a, b) => a.Item2.CompareTo(b.Item2) );
			var heap = new IntervalHeap<Tuple<int, float>>(n, comparer);
			float min_score = float.MinValue;

			foreach (var item in items)
				if (item.Value > min_score)
				{
					heap.Add(Tuple.Create(item.Key, item.Value));
					if (heap.Count > n)
					{
						heap.DeleteMin();
						min_score = heap.FindMin().Item2;
					}
				}

			System.Collections.Generic.IList<Tuple<int, float>> ordered_items = new Tuple<int, float>[heap.Count];
			for (int i = 0; i < ordered_items.Count; i++)
				ordered_items[i] = heap.DeleteMax();
			return ordered_items;
		}

		System.Collections.Generic.IList<Tuple<int, float>> CooResults(System.Collections.Generic.IList<System.Collections.Generic.IList<Tuple<int, float>>> results)
		{
			int n = results[0].Count;
			var items = new Dictionary<int,float>();
			foreach (var recs in results)
			{
				foreach(var tup in recs)
				{
					if(items.ContainsKey(tup.Item1))
						items[tup.Item1] += num_nodes + (1 - tup.Item2);
					else
						items.Add(tup.Item1,1 - tup.Item2);
				}
			}

			var comparer = new DelegateComparer<Tuple<int, float>>( (a, b) => a.Item2.CompareTo(b.Item2) );
			var heap = new IntervalHeap<Tuple<int, float>>(n, comparer);
			float min_score = float.MinValue;

			foreach (var item in items)
			{
				if (item.Value > min_score)
				{
					heap.Add(Tuple.Create(item.Key, item.Value));
					if (heap.Count > n)
					{
						heap.DeleteMin();
						min_score = heap.FindMin().Item2;
					}
				}
			}

			System.Collections.Generic.IList<Tuple<int, float>> ordered_items = new Tuple<int, float>[heap.Count];
			for (int i = 0; i < ordered_items.Count; i++) 
			{
				var tup = heap.DeleteMax();
				ordered_items[i] = Tuple.Create(tup.Item1, 
					(float) (Math.Floor(tup.Item2 / num_nodes) + 1
						+ (tup.Item2 % num_nodes) / (Math.Floor(tup.Item2 / num_nodes) + 1)));
			}
			return ordered_items;
		}

		System.Collections.Generic.IList<Tuple<int, float>> BestResults(System.Collections.Generic.IList<System.Collections.Generic.IList<Tuple<int, float>>> results)
		{
			int n = results[0].Count;
			var items = new Dictionary<int,float>();
			foreach (var recs in results)
			{
				foreach(var tup in recs)
				{
					if(items.ContainsKey(tup.Item1))
						items[tup.Item1] = Math.Max(tup.Item2, items[tup.Item1]);
					else
						items.Add(tup.Item1,tup.Item2);
				}
			}

			var comparer = new DelegateComparer<Tuple<int, float>>( (a, b) => a.Item2.CompareTo(b.Item2) );
			var heap = new IntervalHeap<Tuple<int, float>>(n, comparer);
			float min_score = float.MinValue;

			foreach (var item in items)
			{
				if (item.Value > min_score)
				{
					heap.Add(Tuple.Create(item.Key, item.Value));
					if (heap.Count > n)
					{
						heap.DeleteMin();
						min_score = heap.FindMin().Item2;
					}
				}
			}

			System.Collections.Generic.IList<Tuple<int, float>> ordered_items = new Tuple<int, float>[heap.Count];
			for (int i = 0; i < ordered_items.Count; i++) 
				ordered_items[i] = heap.DeleteMax();
			return ordered_items;
		}


		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"BaggedBPRMF num_factors={0} bias_reg={1} reg_u={2} reg_i={3} reg_j={4} num_iter={5} learn_rate={6} uniform_user_sampling={7} with_replacement={8} update_j={9} num_nodes={10} aggregation_strategy={11}",
				NumFactors, BiasReg, RegU, RegI, RegJ, NumIter, LearnRate, UniformUserSampling, WithReplacement, UpdateJ, NumNodes, AggregationStrategy);
		}


	}
}

