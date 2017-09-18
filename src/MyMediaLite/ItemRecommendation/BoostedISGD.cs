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
using MathNet.Numerics.Distributions;

namespace MyMediaLite.ItemRecommendation
{
	/// <summary>
	///   Boosting with Incremental Stochastic Gradient Descent (BoostedISGD) algorithm for item prediction. 
	/// 	This version uses online AdaBoost based on recall@N (hit or miss) obtained by each weak model.
	/// </summary>
	/// <remarks>
	///   <para>
	///     Literature:
	/// 	<list type="bullet">
	///       <item><description>
	///         João Vinagre, Alípio Mário Jorge, João Gama:
	///         
	///       </description></item>
	///     </list>
	///   </para>
	///   <para>
	///       This algorithm is primarily designed to use with incremental learning, 
	/// 		batch behavior has not been thoroughly studied.
	///   </para> 
	///   <para>
	///     This algorithm supports (and encourages) incremental updates. 
	///   </para>
	/// </remarks>
	public class BoostedISGD : MF
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
		bool use_multicore = true;

		/// <summary>Number of bootstrap nodes.</summary>
		public int NumNodes { get { return num_nodes; } set { num_nodes = value; } }
		int num_nodes = 4;

		/// <summary>Cutoff (relevance threshold). The size of recommended items' list to check for observed item.</summary>
		public int Cutoff { get { return cutoff; } set { cutoff = value; } }
		int cutoff = 10;


		///
		protected MyMediaLite.Random rand;
		
		/// 
		protected List<ISGD> recommender_nodes;

		///
		protected List<float> hit_rate;

		///
		protected List<float> miss_rate;

		/// 
		protected List<double> weight;

		///
		public BoostedISGD ()
		{
			UpdateUsers = true;
			UpdateItems = true;
			rand = MyMediaLite.Random.GetInstance();
		}

		///
		protected override void InitModel()
		{
			recommender_nodes = new List<ISGD>(num_nodes);
			hit_rate = new List<float>(num_nodes);
			miss_rate = new List<float>(num_nodes);
			weight = new List<double>(num_nodes);
			ISGD recommender_node;
			for (int i = 0; i < num_nodes; i++) {
				recommender_node = new ISGD();
				recommender_node.UpdateUsers = true;
				recommender_node.UpdateItems = true;
				recommender_node.Regularization = Regularization;
				recommender_node.NumFactors = NumFactors;
				recommender_node.LearnRate = LearnRate;
				recommender_node.IncrIter = IncrIter;
				recommender_node.NumIter = NumIter;
				recommender_node.Decay = Decay;
				recommender_node.Feedback = Feedback;
				recommender_nodes.Add(recommender_node);
				hit_rate.Add(0);
				miss_rate.Add(0);
				weight.Add(0);
			}
		}

		///
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

		///
		public override float Predict(int user_id, int item_id)
		{
			return Predict(user_id, item_id, false);
		}

		///
		protected virtual float Predict(int user_id, int item_id, bool bound)
		{
			if (user_id > recommender_nodes[0].MaxUserID || item_id >= recommender_nodes[0].MaxItemID)
				return float.MinValue;

			List<float> results = new List<float>(num_nodes);
			foreach (var rnode in recommender_nodes)
				results.Add(rnode.Predict(user_id, item_id));
			
			float result = results.Average();

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
			float lambda;
			base.AddFeedback(feedback);
			foreach (var entry in feedback)
			{
				lambda = 1;
				for (int i = 0; i < num_nodes; i++)
				{
					var rnode = recommender_nodes[i];
					bool hit = false;
					float err;
					int npoisson = Poisson.Sample(rand, lambda);
					rnode.AddFeedbackRetrainN(new Tuple<int, int>[] {entry}, npoisson);
					var recs = rnode.Recommend(entry.Item1, cutoff);
					foreach (var rec in recs)
						if (hit = (rec.Item1 == entry.Item2))
							break;
					if(hit)
					{
						hit_rate[i] += lambda;
						err = miss_rate[i] / (hit_rate[i] + miss_rate[i]);
						lambda = lambda / (2 * (1 - err));
					}
					else
					{
						miss_rate[i] += lambda;
						err = miss_rate[i] / (hit_rate[i] + miss_rate[i]);
						lambda = lambda / (2 * err);
					}
					weight[i] = Math.Log((1 - err) / err);
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
			var resultsLock = new object ();
			var results = new List<System.Collections.Generic.IList<Tuple<int,float>>>(num_nodes);

			Parallel.ForEach(recommender_nodes, rnode => {
				var res = rnode.Recommend(user_id, n, ignore_items, candidate_items);
				lock(resultsLock) results.Add(res);
			});

			return AggregateResults(results);

		}

		System.Collections.Generic.IList<Tuple<int, float>> AggregateResults(System.Collections.Generic.IList<System.Collections.Generic.IList<Tuple<int, float>>> results)
		{
			int n = results[0].Count;
			var items = new Dictionary<int,float>();
			for (int i = 0; i < num_nodes; i++)
			{
				var recs = results[i];
				foreach(var tup in recs)
				{
					if(items.ContainsKey(tup.Item1))
						items[tup.Item1] += (float) (weight[i] * (1 - tup.Item2));
					else
						items.Add(tup.Item1, (float) (weight[i] * (1 - tup.Item2)));
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

		/// 
		protected override void RetrainUser(int user_id)
		{
		}

		///
		protected override void RetrainItem(int item_id)
		{
		}

		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"BoostedISGD num_factors={0} regularization={1} learn_rate={2} num_iter={3} incr_iter={4} decay={5} num_nodes={6} aggregation_strategy={7}",
				NumFactors, Regularization, LearnRate, NumIter, IncrIter, Decay, NumNodes);
		}


	}
}

