﻿// Copyright (C) 2013 Zeno Gantner
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
	///   User- and Item-based bagging with Incremental Stochastic Gradient Descent (BaggedUserISGD) algorithm for item prediction.
	/// </summary>
	/// <remarks>
	///   <para>
	///     Literature:
	/// 	<list type="bullet">
	///       <item><description>
	///         João Vinagre, Alípio Mário Jorge, João Gama:
	/// 		Online bagging for recommender systems
	/// 		Expert Systems, 2018
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
	public class BaggedUIISGD : IncrementalItemRecommender, IIterativeModel
	{
		/// <summary>Regularization parameter</summary>
		public double Regularization { get { return regularization; } set { regularization = value; } }
		double regularization = 0.032;

		/// <summary>Number of latent factors per user/item</summary>
		public uint NumFactors { get { return (uint) num_factors; } set { num_factors = (int) value; } }
		/// <summary>Number of latent factors per user/item</summary>
		protected int num_factors = 10;

		/// <summary>Number of iterations over the training data</summary>
		public uint NumIter { get; set; } = 30;

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

		/// <summary>Bagging type. Possible values: normal, normal_bounded, bayesian</summary>
		public string BaggingType { get { return bagging_type; } set { bagging_type = value; } }
		string bagging_type = "normal";

		/// <summary>How to combine weights of user and item.</summary>
		public string WeightCombination { get { return weight_combination; } set { weight_combination = value; } }
		string weight_combination = "minimum";

		/// <summary>Aggregation strategy to combine sub-models' predictions. Possible values: "best_score", "average", "cooccurrence"</summary>
		public string AggregationStrategy { get { return aggregation_strategy; } set { aggregation_strategy = value; } }
		string aggregation_strategy = "average";

		///
		protected MyMediaLite.Random rand;

		/// 
		protected List<ISGD> recommender_nodes;

		///
		private List<double>[] user_k, item_k;

		///
		public BaggedUIISGD ()
		{
			UpdateUsers = true;
			UpdateItems = true;
			rand = MyMediaLite.Random.GetInstance();
		}

		///
		protected virtual void InitModel()
		{
			recommender_nodes = new List<ISGD>(num_nodes);
			user_k = new List<double>[num_nodes];
			item_k = new List<double>[num_nodes];
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
				user_k[i] = new List<double>(Feedback.MaxUserID+1);
				item_k[i] = new List<double>(Feedback.MaxItemID+1);
				for (int j = 0; j <= Feedback.MaxUserID; j++)
					user_k[i].Add(Gamma.Sample(rand, 1, 1));
				for (int j = 0; j <= Feedback.MaxItemID; j++)
					item_k[i].Add(Gamma.Sample(rand, 1, 1));
			}
		}

		///
		public override void Train()
		{
			InitModel();
			Parallel.ForEach(recommender_nodes, rnode => { rnode.Train(); });
		}

		///
		public virtual void Iterate()
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
			base.AddFeedback(feedback);
			//recommender_nodes.Shuffle();
			foreach (var entry in feedback)
			{
				for (int i = 0; i < num_nodes; i++)
				{
					double weight = CombineWeights(user_k[i][entry.Item1], item_k[i][entry.Item2]);
					if(BaggingType == "bayesian")
						recommender_nodes[i].AddFeedbackRetrainW(new Tuple<int,int>[] {entry}, Math.Tanh(weight));
					else if(BaggingType == "normal_bounded")
						recommender_nodes[i].AddFeedbackRetrainN(new Tuple<int,int>[] {entry}, Convert.ToInt32(weight));
					else
						recommender_nodes[i].AddFeedbackRetrainN(new Tuple<int,int>[] {entry}, Math.Min(1, Convert.ToInt32(weight)));
				}
			}
		}

		/// <summary>
		/// Combines the weights of user and item.
		/// </summary>
		/// <returns>The combined weight.</returns>
		/// <param name="w1">user weight.</param>
		/// <param name="w2">item weight.</param>
		protected virtual double CombineWeights(double w1, double w2)
		{
			switch(weight_combination) {
			case "user_only":
				return w1;
			case "item_only":
				return w2;
			case "average":
				return (w1 + w2) / 2;
			case "maximum":
				return Math.Max(w1, w2);
			case "minimum":
			default:
				return Math.Min(w1, w2);
			}
		}


		/// 
		protected override void AddUser (int user_id)
		{
			base.AddUser(user_id);
			for (int i = 0; i < num_nodes; i++)
				user_k[i].Add(Gamma.Sample(rand, 1, 1));
		} 

		/// 
		protected override void AddItem (int item_id)
		{
			base.AddItem(item_id);
			for (int i = 0; i < num_nodes; i++)
				item_k[i].Add(Gamma.Sample(rand, 1, 1));
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

			Parallel.For(0, num_nodes, i => {
				if (user_k[i][user_id] > 0)
				{
					var rnode = recommender_nodes[i];
					var res = rnode.Recommend(user_id, n, ignore_items, candidate_items);
					if(res.Count > 0)
						lock(resultsLock) 
							results.Add(res);
				}
			});

			if(results.Count() == 0)
				return new List<Tuple<int, float>>();

			switch(aggregation_strategy) {
			case "average":
				return AvgResults(results);
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
						items[tup.Item1] += (1 - tup.Item2);
					else
						items.Add(tup.Item1,1 - tup.Item2);
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
						items[tup.Item1] = Math.Max(1 - tup.Item2, items[tup.Item1]);
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
				ordered_items[i] = heap.DeleteMax();
			return ordered_items;
		}

		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"BaggedUIISGD num_factors={0} regularization={1} learn_rate={2} num_iter={3} incr_iter={4} decay={5} num_nodes={6} aggregation_strategy={7} bagging_type={8} weight_combination={9}",
				NumFactors, Regularization, LearnRate, NumIter, IncrIter, Decay, NumNodes, AggregationStrategy, BaggingType, WeightCombination);
		}


	}
}

