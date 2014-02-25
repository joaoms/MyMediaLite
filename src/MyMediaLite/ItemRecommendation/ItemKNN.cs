// Copyright (C) 2013 João Vinagre, Zeno Gantner
// Copyright (C) 2011, 2012 Zeno Gantner
// Copyright (C) 2010 Steffen Rendle, Zeno Gantner
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
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MyMediaLite.Correlation;
using MyMediaLite.DataType;

namespace MyMediaLite.ItemRecommendation
{
	/// <summary>k-nearest neighbor (kNN) item-based collaborative filtering</summary>
	/// <remarks>
	/// This recommender supports incremental updates for the BinaryCosine and Cooccurrence similarities.
	/// </remarks>
	public class ItemKNN : KNN, IItemSimilarityProvider
	{
		///
		protected override IBooleanMatrix DataMatrix { get { return Feedback.ItemMatrix; } }

		/// 
		protected ParallelOptions parallel_opts = new ParallelOptions() { MaxDegreeOfParallelism = 4 };

		///
		public override void Train()
		{
			base.Train();

			int num_items = MaxItemID + 1;
			if (k != uint.MaxValue)
			{
				this.nearest_neighbors = new List<IList<int>>(num_items);
				for (int i = 0; i < num_items; i++)
					nearest_neighbors.Add(correlation.GetNearestNeighbors(i, k));
			}
		}

		///
		protected override void AddItem(int item_id)
		{
			base.AddItem(item_id);
			ResizeNearestNeighbors(item_id + 1);
		}

		///
		public override float Predict(int user_id, int item_id)
		{
			if (user_id > MaxUserID)
				return float.MinValue;
			if (item_id > MaxItemID)
				return float.MinValue;

			if (k != uint.MaxValue)
			{
				double sum = 0;
				double normalization = 0;
				if (nearest_neighbors[item_id] != null)
				{
					foreach (int neighbor in nearest_neighbors[item_id])
					{
						normalization += Math.Pow(correlation[item_id, neighbor], Q);
						if (Feedback.ItemMatrix[neighbor, user_id])
							sum += Math.Pow(correlation[item_id, neighbor], Q);
					}
				}
				if (sum == 0) return 0;
				return (float) (sum / normalization);
			}
			else
			{
				// roughly 10x faster
				// TODO: implement normalization
				return (float) correlation.SumUp(item_id, Feedback.UserMatrix[user_id], Q);
			}
		}

		///
		public float GetItemSimilarity(int item_id1, int item_id2)
		{
			return correlation[item_id1, item_id2];
		}

		///
		public IList<int> GetMostSimilarItems(int item_id, uint n = 10)
		{
			if (n <= k)
				return nearest_neighbors[item_id].Take((int) n).ToArray();
			else
				return correlation.GetNearestNeighbors(item_id, n);
		}


		///
		public override void AddFeedback(ICollection<Tuple<int, int>> feedback)
		{
			base.AddFeedback(feedback);
			if (UpdateItems)
				Update(feedback);
		}

		///
		public override void RemoveFeedback(ICollection<Tuple<int, int>> feedback)
		{
			base.RemoveFeedback(feedback);
			if (UpdateItems)
				Update(feedback);
		}

		///
		protected override void RecomputeNeighbors(ICollection<int> new_items)
		{
			var retrain_items = new HashSet<int>(); 

			// Option 1 (complete): Update every neighbor list that may have changed
			/*
			float min;
			foreach (int item in Feedback.AllItems.Except(new_items))
			{
				// Get the correlation of the least correlated neighbor
				if (nearest_neighbors[item] == null)
					min = 0;
				else if (nearest_neighbors[item].Count < k)
					min = 0;
				else
					min = correlation[item, nearest_neighbors[item].Last()];

				// Check if any of the added items have a higher correlation
				// (requires retraining if it is a new neighbor or an existing one)
				foreach (int new_item in new_items)
					if (correlation[item, new_item] > min)
						retrain_items.Add(item);
			}
			*/

			// Option 2 (heuristic): Update only neighbors of added items
			/*
			foreach (int item in new_items)
			{
				if(nearest_neighbors[item] != null)
					if(nearest_neighbors[item].Count > 0)
						retrain_items.UnionWith(nearest_neighbors[item]);
			}
			*/

			// Option 3: Only update neighbor lists of added users

			// Recently added users also need retraining
			retrain_items.UnionWith(new_items);
			// Recalculate neighborhood of selected users
			Parallel.ForEach(retrain_items, parallel_opts, r_item => {
				var neighbors = correlation.GetNearestNeighbors(r_item, k);
				lock(nearest_neighbors) {
					nearest_neighbors[r_item] = neighbors;
				}
			});
		}

		/// <summary>
		/// Selectively retrains items based on new items removed from feedback.
		/// </summary>
		/// <param name='removed_items'>
		/// Removed items.
		/// </param>
		protected void RecomputeNeighborsRemoved(ICollection<int> removed_items)
		{
			var retrain_items = new HashSet<int>();
			foreach (int item in Feedback.AllItems.Except(removed_items))
				foreach (int r_item in removed_items)
					if (nearest_neighbors[item] != null)
						if (nearest_neighbors[item].Contains(r_item))
							retrain_items.Add(item);
			retrain_items.UnionWith(removed_items);
			foreach (int r_item in retrain_items)
				nearest_neighbors[r_item] = correlation.GetNearestNeighbors(r_item, k);
		}
				
	}
}
