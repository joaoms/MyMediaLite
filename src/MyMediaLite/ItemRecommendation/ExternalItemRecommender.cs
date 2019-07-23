// Copyright (C) 2012, 2019 Zeno Gantner
// Copyright (C) 2019 Petr Sobotka
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
using System.Collections.Generic;
using MyMediaLite.Data;
using MyMediaLite.IO;

namespace MyMediaLite.ItemRecommendation
{
	/// <summary>Uses externally computed predictions</summary>
	/// <remarks>
	/// <para>
	///   This recommender is for loading predictions made by external (non-MyMediaLite) recommenders,
	///   so that we can use MyMediaLite's evaluation framework to evaluate their accuracy.
	/// </para>
	/// <para>
	///   This recommender does NOT support incremental updates.
	/// </para>
	/// </remarks>
	public class ExternalItemRecommender : ItemRecommender, INeedsMappings
	{
		/// <summary>the file with the stored ratings</summary>
		public string PredictionFile { get; set; }

		///
		public IMapping UserMapping { get; set; }
		///
		public IMapping ItemMapping { get; set; }

		private Ratings external_scores;

		/// <summary>Default constructor</summary>
		public ExternalItemRecommender()
		{
			PredictionFile = "FILENAME";
		}

		///
		public override void Train()
		{
			external_scores = (Ratings) RatingData.Read(PredictionFile, UserMapping, ItemMapping);
			// we dont use it locally but it should be initialized before entering the parallel code
			IList<IList<int>> nonsense = external_scores.ByUser;
		}

		///
		public override bool CanPredict(int user_id, int item_id)
		{
			foreach (int index in external_scores.ByUser[user_id])
				if (external_scores.Items[index] == item_id)
					return true;
			return false;
		}

		///
		public override float Predict(int user_id, int item_id)
		{
			foreach (int index in external_scores.ByUser[user_id])
				if (external_scores.Items[index] == item_id)
					return external_scores[index];
           	return float.MinValue;
		}

		///
		public override void SaveModel(string filename) { /* do nothing */ }

		///
		public override void LoadModel(string filename) { /* do nothing */ }

		///
		public override string ToString()
		{
			return string.Format(
				"{0} prediction_file={1}",
				this.GetType().Name, PredictionFile);
		}
	}
}
