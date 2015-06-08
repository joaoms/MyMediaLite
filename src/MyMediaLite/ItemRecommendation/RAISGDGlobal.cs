// Copyright (C) 2012 Zeno Gantner
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
using System.Collections.Generic;
using MyMediaLite;

namespace MyMediaLite.ItemRecommendation
{
	public class RAISGDGlobal : SimpleSGDZI
	{
		public int SampleSize { get { return sample_size; } set { sample_size = value; } }
		int sample_size = 100;

		public float GlobalMean { get { return global_mean; } set { global_mean = value; }}
		float global_mean = 0f;

		private IList<int> sample_users;

		public RAISGDGlobal ()
		{
			ForgetUsersInUsers = false;
			ForgetUsersInItems = false;
			ForgetItemsInUsers = true;
			ForgetItemsInItems = false;
			Horizon = 0;
		}

		protected override void InitModel()
		{
			base.InitModel();

			if(MaxUserID < sample_size) sample_size = MaxUserID;
			sample_users = new List<int>(sample_size);
			while (sample_users.Count < sample_size)
			{
				var u = MyMediaLite.Random.GetInstance().Next(MaxUserID);
				if (!sample_users.Contains(u))
					sample_users.Add(u);
			}
		}


		public override void Train()
		{
			base.Train();
			CalculateGlobalMean();
		}

		private void CalculateGlobalMean()
		{
			float sum = 0f;
			int i = 0;
			foreach (var u in sample_users)
				for(; i < Feedback.MaxItemID; i++)
					sum += Predict(u, i);
			global_mean = sum / (sample_size * i);
		}


		private void UpdateHorizon()
		{
			CalculateGlobalMean();
			if(global_mean < 0.5)
				horizon--;
			if(global_mean > 0.5)
				horizon++;
			if(horizon < 0)
				horizon = 0;
		}
			
		protected override void Retrain(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			UpdateHorizon();
			base.Retrain(feedback);
		}
	
	}
}

