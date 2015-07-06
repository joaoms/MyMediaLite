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
using System.Linq;
using MyMediaLite;

namespace MyMediaLite.ItemRecommendation
{
	public class RAISGDGlobalStdev : SimpleSGDZI
	{

		private double delta_m;
		private double delta_s;

		public RAISGDGlobalStdev ()
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

			delta_m = 0;
			delta_s = 0;
		}

		private void UpdateStats(double val)
		{
			int n = Feedback.Count;
			if (n == 1)
			{
				delta_m = val;
				delta_s = 0;
			}
			else
			{
				var old_m = delta_m;
				delta_m += (val - old_m) / n;
				delta_s += (val - old_m) * (val - delta_m);
			}
		}

		private double GetDeltaStdev()
		{
			int n = Feedback.Count;
			if(n < 2) return 0;
			return Math.Sqrt(delta_s/(n-1));
		}

		private void UpdateHorizon(double delta)
		{
			var stdev = GetDeltaStdev();
			if (delta > stdev)
				horizon++;
			if(delta < stdev)
				horizon--;
			if(horizon < 0)
				horizon = 0;
		}

		protected override void RetrainEntry(Tuple<int, int> entry)
		{
			double delta = 0;
			var u = entry.Item1;
			var old_user_factors = user_factors.GetRow(u);
			for (int i = 0; i < IncrIter; i++)
				UpdateFactors(entry.Item1, entry.Item2, UpdateUsers, UpdateItems, 1);
			for (int i = 0; i < num_factors; i++)
				delta += Math.Pow(user_factors[u, i] - old_user_factors[i], 2);
			UpdateHorizon(delta);
			UpdateStats(delta);
			user_factors.SetRow(u, old_user_factors);
			base.RetrainEntry(entry);
		}

	}
}

