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
	public class RAISGDUser : SimpleSGDZI
	{

		private IList<double> user_delta_m;
		private IList<double> user_delta_s;
		private IList<int> user_horizon;

		public RAISGDUser ()
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

			user_delta_m = new List<double>(new double[MaxUserID + 1]);
			user_delta_s = new List<double>(new double[MaxUserID + 1]);
			user_horizon = new List<int>(new int[MaxUserID + 1]);
		}

		protected override void AddUser(int userid)
		{
			base.AddUser(userid);
			user_delta_m.Add(0);
			user_delta_s.Add(0);
			user_horizon.Add(0);
		}

		private void UpdateUserStats(int u, double val)
		{
			int n = Feedback.UserMatrix[u].Count;
			if (n == 1)
			{
				user_delta_m[u] = val;
				user_delta_s[u] = 0;
			}
			else
			{
				var old_m = user_delta_m[u];
				user_delta_m[u] += (val - old_m) / n;
				user_delta_s[u] += (val - old_m) * (val - user_delta_m[u]);
			}
		}

		private double GetUserDeltaStdev(int u)
		{
			int n = Feedback.UserMatrix[u].Count;
			if(n < 2) return 0;
			return Math.Sqrt(user_delta_s[u]/(n-1));
		}

		private void UpdateUserHorizon(int user, double delta)
		{
			var stdev = GetUserDeltaStdev(user);
			if (delta > stdev)
				user_horizon[user]++;
			if(delta < stdev)
				user_horizon[user]--;
			if(user_horizon[user] < 0)
				user_horizon[user] = 0;
		}

		protected override void RetrainEntry(Tuple<int, int> entry)
		{
			var u = entry.Item1;
			var old_user_factors = user_factors.GetRow(u);
			horizon = user_horizon[u];
			base.RetrainEntry(entry);
			double delta = 0;
			for (int i = 0; i < num_factors; i++)
				delta += Math.Pow(user_factors[u, i] - old_user_factors[i], 2);
			UpdateUserHorizon(u, delta);
			UpdateUserStats(u, delta);
		}
			
	}
}

