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
using C5;
using MyMediaLite.DataType;
using System.Linq;

namespace MyMediaLite.ItemRecommendation
{
	public class BPRMFRecency : BPRMF
	{
		protected HashedLinkedList<int> item_queue;

		protected override void InitModel()
		{
			base.InitModel();
			item_queue = new HashedLinkedList<int>();
			item_queue.AddAll(Feedback.Items);
		}

		/// <summary>Sample a pair of items, given a user</summary>
		/// <param name="user_id">the user ID</param>
		/// <param name="item_id">the ID of the first item</param>
		/// <param name="other_item_id">the ID of the second item</param>
		protected override void SampleItemPair(int user_id, out int item_id, out int other_item_id)
		{
			var user_items = Feedback.UserMatrix[user_id];
			item_id = user_items.ElementAt(random.Next(user_items.Count));
			do
			{
				other_item_id = item_queue.RemoveFirst();
				item_queue.InsertLast(other_item_id);
			}
			while (user_items.Contains(other_item_id));
		}

		void Retrain(ICollection<Tuple<int, int>> feedback)
		{
			var users = from t in feedback select t.Item1;
			var items = from t in feedback select t.Item2;

			if (UpdateUsers)
				foreach (int user_id in users)
					RetrainUser(user_id);

			foreach (int item_id in items)
			{
				if (UpdateItems)
					RetrainItem(item_id);
				item_queue.Remove(item_id);
				item_queue.InsertLast(item_id);
			}
		}

	}
}

