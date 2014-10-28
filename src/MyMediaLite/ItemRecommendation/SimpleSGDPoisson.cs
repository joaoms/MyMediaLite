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

namespace MyMediaLite.ItemRecommendation
{
	public class SimpleSGDPoisson : SimpleSGD
	{

		public bool ForgetUsersInUsers { get; set; }
		public bool ForgetItemsInUsers { get; set; }
		public bool ForgetUsersInItems { get; set; }
		public bool ForgetItemsInItems { get; set; }
		public int ForgetHorizon { get { return forget_horizon; } set { forget_horizon = value; } }
		public float Alpha { get { return alpha; } set { alpha = value; } }
		int forget_horizon = 1;
		float alpha = 0.001f;

		private bool forget_users;
		private bool forget_items;

		BinaryHeap item_queue;
		BinaryHeap user_queue;

		// float max_score = 1.0f;

		public SimpleSGDPoisson()
		{
			ForgetUsersInUsers = false;
			ForgetUsersInItems = false;
			ForgetItemsInUsers = true;
			ForgetItemsInItems = false;
		}

		protected override void InitModel()
		{
			base.InitModel();


			forget_users = ForgetUsersInUsers || ForgetUsersInItems;
			forget_items = ForgetItemsInUsers || ForgetItemsInItems;

			if (forget_users)
			{
				user_queue = new BinaryHeap();
				foreach (int usr in Feedback.Users)
					user_queue.AddOrUpdate(usr, Alpha);
			}
			if (forget_items)
			{
				item_queue = new BinaryHeap();
				foreach (int item in Feedback.Items)
					item_queue.AddOrUpdate(item, Alpha);
			}
		}

		protected override void Retrain(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			BHNode[] qu = new BHNode[Math.Min(ForgetHorizon, user_queue.Count - 1)];
			BHNode[] qi = new BHNode[Math.Min(ForgetHorizon, item_queue.Count - 1)];
			foreach (var entry in feedback)
			{
				if (forget_users)
				{
					for (uint i = 0; i < qu.Length; i++)
						qu[i] = user_queue.Remove();
				}
				if (forget_items)
				{
					for (uint i = 0; i < qi.Length; i++)
						qi[i] = item_queue.Remove();
				}
				//Console.WriteLine("Forgetting item "+qi);
				if (forget_users)
					foreach (var usr in qu.Reverse())
						if (usr.Idx >= 0 && usr.Idx != entry.Item1) 
							for (uint i = 0; i < IncrIter; i++)
								UpdateFactors(usr.Idx, entry.Item2, ForgetUsersInUsers, ForgetUsersInItems, 0);
				if (forget_items)
					foreach (var itm in qi.Reverse())
						if (itm.Idx >= 0 && itm.Idx != entry.Item2)
							for (uint i = 0; i < IncrIter; i++)
								UpdateFactors(entry.Item1, itm.Idx, ForgetItemsInUsers, ForgetItemsInItems, 0);
				for (uint i = 0; i < IncrIter; i++)
					UpdateFactors(entry.Item1, entry.Item2, UpdateUsers, UpdateItems, 1);

				if (forget_items)
				{
					item_queue.AddOrUpdate(entry.Item2, Alpha);

					foreach (var itm in qi.Reverse())
						if (itm.Idx != entry.Item2)
						{
							itm.Lambda = alpha + ((1-alpha)*itm.Lambda);
							item_queue.Add(itm);
						}
				}
				if (forget_users)
				{
					user_queue.AddOrUpdate(entry.Item1, Alpha);

					foreach (var usr in qu.Reverse())
						if (usr.Idx != entry.Item1)
						{
							usr.Lambda = alpha + ((1-alpha)*usr.Lambda);
							user_queue.Add(usr);
						}
				}
			}
		}

		///
		public override void RemoveItem(int item_id)
		{
			base.RemoveItem(item_id);

			if (forget_items)
				item_queue.Remove(item_id);
		}

		///
		public override void RemoveUser(int user_id)
		{
			base.RemoveUser(user_id);

			if (forget_users)
				user_queue.Remove(user_id);
		}

		/// <summary>
		/// Performs factor updates for a user and item pair.
		/// </summary>
		/// <param name="user_id">User_id.</param>
		/// <param name="item_id">Item_id.</param>
		protected virtual void UpdateFactors(int user_id, int item_id, bool update_user, bool update_item, float base_val = 1)
		{
			//Console.WriteLine(float.MinValue);
			float err = base_val - Predict(user_id, item_id, false);

			// adjust factors
			for (int f = 0; f < NumFactors; f++)
			{
				float u_f = user_factors[user_id, f];
				float i_f = item_factors[item_id, f];
				
				// if necessary, compute and apply updates
				if (update_user)
				{
					double delta_u = err * i_f - Regularization * u_f;
					user_factors.Inc(user_id, f, current_learnrate * delta_u);
				}
				if (update_item)
				{
					double delta_i = err * u_f - Regularization * i_f;
					item_factors.Inc(item_id, f, current_learnrate * delta_i);
				}
			}

		}


		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"NaiveSVDZI num_factors={0} regularization={1} learn_rate={2} num_iter={3} incr_iter={4} decay={5}, forget_horizon={6}",
				NumFactors, Regularization, LearnRate, NumIter, IncrIter, Decay, ForgetHorizon);
		}


	}

	class BHNode : IComparable<BHNode>
	{
		private int idx;
		private double priority;
		public int Idx { get { return idx; } set { idx = value; } }
		public double Lambda { get { return priority; } set { priority = value; } }
		
		public BHNode(int in_idx, double lambda)
		{
			Idx = in_idx;
			Lambda = lambda;
		}

		public BHNode(int in_idx)
		{
			Idx = in_idx;
			Lambda = 0;
		}

		public int CompareTo(BHNode other)
		{
			return this.Idx - other.Idx;
		}

	}

	/// <summary>
	/// A binary heap, useful for sorting data and priority queues.
	/// </summary>
	class BinaryHeap : IEnumerable<BHNode>
	{
		// Constants
		private const int DEFAULT_SIZE = 1000;
		// Fields
		private BHNode[] _data = new BHNode[DEFAULT_SIZE];
		private int _count = 0;
		private int _capacity = DEFAULT_SIZE;
		private bool _sorted;

		// Properties
		/// <summary>
		/// Gets the number of values in the heap. 
		/// </summary>
		public int Count
		{
			get { return _count; }
		}
		/// <summary>
		/// Gets or sets the capacity of the heap.
		/// </summary>
		public int Capacity
		{
			get { return _capacity; }
			set
			{
				int previousCapacity = _capacity;
				_capacity = Math.Max(value, _count);
				if (_capacity != previousCapacity)
				{
					BHNode[] temp = new BHNode[_capacity];
					Array.Copy(_data, temp, _count);
					_data = temp;
				}
			}
		}
		// Methods
		/// <summary>
		/// Creates a new binary heap.
		/// </summary>
		public BinaryHeap()
		{
		}
		private BinaryHeap(BHNode[] data, int count)
		{
			Capacity = count;
			_count = count;
			Array.Copy(data, _data, count);
		}
		/// <summary>
		/// Gets the first value in the heap without removing it.
		/// </summary>
		/// <returns>The lowest value of type TValue.</returns>
		public BHNode Peek()
		{
			return _data[0];
		}

		/// <summary>
		/// Removes all items from the heap.
		/// </summary>
		public void Clear()
		{
			this._count = 0;
			_data = new BHNode[_capacity];
		}
		/// <summary>
		/// Adds a key and value to the heap.
		/// </summary>
		/// <param name="item">The item to add to the heap.</param>
		public void Add(BHNode item)
		{
			if (_count == _capacity)
			{
				Capacity *= 2;
			}
			_data[_count] = item;
			UpHeap();
			_count++;
		}

		public void AddOrUpdate(int idx, float alpha)
		{
			double z0 = 0;
			var removed = Remove(idx);
			if (removed != null)
				z0 = removed.Lambda;
			Scale(alpha);
			Add(new BHNode(idx, ((1-alpha)*z0) + alpha));
		}
			
		/// <summary>
		/// Removes and returns the first item in the heap.
		/// </summary>
		/// <returns>The next value in the heap.</returns>
		public BHNode Remove()
		{
			if (this._count == 0)
			{
				throw new InvalidOperationException("Cannot remove item, heap is empty.");
			}
			BHNode v = _data[0];
			_count--;
			_data[0] = _data[_count];
			_data[_count] = null; //Clears the Last Node
			DownHeap();
			return v;
		}
		private void UpHeap()
		//helper function that performs up-heap bubbling
		{
			_sorted = false;
			int p = _count;
			BHNode item = _data[p];
			int par = Parent(p);
			while (par > -1 && item.Lambda < _data[par].Lambda)
			{
				_data[p] = _data[par]; //Swap nodes
				p = par;
				par = Parent(p);
			}
			_data[p] = item;
		}
		private void DownHeap()
		//helper function that performs down-heap bubbling
		{
			_sorted = false;
			int n;
			int p = 0;
			BHNode item = _data[p];
			while (true)
			{
				int ch1 = Child1(p);
				if (ch1 >= _count) break;
				int ch2 = Child2(p);
				if (ch2 >= _count)
				{
					n = ch1;
				}
				else
				{
					n = _data[ch1].Lambda < _data[ch2].Lambda ? ch1 : ch2;
				}
				if (item.Lambda > _data[n].Lambda)
				{
					_data[p] = _data[n]; //Swap nodes
					p = n;
				}
				else
				{
					break;
				}
			}
			_data[p] = item;
		}
		private void EnsureSort()
		{
			if (_sorted) return;
			Array.Sort(_data, 0, _count);
			_sorted = true;
		}
		private static int Parent(int index)
		//helper function that calculates the parent of a node
		{
			return (index - 1) >> 1;
		}
		private static int Child1(int index)
		//helper function that calculates the first child of a node
		{
			return (index << 1) + 1;
		}
		private static int Child2(int index)
		//helper function that calculates the second child of a node
		{
			return (index << 1) + 2;
		}

		/// <summary>
		/// Creates a new instance of an identical binary heap.
		/// </summary>
		/// <returns>A BinaryHeap.</returns>
		public BinaryHeap Copy()
		{
			return new BinaryHeap(_data, _count);
		}

		/// <summary>
		/// Gets an enumerator for the binary heap.
		/// </summary>
		/// <returns>An IEnumerator of type T.</returns>
		public IEnumerator<BHNode> GetEnumerator()
		{
			EnsureSort();
			for (int i = 0; i < _count; i++)
			{
				yield return _data[i];
			}
		}
		System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
		{
			return GetEnumerator();
		}

		/// <summary>
		/// Checks to see if the binary heap contains the specified item.
		/// </summary>
		/// <param name="item">The item to search the binary heap for.</param>
		/// <returns>A boolean, true if binary heap contains item.</returns>
		public bool Contains(BHNode item)
		{
			EnsureSort();
			return Array.BinarySearch<BHNode>(_data, 0, _count, item) >= 0;
		}

		public bool Contains(int idx)
		{
			return Contains(new BHNode(idx));
		}

		/// <summary>
		/// Copies the binary heap to an array at the specified index.
		/// </summary>
		/// <param name="array">One dimensional array that is the destination of the copied elements.</param>
		/// <param name="arrayIndex">The zero-based index at which copying begins.</param>
		public void CopyTo(BHNode[] array, int arrayIndex)
		{
			EnsureSort();
			Array.Copy(_data, array, _count);
		}
		/// <summary>
		/// Gets whether or not the binary heap is readonly.
		/// </summary>
		public bool IsReadOnly
		{
			get { return false; }
		}
		/// <summary>
		/// Removes an item from the binary heap. This utilizes the type T's Comparer and will not remove duplicates.
		/// </summary>
		/// <param name="item">The item to be removed.</param>
		/// <returns>The removed item or null.</returns>
		public BHNode Remove(BHNode item)
		{
			EnsureSort();
			int i = Array.BinarySearch<BHNode>(_data, 0, _count, item);
			if (i < 0) return null;
			BHNode removed = _data[i];
			Array.Copy(_data, i + 1, _data, i, _count - i);
			_data[_count] = null;
			_count--;
			return removed;
		}

		public BHNode Remove(int idx)
		{
			return Remove(new BHNode(idx));
		}

		public void Scale(float alpha)
		{
			if (alpha <= 0) throw new Exception("only order-preserving scaling is allowed");
			for (int i = 0; i < _count; i++)
			{
				_data[i].Lambda *= (1 - alpha);
			}
		}

	}
}

