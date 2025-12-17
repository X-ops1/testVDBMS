#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <functional>

#include "utils/tsl/robin_set.h"


namespace v2 {
// 对每个点 v 维护一个小的邻居表 L1[v]，用于存储 “new -> v” 的回边。
// 特性：
//  - |L1[v]| <= fanout_B_ （软上限）
//  - 使用 robin_set 做 O(1) 去重
//  - 支持并发：多线程 insert + 多线程 search + 后台 compaction
//  - 能快速枚举目前 L1[v] 非空的点，用于后台重构
//
// 注意：
//   这里假设构造时一个最大点数 max_nodes
//   为 [0, max_nodes) 分配 slots，这样 nodes_ 向量就不会再 resize，
//   省掉一堆并发上的复杂度。
//   DynamicSSDIndex 里可以用 base_n + max_inserts 来作为 max_nodes。

class L1NeighborTable {
 public:
  using NodeId = uint32_t;

  using MergeHook = std::function<void(NodeId, tsl::robin_set<uint32_t> *)>;
  // 注册当某个 v 的 L1[v] 达到 merge 阈值（这里用 fanout_B_）时的回调。
  // 由 SSDIndex 在 set_l1_table 时设置，典型实现是调用 SSDIndex::enqueue_merge_node(v)。
  void set_merge_hook(const MergeHook &hook) {
    merge_hook_ = hook;
  }

  // 构造函数：为 [0, num_nodes) 预分配槽位，设置软上限 B。
  L1NeighborTable(NodeId num_nodes, uint32_t fanout_B,
                  uint32_t num_lock_shards = 1u << 14)
      : fanout_B_(fanout_B),
        nodes_(num_nodes),
        lock_shards_(num_lock_shards) {}

  L1NeighborTable() = default;

  void reset(NodeId num_nodes,
             uint32_t fanout_B,
             uint32_t /*num_lock_shards*/ = 1u << 14) {
    fanout_B_ = fanout_B;

    nodes_.clear();
    nodes_.resize(num_nodes);

    std::lock_guard<std::mutex> g(active_nodes_mtx_);
    active_nodes_.clear();
  }

  inline uint32_t fanout_B() const {
    return fanout_B_;
  }

  inline NodeId num_nodes() const {
    return static_cast<NodeId>(nodes_.size());
  }
  // =============== 插入路径 ===============
  //
  // 对每个 v ∈ N_out(new_id)，调用：
  //
  //   l1->add_backlink(v, new_id, light_prune_lambda);
  
  bool add_backlink(NodeId v, NodeId new_id, tsl::robin_set<uint32_t> *deletion_set,
                    bool check_merge = true, std::vector<uint32_t> *check_list=nullptr) {

    if (v >= nodes_.size()) {
      // 越界：调用方应该保证 num_nodes 足够大。
      return false;
    }

    auto &lock = shard_lock(v);
    std::unique_lock<std::shared_timed_mutex> guard(lock);

    NodeDelta &delta = nodes_[v];

    // O(1) 去重
    auto insert_res = delta.neighbor_set.insert(new_id);
    if (!insert_res.second) {
      // 已经存在
      return false;
    }

    const bool was_empty = delta.neighbors.empty();
    delta.neighbors.emplace_back(new_id);

    // 不再在 L1 上做轻剪，只在 L0+L1 merge 时统一精剪。
    // 这里仅在 L1[v] 的长度达到 merge 阈值（默认 fanout_B_）时，
    // 触发一次 merge 回调。
    bool trigger_merge = false;
    // if (merge_hook_ && delta.neighbors.size() >= fanout_B_) {
    //   trigger_merge = true;
    // }
    if( check_merge) {
      uint32_t total = 0;
      for( auto cur_id : *check_list) {
        // if (merge_hook_ && nodes_[cur_id].neighbors.size() >= fanout_B_) {
        //   trigger_merge = true;
        //   break;
        // }
        total += nodes_[cur_id].neighbors.size();
      }
      if( total > 6 * fanout_B_) trigger_merge = true;
    }

    // 如果之前是空的，现在非空了，记录到 active_nodes_ 里，方便后台遍历。
    // if (was_empty && !delta.neighbors.empty()) {
    //   std::lock_guard<std::mutex> g(active_nodes_mtx_);
    //   active_nodes_.insert(v);
    // }

    // 释放 L1[v] 的锁，再调用外部回调，避免死锁。
    guard.unlock();

    if (trigger_merge && merge_hook_) {
      merge_hook_(v, deletion_set);
    }

    return true;
  }



  // // =============== 插入路径 ===============
  // //
  // // 对每个 v ∈ N_out(new_id)，调用：
  // //
  // //   l1->add_backlink(v, new_id, light_prune_lambda);
  // //
  // // 语义：
  // //   - 如果 new_id 已经在 L1[v] 里，直接返回 false，不做任何事。
  // //   - 否则插入，若 |L1[v]| <= B：直接 append；
  // //   - 若 |L1[v]| >  B：调用 prune(v, neighbors) 做一次轻剪，
  // //     要求剪完后 neighbors.size() <= B。
  // //
  // template<typename PruneFn>
  // bool add_backlink(NodeId v, NodeId new_id, PruneFn &&prune) {
  //   if (v >= nodes_.size()) {
  //     // 越界：调用方应该保证 num_nodes 足够大。
  //     return false;
  //   }

  //   auto &lock = shard_lock(v);
  //   std::unique_lock<std::shared_timed_mutex> guard(lock);

  //   NodeDelta &delta = nodes_[v];

  //   // O(1) 去重
  //   auto insert_res = delta.neighbor_set.insert(new_id);
  //   if (!insert_res.second) {
  //     // 已经存在
  //     return false;
  //   }

  //   const bool was_empty = delta.neighbors.empty();
  //   delta.neighbors.emplace_back(new_id);

  //   if (delta.neighbors.size() > fanout_B_) {
  //     // 调用轻剪逻辑，要求剪到 <= B。
  //     prune(v, delta.neighbors);

  //     // 轻剪可能删掉了一些 id，这里重建一下 neighbor_set 保持一致。
  //     delta.neighbor_set.clear();
  //     for (auto id : delta.neighbors) {
  //       delta.neighbor_set.insert(id);
  //     }
  //   }

  //   // 如果之前是空的，现在非空了，记录到 active_nodes_ 里，方便后台遍历。
  //   if (was_empty && !delta.neighbors.empty()) {
  //     std::lock_guard<std::mutex> g(active_nodes_mtx_);
  //     active_nodes_.insert(v);
  //   }

  //   return true;
  // }

  // =============== 搜索路径 ===============
  //
  // 把 L1[u] 复制到 out（覆盖 out）。
  void copy_neighbors(NodeId u, std::vector<NodeId> &out) const {
    out.clear();
    if (u >= nodes_.size()) {
      return;
    }

    auto &lock = shard_lock(u);
    std::shared_lock<std::shared_timed_mutex> guard(lock);
    const NodeDelta &delta = nodes_[u];
    out.insert(out.end(), delta.neighbors.begin(), delta.neighbors.end());
  }

  // 追加模式：把 L1[u] 追加到已有的 out 后面，不清 out。
  void append_neighbors(NodeId u, std::vector<NodeId> &out) const {
    if (u >= nodes_.size()) {
      return;
    }

    auto &lock = shard_lock(u);
    std::shared_lock<std::shared_timed_mutex> guard(lock);
    const NodeDelta &delta = nodes_[u];
    out.insert(out.end(), delta.neighbors.begin(), delta.neighbors.end());
  }

  // =============== 后台重构路径 ===============
  //
  // drain_neighbors：把 L1[v] 的内容搬到 out，并清空 L1[v]。
  // 常用于 compaction：
  //   L0[v] + out 作为候选，做「近似预筛 → 精剪」，然后写回 G0，最后 L1[v] 清空。
  void drain_neighbors(NodeId v, std::vector<NodeId> &out) {
    out.clear();
    if (v >= nodes_.size()) {
      return;
    }

    auto &lock = shard_lock(v);
    std::unique_lock<std::shared_timed_mutex> guard(lock);
    NodeDelta &delta = nodes_[v];

    out.swap(delta.neighbors);
    delta.neighbor_set.clear();

    if (out.empty()) {
      std::lock_guard<std::mutex> g(active_nodes_mtx_);
      active_nodes_.erase(v);
    }
  }

  // 单纯清空 L1[v]，不返回内容。
  void clear(NodeId v) {
    if (v >= nodes_.size()) {
      return;
    }
    auto &lock = shard_lock(v);
    std::unique_lock<std::shared_timed_mutex> guard(lock);
    NodeDelta &delta = nodes_[v];
    delta.neighbors.clear();
    delta.neighbor_set.clear();

    std::lock_guard<std::mutex> g(active_nodes_mtx_);
    active_nodes_.erase(v);
  }

  // 拿到当前所有「L1[v] 非空」的点的一个快照。
  // 后台 compaction 可以遍历这个列表。
  void get_active_nodes(std::vector<NodeId> &out) const {
    std::lock_guard<std::mutex> g(active_nodes_mtx_);
    out.assign(active_nodes_.begin(), active_nodes_.end());
  }

 private:
  struct NodeDelta {
    std::vector<NodeId> neighbors;       // L1[v] 的邻居列表
    tsl::robin_set<NodeId> neighbor_set; // 用于 O(1) 去重
  };

  inline std::shared_timed_mutex &shard_lock(NodeId v) const {
    return lock_shards_[v % lock_shards_.size()];
  }

  uint32_t fanout_B_{0};
  std::vector<NodeDelta> nodes_;

  // 分片锁：避免给每个点搞一个 mutex。
  mutable std::vector<std::shared_timed_mutex> lock_shards_;

  // 当前 L1[v] 非空的点的集合。
  tsl::robin_set<NodeId> active_nodes_;
  mutable std::mutex active_nodes_mtx_;

  // 当某个 v 的 L1[v] 达到 merge 阈值时触发的回调（由 SSDIndex 设置）。
  MergeHook merge_hook_;
};

}  // namespace v2
