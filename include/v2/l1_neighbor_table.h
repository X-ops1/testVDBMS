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
  //
  // 语义：
  //   - 如果 new_id 已经在 L1[v] 里，直接返回 false，不做任何事。
  //   - 否则插入，若 |L1[v]| <= B：直接 append；
  //   - 若 |L1[v]| >  B：调用 prune(v, neighbors) 做一次轻剪，
  //     要求剪完后 neighbors.size() <= B。
  //
  template<typename PruneFn>
  bool add_backlink(NodeId v, NodeId new_id, PruneFn &&prune) {
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

    if (delta.neighbors.size() > fanout_B_) {
      // 调用轻剪逻辑，要求剪到 <= B。
      prune(v, delta.neighbors);

      // 轻剪可能删掉了一些 id，这里重建一下 neighbor_set 保持一致。
      delta.neighbor_set.clear();
      for (auto id : delta.neighbors) {
        delta.neighbor_set.insert(id);
      }
    }

    // 如果之前是空的，现在非空了，记录到 active_nodes_ 里，方便后台遍历。
    if (was_empty && !delta.neighbors.empty()) {
      std::lock_guard<std::mutex> g(active_nodes_mtx_);
      active_nodes_.insert(v);
    }

    return true;
  }

  // 方便版：不传轻剪函数时，用一个非常简单的策略：超过 B 就保留最后 B 个。
  bool add_backlink(NodeId v, NodeId new_id) {
    auto default_prune = [this](NodeId /*v*/, std::vector<NodeId> &nbrs) {
      if (nbrs.size() > fanout_B_) {
        const size_t keep = fanout_B_;
        nbrs.erase(
            nbrs.begin(),
            nbrs.begin() +
                static_cast<std::ptrdiff_t>(nbrs.size() - keep));
      }
    };
    return add_backlink(v, new_id, default_prune);
  }

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
};

}  // namespace v2
