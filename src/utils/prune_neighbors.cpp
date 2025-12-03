#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "utils/timer.h"
#include "utils/tsl/robin_map.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::occlude_list(std::vector<Neighbor> &pool, const tsl::robin_map<uint32_t, T *> &coord_map,
                                       std::vector<Neighbor> &result, std::vector<float> &occlude_factor) {
    if (pool.empty())
      return;

    std::set<Neighbor> result_set;  // deduplication.
    float cur_alpha = 1;
    while (cur_alpha <= alpha && result_set.size() < range) {
      uint32_t start = 0;
      while (result_set.size() < range && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result_set.insert(p);
        for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          auto iter_right = coord_map.find(p.id);
          auto iter_left = coord_map.find(pool[t].id);
          float djk = this->dist_cmp->compare(iter_left->second, iter_right->second, this->data_dim);
          occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2f;
    }
    for (auto &x : result_set) {
      result.push_back(x);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::occlude_list_pq(std::vector<Neighbor> &pool, std::vector<Neighbor> &result,
                                          std::vector<float> &occlude_factor, uint8_t *scratch) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    std::set<Neighbor> result_set;  // deduplication, and keep distance sorted.
    float cur_alpha = 1;
    while (cur_alpha <= alpha && result_set.size() < range) {
      uint32_t start = 0;
      while (result_set.size() < range && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result_set.insert(p);
        // dynamic programming, if p (current) is included,
        // then D(t, p0) / D(t, p) should be updated.
        for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          // djk = dist(p.id, pool[t.id])
          float djk;
          nbr_handler->compute_dists(p.id, &(pool[t].id), 1, &djk, scratch);
          // LOG(INFO) << pool[t].distance << " " << djk << " " << alpha << " " << result_set.size();
          occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2f;
    }
    for (auto &x : result_set) {
      result.push_back(x);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::prune_neighbors_pq(std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list,
                                             uint8_t *scratch) {
    if (pool.size() == 0)
      return;

    // sort the pool based on distance to query

    std::vector<Neighbor> result;
    result.reserve(this->range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list_pq(pool, result, occlude_factor, scratch);

    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      pruned_list.emplace_back(iter.id);
    }

    if (alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if (std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) == pruned_list.end())
          pruned_list.emplace_back(pool[i].id);
      }
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::delta_prune_neighbors_pq(std::vector<TriangleNeighbor> &pool,
                                                   std::vector<uint32_t> &pruned_list, uint8_t *scratch, int tgt_idx) {
    if (unlikely(pool.size() != this->range + 1)) {
      LOG(ERROR) << "Pool size " << pool.size() << " not equal to " << this->range + 1;
    }
    pruned_list.clear();
    float cur_alpha = alpha;
    int to_evict = -1;
    float tgt_nbr_dis = pool[tgt_idx].distance;
    // step 1: fast path
    // determine which to evict using triangular inequality.
    while (cur_alpha >= (1 - 1e-5) && to_evict == -1) {
      for (int i = (int) pool.size() - 1; i >= 0; --i) {
        if (i == tgt_idx) {
          continue;
        }
        if (pool[i].distance > tgt_nbr_dis) {
          // pool[i] -> nbr is the longest edge.
          if (pool[i].distance / pool[i].tgt_dis > cur_alpha) {
            to_evict = i;
            break;
          }
        } else {
          // tgt -> nbr is the longest edge.
          if (tgt_nbr_dis / pool[i].tgt_dis > cur_alpha) {
            to_evict = tgt_idx;
            break;
          }
        }
      }
      cur_alpha /= 1.2f;
    }

    auto finish = [&]() {
      for (int i = 0; i < (int) pool.size(); i++) {
        if (i == to_evict) {
          continue;
        }
        pruned_list.emplace_back(pool[i].id);
      }
    };

    if (to_evict != -1) {
      finish();
      return;
    }
    // The point to insert is with high quality.
    // Step 2: Seek one with low quality to evict, early stop.

    std::vector<uint32_t> ids(pool.size());
    for (uint32_t i = 0; i < pool.size(); i++) {
      ids[i] = pool[i].id;
    }
    std::vector<float> dists(pool.size());

    for (int start = 0; start < (int) pool.size(); ++start) {
      if (start == tgt_idx) {
        continue;
      }
      auto &p = pool[start];
      nbr_handler->compute_dists(p.id, ids.data() + start + 1, pool.size() - start - 1, dists.data() + start + 1,
                                 scratch);
      for (uint32_t t = start + 1; t < pool.size(); t++) {
        if (pool[t].distance / dists[t] > alpha) {
          to_evict = t;
          finish();
          return;
        }
      }
    }

    // Step 3: all the points satisfy alpha-RNG, evict the farthest.
    to_evict = pool.size() - 1;
    finish();
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::prune_l1_delta(uint32_t center,
                                        uint32_t new_id,
                                        std::vector<uint32_t> &nbrs,
                                        uint8_t *scratch) {
    // 约定：L1 的软上限 B == this->range
    // L1::add_backlink 保证：超过 B 时立刻调用 prune，
    // 所以这里 nbrs.size() 应该是 range + 1。
    if (nbrs.size() <= this->range)
      return;

    if (nbrs.size() != this->range + 1) {
      LOG(WARNING) << "L1 prune: neighbors size " << nbrs.size()
                  << " != range + 1 (" << this->range + 1 << ")";
      return;
    }

    const size_t pool_size = nbrs.size();

    // 1) 计算 d(new_id, nbrs[k]) 和 d(center, nbrs[k]) （用 PQ 近似）
    std::vector<float> tgt_dists(pool_size), nbr_dists(pool_size);
    // new_id -> neighbors
    nbr_handler->compute_dists(new_id,
                              nbrs.data(),
                              pool_size,
                              tgt_dists.data(),
                              scratch);
    // center(v) -> neighbors
    nbr_handler->compute_dists(center,
                              nbrs.data(),
                              pool_size,
                              nbr_dists.data(),
                              scratch);

    // 2) 组装 TriangleNeighbor 池
    std::vector<TriangleNeighbor> tri_pool(pool_size);
    for (size_t k = 0; k < pool_size; ++k) {
      tri_pool[k].id       = nbrs[k];
      tri_pool[k].tgt_dis  = tgt_dists[k];  // new_id -> nbr_k
      tri_pool[k].distance = nbr_dists[k];  // center  -> nbr_k
    }
    std::sort(tri_pool.begin(), tri_pool.end());  // 按 distance 升序

    // 3) 找到 new_id 在 tri_pool 里的下标
    int tgt_idx = -1;
    for (int k = 0; k < (int) pool_size; ++k) {
      if (tri_pool[k].id == new_id) {
        tgt_idx = k;
        break;
      }
    }
    if (unlikely(tgt_idx == -1)) {
      LOG(ERROR) << "L1 prune: new_id " << new_id
                << " not found in tri_pool for center " << center;
      return;
    }

    // 4) 调用原来的 delta_prune_neighbors_pq 做
    //    “三角预筛 + occlusion”，输出保留的 id 列表
    std::vector<uint32_t> pruned_ids;
    this->delta_prune_neighbors_pq(tri_pool, pruned_ids, scratch, tgt_idx);

    // 5) 回写到 nbrs
    nbrs.assign(pruned_ids.begin(), pruned_ids.end());
  }


  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::prune_neighbors(const tsl::robin_map<uint32_t, T *> &coord_map, std::vector<Neighbor> &pool,
                                          std::vector<uint32_t> &pruned_list) {
    if (pool.size() == 0)
      return;

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list(pool, coord_map, result, occlude_factor);

    pruned_list.clear();

    // SPACEV1B frequently inserts medoid, which can not be excluded by triangular ineq.
    size_t medoid_threshold = result.size() * 3 / 4;
    for (size_t i = 0; i < result.size(); ++i) {
      if (i > medoid_threshold && result[i].id == medoid) {
        continue;
      }
      pruned_list.emplace_back(result[i].id);
    }

    if (alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if (std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) == pruned_list.end()) {
          pruned_list.emplace_back(pool[i].id);
        }
      }
    }
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
