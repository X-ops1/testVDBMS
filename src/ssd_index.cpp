#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <malloc.h>
#include <filesystem>

#include <omp.h>
#include <cmath>
#include "nbr/abstract_nbr.h"
#include "ssd_index_defs.h"
#include "utils/timer.h"
#include "utils.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "utils/tsl/robin_set.h"

#include <algorithm>
#include <map>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
    this->coords = coords;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }

  // structs for DiskNode
  template struct DiskNode<float>;
  template struct DiskNode<uint8_t>;
  template struct DiskNode<int8_t>;

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &fileReader,
                              AbstractNeighbor<T> *nbr_handler, bool tags, Parameters *params)
      : reader(fileReader), nbr_handler(nbr_handler), data_is_normalized(false), enable_tags(tags) {
    if (m == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                     "Changing distance to L2 to boost accuracy.";
        m = pipeann::Metric::L2;
        data_is_normalized = true;

      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " This may result in erroneous results or poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    this->dist_cmp.reset(pipeann::get_distance_function<T>(m));

    if (params != nullptr) {
      this->beam_width = params->beam_width;
      this->l_index = params->L;
      this->range = params->R;
      this->maxc = params->C;
      this->alpha = params->alpha;
      LOG(INFO) << "Beamwidth: " << this->beam_width << ", L: " << this->l_index << ", R: " << this->range
                << ", C: " << this->maxc;
    }
    LOG(INFO) << "Use " << nbr_handler->get_name() << " as neighbor handler.";
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::~SSDIndex() {
    LOG(INFO) << "Lock table size: " << this->idx_lock_table.size();
    LOG(INFO) << "Page cache size: " << v2::cache.cache.size();

    if (load_flag) {
      this->destroy_buffers();
      reader->close();
    }
  }

  // --- merge 队列的简单封装 ---
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::push(uint32_t id) {
    // 简单过滤不把空值推到队列里
    if (id == MERGE_NULL_ID) {
      return;
    }
    merge_queue_.push(id);
    // 唤醒在 pop_blocking 中等待的线程
    merge_queue_.push_notify_all();
  }

  template<typename T, typename TagT>
  bool SSDIndex<T, TagT>::try_pop(uint32_t &id) {
    // ConcurrentQueue 的 pop 是非阻塞的，队列空时会返回 null_T
    id = merge_queue_.pop();
    if (id == merge_queue_.null_T) {
      return false;
    }
    return true;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::pop_blocking(uint32_t &id) {
    // 先尝试一次 pop，如果是空值，就进入等待
    id = merge_queue_.pop();
    while (id == merge_queue_.null_T) {
      // 没有新的任务，等待 push 通知
      merge_queue_.wait_for_push_notify();
      id = merge_queue_.pop();
    }
  }

  // --- L1 增量图合并线程：从队列取节点，按扇区分组，然后调用 merge_nodes_on_sector ---
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_worker_thread() {
    LOG(INFO) << "SSDIndex merge worker thread started.";

    // 为当前线程创建 I/O 上下文（io_uring / AIO）
    void *ctx = reader->get_ctx();

    // 单次批处理最多从队列拉这么多点，可以后面再调参
    constexpr size_t kMergeBatchSize = 10000;

    std::vector<uint32_t> batch;
    batch.reserve(kMergeBatchSize);

    auto timer = pipeann::Timer();

    while (true) {
      uint32_t id = 0;

      // 1）阻塞式拿到至少一个节点 id
      pop_blocking(id);

      // 收到终止哨兵，直接退出
      if (id == MERGE_TERMINATE_ID) {
        LOG(INFO) << "Merge worker thread received terminate sentinel.";
        break;
      }

      // 简单过滤非法 id
      if (id < num_points) {
        batch.push_back(id);
        // 统计：merge 线程从队列里成功取出一个合法 id
        stats_merge_pop_.fetch_add(1);
      }

      // 2）非阻塞地尽可能多地把当前队列里的任务拉出来，形成一个 batch
      uint32_t tmp = 0;
      while (batch.size() < kMergeBatchSize && try_pop(tmp)) {
        if (tmp == MERGE_TERMINATE_ID) {
          // 如果还有任务没处理完，就把终止哨兵丢回去，留给下一轮退出
          push(MERGE_TERMINATE_ID);
          break;
        }
        if (tmp < num_points) {
          batch.push_back(tmp);
          // 同样这里也统计一次 pop 成功
          stats_merge_pop_.fetch_add(1);
        }
      }

      if (batch.empty()) {
        // 有可能刚才只拿到了终止哨兵或非法 id，继续下一轮
        continue;
      }
      // if(timer.elapsed() >= 5000000)
      // {
        LOG(INFO) << "[MERGE] new batch: raw_ids=" << batch.size();
      // }

      // 去重前的 batch 大小，可以作为一个观察值
      const size_t before_dedup = batch.size();

      // 3）去重：同一个点在队列里出现多次，只合并一次
      std::sort(batch.begin(), batch.end());
      batch.erase(std::unique(batch.begin(), batch.end()), batch.end());

      const size_t after_dedup = batch.size();
      // 累加这次真正要 merge 的节点数
      auto merged_total = stats_merge_nodes_merged_.fetch_add(after_dedup) + after_dedup;

      // if(timer.elapsed() >= 5000000)
      // {
        LOG(INFO) << "[MERGE] new batch true merge: raw_ids=" << after_dedup;
      // }

      // 每 1000 个节点左右打印一次大致统计信息
      if (merged_total % 1000 == 0) {
        LOG(INFO) << "[MERGE] merged_nodes=" << merged_total
                  << ", l1_triggers=" << stats_l1_merge_triggers_.load()
                  << ", popped_ids=" << stats_merge_pop_.load()
                  << ", last_batch_before_dedup=" << before_dedup
                  << ", last_batch_after_dedup=" << after_dedup;
      }

      // 4）按扇区号分组：sector -> [nodes...]
      std::map<uint64_t, std::vector<uint32_t>> sector_nodes;
      for (uint32_t vid : batch) {
        uint64_t sector = node_sector_no(vid);  // 利用已有的 id -> loc -> sector 映射
        sector_nodes[sector].push_back(vid);
      }

      // if(timer.elapsed() >= 5000000)
      // {
        LOG(INFO) << "[MERGE] batch sectors=" << sector_nodes.size();
        timer.reset();
      // }

      // 5）对每个扇区调用真正的合并逻辑
      //   读盘 + 抽干 L1 增量邻居 + pool 合并 + prune_neighbors_pq + BgTask + wbc_write
      for (auto &kv : sector_nodes) {
        uint64_t sector = kv.first;
        const std::vector<uint32_t> &nodes = kv.second;
        merge_nodes_on_sector(sector, nodes, ctx);
      }

      // 6）清空 batch，处理下一轮
      batch.clear();
    }

    LOG(INFO) << "SSDIndex merge worker thread exited. "
              << "l1_triggers=" << stats_l1_merge_triggers_.load()
              << ", popped_ids=" << stats_merge_pop_.load()
              << ", merged_nodes=" << stats_merge_nodes_merged_.load();
  }

  // 然后把这一整页合并并写回（通过 BgTask 交给后台线程写盘）
  template<class T, class TagT>
  void SSDIndex<T, TagT>::merge_nodes_on_sector(uint64_t sector,
                                                const std::vector<uint32_t> &nodes,
                                                void *ctx) {
    if (nodes.empty()) {
      return;
    }

    // LOG(INFO) << "[MERGE] merge_nodes_on_sector BEGIN: sec tor=" << sector
    //           << ", nodes_to_merge=" << nodes.size();

    // 1. 拿一个 QueryBuffer，当成本次 merge 的 scratch
    //    这里传 nullptr 就不会去填充 aligned_query_T
    QueryBuffer<T> *buf = this->pop_query_buf(nullptr);
    char *sector_buf = buf->sector_scratch;
    uint8_t *thread_pq_buf = buf->nbr_vec_scratch;  // PQ 剪枝用的 scratch

    // 2. 页级锁：锁住这一页，防止并发写
    std::vector<IORequest> pages_to_rmw;
    pages_to_rmw.emplace_back(
        IORequest(sector * SECTOR_LEN, this->size_per_io, nullptr, 0, 0));
    std::vector<uint64_t> pages_locked = v2::lockReqs(this->page_lock_table, pages_to_rmw);

    // 3. 读入这一页（带 page cache 的 read_alloc）
    std::vector<IORequest> reads;
    reads.emplace_back(
        IORequest(sector * SECTOR_LEN, this->size_per_io, sector_buf, 0, 0));
    std::vector<uint64_t> read_page_ref;
    this->reader->read_alloc(reads, ctx, &read_page_ref);

    // 4. 对属于这一页的每个点 v 做「L0 + L1 → 重剪枝」
    auto *l1 = this->l1_table_;

    for (uint32_t id : nodes) {
      // 4.0 保护一下非法 id / loc
      uint64_t loc = this->id2loc(id);
      if (loc >= this->cur_loc) {
        continue;  // 已经被删掉或无效
      }
      if (this->loc_sector_no(loc) != sector) {
        continue;  // 双重保险：只处理确实在这一页上的点
      }

      // 4.1 在该页 buffer 中定位到该点的 node 区域
      char *node_buf = this->offset_to_loc(sector_buf, loc);
      DiskNode<T> node(id,
                      this->offset_to_node_coords(node_buf),
                      this->offset_to_node_nhood(node_buf));

      uint32_t old_deg = node.nnbrs;
      uint32_t *old_nbrs = node.nbrs;

      // 4.2 收集候选邻居：旧的 L0 邻居
      std::vector<uint32_t> cand;
      cand.reserve(old_deg + 32);

      for (uint32_t i = 0; i < old_deg; ++i) {
        uint32_t nbr = old_nbrs[i];
        if (nbr != id) {  // 自环直接跳过
          cand.push_back(nbr);
        }
      }

      // 4.3 把 L1[v] 的增量邻居抽干并追加到候选里
      if (l1 != nullptr) {
        std::vector<uint32_t> delta;
        l1->drain_neighbors(id, delta);  // L1[id] → delta，并清空 L1[id]
        cand.insert(cand.end(), delta.begin(), delta.end());
      }

      if (cand.empty()) {
        // 没有邻居了，直接把 L0 清空
        node.nnbrs = 0;
        *(node.nbrs - 1) = 0;  // DiskNode 约定：nbrs[-1] 存 nnbrs
        continue;
      }

      // 4.4 去重 + 去 self-loop
      std::sort(cand.begin(), cand.end());
      cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
      cand.erase(std::remove(cand.begin(), cand.end(), id), cand.end());

      if (cand.empty()) {
        node.nnbrs = 0;
        *(node.nbrs - 1) = 0;
        continue;
      }

      if (cand.size() > MAX_N_EDGES) {
        cand.resize(MAX_N_EDGES);
      }

      // 4.5 基于 PQ 的 L0 重剪枝：cand (L0+L1) → new_nhood (最多 range 条边)
      std::vector<uint32_t> new_nhood;
      new_nhood.reserve(cand.size());

      if (cand.size() <= this->range) {
        // 候选数量本身不超过出度上限，直接使用，也可以选择仍然做一次 occlusion
        new_nhood.assign(cand.begin(), cand.end());
      } else {
        // 使用 PQ 近似距离 + occlusion 进行剪枝，逻辑复用 prune_neighbors_pq
        std::vector<float> dists(cand.size(), 0.0f);
        std::vector<Neighbor> pool(cand.size());

        // 4.5.1 用 PQ 估计 dist(id, cand[k])
        this->nbr_handler->compute_dists(
            id,
            cand.data(),
            cand.size(),
            dists.data(),
            thread_pq_buf  // 复用 QueryBuffer 里的 scratch
        );

        // 4.5.2 组装 Neighbor 池并按距离排序
        for (uint32_t k = 0; k < cand.size(); ++k) {
          pool[k].id = cand[k];
          pool[k].distance = dists[k];
        }
        std::sort(pool.begin(), pool.end());   // Neighbor::operator< 默认按 distance 升序
        if (pool.size() > this->maxc) {
          pool.resize(this->maxc);
        }

        // 4.5.3 用 PQ 的 occlusion 规则剪枝为 ≤ range 条边
        this->prune_neighbors_pq(pool, new_nhood, thread_pq_buf);
      }

      // 4.6 把新的邻居列表写回 page buffer 里的 node
      node.nnbrs = static_cast<uint32_t>(new_nhood.size());
      *(node.nbrs - 1) = node.nnbrs;
      if (node.nnbrs > 0) {
        memcpy(node.nbrs, new_nhood.data(), node.nnbrs * sizeof(uint32_t));
      }

      // 如果后面给 PQ 邻接（nbr_handler）增加 “set_nbrs / update_nbrs” 接口，
      // 可以在这里同步 PQ 图，比如：
      //
      //   this->nbr_handler->update_nbrs(id, new_nhood.data(), node.nnbrs);
      //
      // 为了不和现有接口冲突，这里先不调用任何 PQ 更新函数。
    }

    // 5. 把修改过的这一页复制到 update_buf，交给后台写盘线程
    //    update_buf 在 bg_io_thread 里写完之后会被 aligned_free，并把 buf 归还 buffer 池。
    assert(buf->update_buf == nullptr);
    pipeann::alloc_aligned(reinterpret_cast<void **>(&buf->update_buf),
                          this->size_per_io,
                          SECTOR_LEN);
    memcpy(buf->update_buf, sector_buf, this->size_per_io);

    char *update_buf = buf->update_buf;

    // 6. 封装 BgTask，异步写盘 + 解锁 + deref cache
    auto *task = new BgTask();
    task->thread_data = buf;
    task->terminate = false;

    task->writes.clear();
    task->writes.emplace_back(
        IORequest(sector * SECTOR_LEN, this->size_per_io, update_buf, 0, 0));

    task->pages_to_unlock = std::move(pages_locked);
    task->pages_to_deref = std::move(read_page_ref);

    this->bg_tasks.push(task);
    this->bg_tasks.push_notify_all();

    // LOG(INFO) << "[MERGE] merge_nodes_on_sector END: sector=" << sector
    //           << ", nodes_to_merge=" << nodes.size()
    //           << ", write_bytes=" << this->size_per_io;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_page_for_node_inline(uint32_t center) {
  #if SSD_L1_MERGE_MODE == SSD_L1_MERGE_INLINE_PAGE
    // 0. 简单过滤非法 id
    // if (center >= this->num_points) {
    //   return;
    // }
    assert( center < this->num_points);
    // assert( page_lock_table.size() == 0);

    // 1. 找到 center 所在页
    // uint64_t page = this->id2page(center);
    // if (page == kInvalidID) {
    //   LOG(WARNING) << "[MERGE][INLINE] center=" << center
    //               << " has invalid page, skip.";
    //   return;
    // }
    uint32_t center_loc = optimize_lock_id( center);
    uint64_t center_page = loc_sector_no( center_loc);
    // if( cur_loc == kInvalidID || deletion_set->find(hop12_nbr_array[i]) != deletion_set->end()) {
    if( center_loc == kInvalidID) {
      page_lock_table.unlock( center_page);
      return;
    }

    // 2. 收集这一页上的所有有效点及其旧 loc
    auto layout = this->get_page_layout(static_cast<uint32_t>(center_page));

    std::vector<uint32_t> ids;
    std::vector<uint64_t> old_locs;
    ids.reserve(layout.size());
    old_locs.reserve(layout.size());

    for (uint32_t vid : layout) {
      if (vid == kInvalidID || vid == kAllocatedID) {
        continue;
      }
      uint64_t loc = this->id2loc(vid);
      if (loc == kInvalidID) {
        continue;
      }
      // if (this->loc_sector_no(loc) != cur_page) {
      //   continue;  // 双重保险：只搬这一页上的点
      // }
      assert( loc < this->cur_loc);
      assert( loc_sector_no(loc) == cur_page);
      ids.push_back(vid);
      old_locs.push_back(loc);
    }

    // if (ids.empty()) {
    //   // LOG(INFO) << "[MERGE][INLINE] page=" << page
    //   //           << " has no valid nodes, center=" << center;
    //   return;
    // }
    assert( !ids.empty());

    // 3. 申请 QueryBuffer，当成本次 merge 的 scratch
    QueryBuffer<T> *buf = this->pop_query_buf(nullptr);
    char *sector_buf = buf->sector_scratch;
    uint8_t *thread_pq_buf = buf->nbr_vec_scratch;

    auto reader = this->reader;
    void *ctx = reader->get_ctx();

    // 4. 页级锁 + 读入这一页
    // char *page_buf = nullptr;
    // // remember free!!
    // pipeann::alloc_aligned( (void **) &page_buf, size_per_io, SECTOR_LEN);
    std::vector<IORequest> nxt_read_req;
    nxt_read_req.emplace_back( center_page * SECTOR_LEN, this->size_per_io, sector_buf, 0, 0);
    // std::vector<uint64_t> pages_locked =
    //     v2::lockReqs(this->page_lock_table, pages_to_rmw);
    reader->read_alloc( nxt_read_req, ctx);

    // std::vector<IORequest> reads;
    // reads.emplace_back(
    //     IORequest(page * SECTOR_LEN, this->size_per_io, sector_buf, 0, 0));
    // std::vector<uint64_t> read_page_ref;
    // reader->read_alloc(reads, ctx, &read_page_ref);

    // 5. 对这一页上的每个点做「L0 + L1 → 重剪枝」，只改 sector_buf，不写回旧页
    auto *l1 = this->l1_table_;
    // LOG(INFO) << "1";

    for (size_t idx = 0; idx < ids.size(); ++idx) {
      uint32_t id = ids[idx];
      uint64_t loc = old_locs[idx];

      // 再次确认 loc 还在这一页（防御性检查）
      // if (loc >= this->cur_loc || this->loc_sector_no(loc) != cur_page) {
      //   continue;
      // }
      assert( loc < this->cur_loc && loc_sector_no(loc) == center_page);

      char *node_buf = this->offset_to_loc(sector_buf, loc);
      DiskNode<T> node(id,
                      this->offset_to_node_coords(node_buf),
                      this->offset_to_node_nhood(node_buf));

      uint32_t old_deg = node.nnbrs;
      uint32_t *old_nbrs = node.nbrs;

      // 5.1 旧 L0 邻居
      std::vector<uint32_t> cand;
      cand.reserve(old_deg + 32);
      for (uint32_t i = 0; i < old_deg; ++i) {
        uint32_t nbr = old_nbrs[i];
        assert( nbr != id);
        // if (nbr != id) {
        //   cand.push_back(nbr);
        // }
        assert( nbr != kInvalidID);
        assert( nbr != kAllocatedID);
        idx_lock_table.rdlock( nbr);
        if( id2loc( nbr) != kInvalidID) {
          cand.push_back(nbr);
        }
        idx_lock_table.unlock( nbr);
      }

      // 5.2 把 L1[id] 的增量邻居抽干并追加到候选里
      if (l1 != nullptr) {
        std::vector<uint32_t> delta;
        l1->drain_neighbors(id, delta);  // L1[id] -> delta，并清空 L1[id]
        // cand.insert(cand.end(), delta.begin(), delta.end());
        for( auto nbr : delta) {
          assert( nbr != kInvalidID);
          assert( nbr != kAllocatedID);
          idx_lock_table.rdlock( nbr);
          if( id2loc( nbr) != kInvalidID) {
            cand.push_back(nbr);
          }
          idx_lock_table.unlock( nbr);
        }
      }

      assert( new_nhood_set.size() > 0);
      if (cand.empty()) {
        node.nnbrs = 0;
        *(node.nbrs - 1) = 0;
        continue;
      }

      // 5.3 去重 + 去 self-loop
      std::sort(cand.begin(), cand.end());
      cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
      cand.erase(std::remove(cand.begin(), cand.end(), id), cand.end());

      if (cand.empty()) {
        node.nnbrs = 0;
        *(node.nbrs - 1) = 0;
        continue;
      }

      if (cand.size() > MAX_N_EDGES) {
        cand.resize(MAX_N_EDGES);
      }

      // 5.4 PQ 剪枝
      std::vector<uint32_t> new_nhood;
      new_nhood.reserve(cand.size());

      if (cand.size() <= this->range) {
        new_nhood.assign(cand.begin(), cand.end());
      } else {
        std::vector<float> dists(cand.size(), 0.0f);
        std::vector<Neighbor> pool(cand.size());

        this->nbr_handler->compute_dists(
            id, cand.data(), cand.size(), dists.data(), thread_pq_buf);

        for (uint32_t k = 0; k < cand.size(); ++k) {
          pool[k].id = cand[k];
          pool[k].distance = dists[k];
        }
        std::sort(pool.begin(), pool.end());
        if (pool.size() > this->maxc) {
          pool.resize(this->maxc);
        }

        this->prune_neighbors_pq(pool, new_nhood, thread_pq_buf);
      }

      node.nnbrs = static_cast<uint32_t>(new_nhood.size());
      *(node.nbrs - 1) = node.nnbrs;
      if (node.nnbrs > 0) {
        memcpy(node.nbrs, new_nhood.data(), node.nnbrs * sizeof(uint32_t));
      }
    }
    this->push_query_buf( buf);

    // 注意：sector_buf 现在是「重剪枝后」的这一页，我们不会再写回旧页，而是搬到新 loc
    // LOG(INFO) << "2";

    uint64_t write_page = (uint64_t) alloc_1page();
    page_lock_table.wrlock( write_page);
    std::vector<IORequest> nxt_write_req;
    nxt_write_req.emplace_back( write_page * SECTOR_LEN, size_per_io, sector_buf, 0, 0);
    reader->wbc_write( nxt_write_req, ctx);
    auto locked = lock_idx(idx_lock_table, kInvalidID, ids);
    for (uint32_t i = 0; i < ids.size(); ++i) {
      // set_loc2id( sector_to_loc( cur_page, i), kInvalidID);
      assert( ids[i] != kInvalidID);
      set_loc2id( sector_to_loc( write_page, i), ids[i]);
      set_id2loc( ids[i], sector_to_loc( write_page, i));
    }
    unlock_idx(idx_lock_table, locked);
    
    std::vector<uint64_t> page2deref;
    reader->write(nxt_write_req, ctx);
    page2deref.push_back( center_page);
    page2deref.push_back( write_page);
    reader->deref( &page2deref, ctx);
    empty_pages.push( center_page);
    page_lock_table.unlock( center_page);
    page_lock_table.unlock( write_page);

    // 6. 为这一页上的所有点一次性分配新的 loc
    // std::set<uint64_t> page_need_to_read;
    // std::vector<uint64_t> new_locs =
    //     this->alloc_loc(static_cast<int>(ids.size()),
    //                     std::vector<uint64_t>{},  // hint_pages 为空
    //                     page_need_to_read);

    // if (new_locs.size() != ids.size()) {
    //   LOG(ERROR) << "[MERGE][INLINE] alloc_loc failed: ids=" << ids.size()
    //             << " new_locs=" << new_locs.size();
    //   // 这里直接把资源释放掉，走同步路径的清理
    //   v2::unlockReqs(this->page_lock_table, pages_locked);
    //   reader->deref(&read_page_ref, ctx);
    //   this->push_query_buf(buf);
    //   return;
    // }

    // 7. 为每个目标页准备一个 page buffer（不 pre-read，因为 alloc_loc 只用了空页/新页）
    // std::map<uint64_t, char *> page_buf_map;
    // for (uint64_t loc : new_locs) {
    //   uint64_t dst_page = this->loc_sector_no(loc);
    //   if (page_buf_map.find(dst_page) == page_buf_map.end()) {
    //     char *dst_buf = nullptr;
    //     pipeann::alloc_aligned((void **) &dst_buf,
    //                           this->size_per_io,
    //                           SECTOR_LEN);
    //     memset(dst_buf, 0, this->size_per_io);
    //     page_buf_map[dst_page] = dst_buf;
    //   }
    // }

    // 8. 把旧页里的每个点拷贝到对应的新 loc（新页 buffer）
    // for (size_t i = 0; i < ids.size(); ++i) {
    //   uint64_t old_loc = old_locs[i];
    //   uint64_t new_loc = new_locs[i];

    //   uint64_t dst_page = this->loc_sector_no(new_loc);
    //   char *dst_buf = page_buf_map[dst_page];

    //   char *src_node_buf = this->offset_to_loc(sector_buf, old_loc);
    //   char *dst_node_buf = this->offset_to_loc(dst_buf, new_loc);

    //   memcpy(dst_node_buf, src_node_buf, this->max_node_len);
    // }

    // 9. 在锁保护下更新 id2loc / loc2id，并让旧页进入 empty_pages
    // auto locked_idx = this->lock_idx(this->idx_lock_table, kInvalidID, ids);
    // auto locked_page_idx = this->lock_page_idx(this->page_idx_lock_table, kInvalidID, ids);

    // for (size_t i = 0; i < ids.size(); ++i) {
    //   this->set_id2loc(ids[i], static_cast<uint32_t>(new_locs[i]));
    // }
    // this->erase_and_set_loc(old_locs, new_locs, ids);

    // this->unlock_page_idx(this->page_idx_lock_table, locked_page_idx);
    // this->unlock_idx(this->idx_lock_table, locked_idx);

    // 10. 把新页写回磁盘：由宏控制同步/异步写盘
  #ifdef BG_IO_THREAD
    // 10.1 异步写盘：把 page_buf_map 搬到一个连续的 update_buf 中
    assert(buf->update_buf == nullptr);
    const size_t num_pages = page_buf_map.size();
    const size_t total_bytes = num_pages * this->size_per_io;
    pipeann::alloc_aligned(reinterpret_cast<void **>(&buf->update_buf),
                          total_bytes,
                          SECTOR_LEN);
    char *update_base = buf->update_buf;

    std::vector<IORequest> writes;
    writes.reserve(num_pages);

    size_t page_idx = 0;
    for (auto &kv : page_buf_map) {
      uint64_t dst_page = kv.first;
      char *src = kv.second;
      char *dst = update_base + page_idx * this->size_per_io;
      memcpy(dst, src, this->size_per_io);

      writes.emplace_back(
          IORequest(dst_page * SECTOR_LEN,
                    this->size_per_io,
                    dst,
                    0,
                    0));
      ++page_idx;
    }

    // 本地的 page buffer 可以释放了（真正的写盘 buffer 在 update_buf 里）
    for (auto &kv : page_buf_map) {
      pipeann::aligned_free(kv.second);
    }
    page_buf_map.clear();

    // 10.2 封装 BgTask：写新页 + 解锁旧页 + deref + 归还 QueryBuffer
    auto *task = new BgTask();
    task->thread_data = buf;
    task->terminate = false;
    task->writes = std::move(writes);
    task->pages_to_unlock = std::move(pages_locked);   // 解锁旧页
    task->pages_to_deref = std::move(read_page_ref);   // deref 旧页的 cache

    this->bg_tasks.push(task);
    this->bg_tasks.push_notify_all();

  #else   // 同步写盘路径

    // std::vector<IORequest> writes;
    // writes.reserve(page_buf_map.size());
    // for (auto &kv : page_buf_map) {
    //   uint64_t dst_page = kv.first;
    //   char *buf_page = kv.second;
    //   writes.emplace_back(
    //       IORequest(dst_page * SECTOR_LEN,
    //                 this->size_per_io,
    //                 buf_page,
    //                 0,
    //                 0));
    // }

    // 当前线程直接写盘
    // reader->write(writes, ctx, /*async=*/false);

    // 释放新页 buffer
    // for (auto &kv : page_buf_map) {
    //   pipeann::aligned_free(kv.second);
    // }
    // page_buf_map.clear();

    // 解锁旧页 + deref cache
    // v2::unlockReqs(this->page_lock_table, pages_locked);
    // reader->deref(&read_page_ref, ctx);

    // 同步模式下由当前线程归还 QueryBuffer
    // this->push_query_buf(buf);

  #endif  // BG_IO_THREAD

    // LOG(INFO) << "[MERGE][INLINE] page=" << page
    //           << " relocated, num_ids=" << ids.size();

  #else
    (void) center;
  #endif  // SSD_L1_MERGE_MODE == SSD_L1_MERGE_INLINE_PAGE
  }





  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::copy_index(const std::string &prefix_in, const std::string &prefix_out) {
    LOG(INFO) << "Copying disk index from " << prefix_in << " to " << prefix_out;
    std::filesystem::copy(prefix_in + "_disk.index", prefix_out + "_disk.index",
                          std::filesystem::copy_options::overwrite_existing);
    if (std::filesystem::exists(prefix_in + "_disk.index.tags")) {
      std::filesystem::copy(prefix_in + "_disk.index.tags", prefix_out + "_disk.index.tags",
                            std::filesystem::copy_options::overwrite_existing);
    } else {
      // remove the original tags.
      std::filesystem::remove(prefix_out + "_disk.index.tags");
    }

    // nbr.
    this->nbr_handler->load(prefix_in.c_str());
    this->nbr_handler->save(prefix_out.c_str());

    // partition data
    if (std::filesystem::exists(prefix_in + "_partition.bin.aligned")) {
      std::filesystem::copy(prefix_in + "_partition.bin.aligned", prefix_out + "_partition.bin.aligned",
                            std::filesystem::copy_options::overwrite_existing);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_buffers(uint64_t n_threads) {
    uint64_t n_buffers = n_threads * 2;
    LOG(INFO) << "Init buffers for " << n_threads << " threads, setup " << n_buffers << " buffers.";
    this->thread_data_queue.null_T = nullptr;
    this->merge_queue_.null_T = MERGE_NULL_ID;
    for (uint64_t i = 0; i < n_buffers; i++) {
      QueryBuffer<T> *data = new QueryBuffer<T>();
      this->init_query_buf(*data);
      this->thread_data_bufs.push_back(data);
      this->thread_data_queue.push(data);
      this->reader->register_buf(data->sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, 0);
    }

#ifndef READ_ONLY_TESTS
    // background thread: 负责 BgTask 的写盘
    LOG(INFO) << "Setup " << kBgIOThreads << " background I/O threads for insert...";
    for (int i = 0; i < kBgIOThreads; ++i) {
      bg_io_thread_[i] = new std::thread(&SSDIndex<T, TagT>::bg_io_thread, this);
      bg_io_thread_[i]->detach();
    }

    // 根据 SSD_L1_MERGE_MODE 决定是否启用后台 L1 merge 线程
#if SSD_L1_MERGE_MODE == SSD_L1_MERGE_BG_WORKER
    LOG(INFO) << "Setup L1 merge worker thread for disk-graph merging (BG_WORKER mode)...";
    std::thread(&SSDIndex<T, TagT>::merge_worker_thread, this).detach();
#else
    LOG(INFO) << "L1 merge worker thread disabled (INLINE_PAGE mode, merge in insert path).";
#endif

#endif
}



  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::destroy_buffers() {
#ifndef READ_ONLY_TESTS
    for (int i = 0; i < kBgIOThreads; ++i) {
      if (bg_io_thread_[i] != nullptr) {
        auto bg_task = new BgTask{
            .thread_data = nullptr, .writes = {}, .pages_to_unlock = {}, .pages_to_deref = {}, .terminate = true};
        bg_tasks.push(bg_task);
        bg_tasks.push_notify_all();
        bg_io_thread_[i] = nullptr;
      }
    }
#endif

    while (!this->thread_data_bufs.empty()) {
      auto buf = this->thread_data_bufs.back();
      pipeann::aligned_free((void *) buf->coord_scratch);
      pipeann::aligned_free((void *) buf->sector_scratch);
      pipeann::aligned_free((void *) buf->nbr_vec_scratch);
      pipeann::aligned_free((void *) buf->nbr_ctx_scratch);
      pipeann::aligned_free((void *) buf->aligned_dist_scratch);
      pipeann::aligned_free((void *) buf->aligned_query_T);
      this->thread_data_bufs.pop_back();
      this->thread_data_queue.pop();
      delete buf;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path) {
    if (mem_index_path.empty()) {
      LOG(ERROR) << "mem_index_path is needed";
      exit(-1);
    }
    mem_index_ = std::make_unique<pipeann::Index<T, uint32_t>>(metric, query_dim, 0, false, false, true);
    mem_index_->load(mem_index_path.c_str());
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, uint32_t num_threads, bool new_index_format,
                              bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_disk.index";
    this->_disk_index_file = disk_index_file;

    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  uint64_t SSDIndex<T, TagT>::return_nd() {
    return this->num_points;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector,
                                           const uint64_t num_points) {
    // 完全忽略 partition 文件，使用等距映射。
    // 这里也干脆去掉 OpenMP 并行，反正只初始化一次，3M 点也就几十毫秒级。
    (void) index_prefix;      // 不再使用 partition.bin
    (void) nnodes_per_sector; // 不再依赖 C
    (void) num_points;        // 使用 this->num_points 更稳妥

    // 保险起见，让 num_points / cur_loc 完全由 metadata 决定
    const size_t npts   = static_cast<size_t>(this->num_points);
    const size_t nslots = static_cast<size_t>(this->cur_loc);

    id2loc_.assign(npts, 0u);           // 大小 = npts
    loc2id_.assign(nslots, kInvalidID); // 大小 = cur_loc

    // 0..(npts-1)：等距映射 id -> loc
    for (size_t i = 0; i < npts; ++i) {
      id2loc_[i] = static_cast<uint32_t>(i);
      loc2id_[i] = static_cast<uint32_t>(i);
    }

    // npts..(nslots-1)：是最后一页对齐产生的 padding，统一标为无效
    // 上面 assign 已经全部设成 kInvalidID 了，这里实际上可以不写。
    LOG(INFO) << "Page layout loaded with equal mapping (no per-page reserved slots). "
              << "Npoints=" << npts << " Cur_loc=" << nslots;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_tags(const std::string &tag_file_name, size_t offset) {
    size_t tag_num, tag_dim;
    std::vector<TagT> tag_v;
    this->tags.clear();

    if (!file_exists(tag_file_name)) {
      LOG(INFO) << "Tags file not found. Using equal mapping";
      // Equal mapping are by default eliminated in tags map.
    } else {
      LOG(INFO) << "Load tags from existing file: " << tag_file_name;
      pipeann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
      LOG(INFO) << "Loaded " << tags.size() << " tags";
    }
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::get_vector_by_id(const uint32_t &id, T *vector_coords) {
    if (!enable_tags) {
      LOG(INFO) << "Tags are disabled, cannot retrieve vector";
      return -1;
    }
    uint32_t pos = id;
    size_t num_sectors = node_sector_no(pos);
    std::ifstream disk_reader(_disk_index_file.c_str(), std::ios::binary);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
    disk_reader.seekg(SECTOR_LEN * num_sectors, std::ios::beg);
    disk_reader.read(sector_buf.get(), size_per_io);
    char *node_coords = (offset_to_node(sector_buf.get(), pos));
    memcpy((void *) vector_coords, (void *) node_coords, data_dim * sizeof(T));
    return 0;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
