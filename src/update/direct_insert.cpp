#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>

#include <mutex>
#include <array>

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

#include "v2/l1_neighbor_table.h"

namespace pipeann {

  namespace {
    // 对扇区做 striping 粗粒度锁，避免多线程对同一 sector 做 RMW 丢更新。
    // 这里用 2^16 个互斥锁，通过 sector_no & (N-1) 做映射，空间开销可以接受。
    constexpr uint32_t kInsertLockStriping = 1u << 16;  // 65536

    std::array<std::mutex, kInsertLockStriping> g_insert_page_locks;

    inline std::mutex& page_lock(uint64_t sector_no) {
      return g_insert_page_locks[sector_no & (kInsertLockStriping - 1)];
    }
  }  // anonymous namespace

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::insert_in_place(const T *point1, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
    if (unlikely(size_per_io != SECTOR_LEN)) {
      LOG(ERROR) << "Insert not supported for size_per_io == " << size_per_io;
    }

    QueryBuffer<T> *read_data = this->pop_query_buf(point1);
    T *point = read_data->aligned_query_T;  // normalized point for cosine.
    void *ctx = reader->get_ctx();

    uint32_t target_id = cur_id++;

    // write neighbor (e.g., PQ).
    nbr_handler->insert(point, target_id);

    std::vector<Neighbor> exp_node_info;
    tsl::robin_map<uint32_t, T *> coord_map;
    coord_map.reserve(10 * this->l_index);
    // Dynamic alloc and not using MAX_N_CMPS to reduce memory footprint.
    T *coord_buf = nullptr;
    alloc_aligned((void **) &coord_buf, 10 * this->l_index * this->aligned_dim, 256);
    std::vector<uint64_t> page_ref{};
    // re-normalize point1 to support inner_product search (it adds one more dimension, so not idempotent).
    this->do_beam_search(point1, 0, l_index, beam_width, exp_node_info, &coord_map, coord_buf, nullptr, deletion_set,
                         false, &page_ref);
    std::vector<uint32_t> new_nhood;
    prune_neighbors(coord_map, exp_node_info, new_nhood);
    // locs[new_nhood.size()] is the target, locs[0:new_nhood.size() - 1] are the neighbors.

    // lock the pages to write
    aligned_free(coord_buf);

    // -------- 4.1 只给新点分配一个新的 loc（append 到尾部） --------
    uint32_t new_id = target_id;  // 已经在前面 cur_id++ 得到
    uint64_t loc = cur_loc.fetch_add(1, std::memory_order_acq_rel);
    set_id2loc(new_id, loc);
    set_loc2id(loc, new_id);

    // -------- 4.2 准备一个 sector buffer，写入新点的 node record --------
    // 先算出这个 loc 所在的 sector 和在文件里的偏移
    uint64_t sector = loc_sector_no(loc);      // 已有辅助函数
    uint64_t off    = sector * SECTOR_LEN;     // 一个 sector = SECTOR_LEN 字节

    // 对同一扇区做并发保护，避免多线程对同一 sector 做 RMW 时丢更新
    std::lock_guard<std::mutex> sector_guard(page_lock(sector));

    // 为这一页申请对齐的缓冲区
    char *page_buf = nullptr;
    pipeann::alloc_aligned((void **) &page_buf, size_per_io, SECTOR_LEN);

    // 先清零，避免读到 EOF 以外时留下脏数据
    memset(page_buf, 0, size_per_io);

    // 把当前 sector 原来的内容读进来（如果是新扇区，就全是 0）
    {
      std::vector<IORequest> reads;
      reads.emplace_back(IORequest(off, size_per_io, page_buf, 0, 0));

      // 使用带 page cache 的 read_alloc：先查 v2::cache，miss 才真正走磁盘
      std::vector<uint64_t> read_page_ref;
      reader->read_alloc(reads, ctx, &read_page_ref);
      reader->deref(&read_page_ref, ctx);
    }

    // 在这一页里定位到 loc 对应的 node 位置
    char *node_buf = offset_to_loc(page_buf, loc);

    DiskNode<T> node(new_id,
                     offset_to_node_coords(node_buf),
                     offset_to_node_nhood(node_buf));

    // 写向量，这里用的是归一化后的 point
    memcpy(node.coords, point, data_dim * sizeof(T));

    // 写邻居数量和邻居 ID
    node.nnbrs = static_cast<uint32_t>(new_nhood.size());
    *(node.nbrs - 1) = node.nnbrs;  // DiskNode 约定：nbrs 前一个位置存 nnbrs
    if (!new_nhood.empty()) {
      memcpy(node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
    }

    // 更新 tag 映射
    tags.insert_or_assign(new_id, tag);

    // 把整个 sector 写回去（同页其它 node 的内容都被保留）
    {
      std::vector<IORequest> writes;
      writes.emplace_back(IORequest(off, size_per_io, page_buf, 0, 0));
      reader->write(writes, ctx, /*async=*/false);
    }

    pipeann::aligned_free(page_buf);

    // -------- 4.3 只更新 L1 里的反向边，不动老页 --------
    auto *l1 = this->l1_table_;
    if (l1 != nullptr) {
      auto &thread_pq_buf = read_data->nbr_vec_scratch;  // 轻剪时用的 scratch
      for (auto v : new_nhood) {
        l1->add_backlink(
            v, new_id,
            [this, new_id, &thread_pq_buf](uint32_t center, std::vector<uint32_t> &nbrs) {
              this->prune_l1_delta(center, new_id, nbrs, thread_pq_buf);
            });
      }
    }

    // -------- 4.4 清理、返回 --------
    this->push_query_buf(read_data);
    return target_id;
  }

  // 对同一个 sector 上的一批点做：
  //   L0[v] + L1[v] → 候选集 cand → delta_prune_neighbors_pq → 新的 L0[v]
  // 然后把这一整页写回（通过 BgTask 交给后台线程写盘）
  template<class T, class TagT>
  void SSDIndex<T, TagT>::merge_nodes_on_sector(uint64_t sector,
                                                const std::vector<uint32_t> &nodes,
                                                void *ctx) {
    if (nodes.empty()) {
      return;
    }

    // 1. 拿一个 QueryBuffer，当成本次 merge 的 scratch
    //    这里传 nullptr 就不会去填充 aligned_query_T
    QueryBuffer<T> *buf = this->pop_query_buf(nullptr);
    char *sector_buf = buf->sector_scratch;
    uint8_t *thread_pq_buf = buf->nbr_vec_scratch;  // PQ 剪枝用的 scratch

    // 2. 页级锁：锁住这一页，防止并发写
    std::vector<IORequest> pages_to_rmw;
    pages_to_rmw.emplace_back(
        IORequest(sector * SECTOR_LEN, this->size_per_io, nullptr, 0, 0));
    // page_lock_table 是 SparseLockTable<uint64_t>，直接用旧逻辑加锁
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

      // 4.5 调用基于 PQ 的 delta 剪枝，得到新的 L0[v]
      std::vector<uint32_t> new_nhood;
      new_nhood.reserve(cand.size());

      // 这个函数在 prune_neighbors.cpp 里已经实现，
      // 负责：
      //   1）用 PQ 估计 dist(v, u)
      //   2）做三角预筛 + occlusion
      //   3）把结果写到 new_nhood
      this->delta_prune_neighbors_pq(id, cand, new_nhood, thread_pq_buf);

      // 4.6 把新的邻居列表写回 page buffer 里的 node
      node.nnbrs = static_cast<uint32_t>(new_nhood.size());
      *(node.nbrs - 1) = node.nnbrs;
      if (!new_nhood.empty()) {
        memcpy(node.nbrs,
              new_nhood.data(),
              node.nnbrs * sizeof(uint32_t));
      }

      // 如果后面你给 PQ 邻接（nbr_handler）增加 “set_nbrs / update_nbrs” 接口，
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
  }

  template<class T, class TagT>
  void SSDIndex<T, TagT>::bg_io_thread() {
    auto ctx = reader->get_ctx();
    auto timer = pipeann::Timer();
    uint64_t n_tasks = 0;

    while (true) {
      auto task = bg_tasks.pop();
      while (task == nullptr) {
        this->bg_tasks.wait_for_push_notify();
        task = bg_tasks.pop();
      }

      if (unlikely(task->terminate)) {
        delete task;
        break;
      }

      reader->write(task->writes, ctx);
      aligned_free(task->thread_data->update_buf);
      task->thread_data->update_buf = nullptr;

      v2::unlockReqs(this->page_lock_table, task->pages_to_unlock);
      reader->deref(&task->pages_to_deref, ctx);
      this->push_query_buf(task->thread_data);
      delete task;
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << (double) n_tasks * 1e6 / timer.elapsed()
                  << " tasks/sec.";
        timer.reset();
        n_tasks = 0;
      }
    }
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
