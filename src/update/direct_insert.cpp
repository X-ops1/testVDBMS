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
#include <cassert>


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


  #ifdef BG_IO_THREAD
    // -------- 4.2 把新点的写盘交给后台写线程 --------

    // 4.2.1 读出这个 sector（或者构造一个全 0 的新页）
    char *page_buf = nullptr;
    pipeann::alloc_aligned((void **) &page_buf, size_per_io, SECTOR_LEN);
    memset(page_buf, 0, size_per_io);

    std::vector<IORequest> reads;
    reads.emplace_back(IORequest(off, size_per_io, page_buf, 0, 0));

    // 使用 lock_table 对这一页加锁，锁的释放交给 bg_io_thread
    auto pages_locked = v2::lockReqs(this->page_lock_table, reads);

    std::vector<uint64_t> read_page_ref;
    reader->read_alloc(reads, ctx, &read_page_ref);
    // 不在前台线程里 deref / 解锁，交给 BgTask + bg_io_thread

    // 4.2.2 在 page_buf 中写入新点
    char *node_buf = offset_to_loc(page_buf, loc);
    DiskNode<T> node(new_id,
                    offset_to_node_coords(node_buf),
                    offset_to_node_nhood(node_buf));

    // 写向量（这里 point 已经是归一化后的向量）
    memcpy(node.coords, point, this->data_dim * sizeof(T));

    // 写邻居数量和邻居 ID
    node.nnbrs = static_cast<uint32_t>(new_nhood.size());
    *(node.nbrs - 1) = node.nnbrs;
    if (!new_nhood.empty()) {
      memcpy(node.nbrs, new_nhood.data(),
            new_nhood.size() * sizeof(uint32_t));
    }

    // 更新 tag 映射（只改内存哈希表）
    this->tags.insert_or_assign(new_id, tag);

    // 4.2.3 把 page_buf 拷贝到 QueryBuffer::update_buf，交给后台写线程
    auto *buf = read_data;  // 当前插入线程用的 QueryBuffer
    assert(buf->update_buf == nullptr);
    pipeann::alloc_aligned((void **) &buf->update_buf, size_per_io, SECTOR_LEN);
    memcpy(buf->update_buf, page_buf, size_per_io);
    pipeann::aligned_free(page_buf);

    std::vector<IORequest> writes;
    writes.emplace_back(IORequest(off, size_per_io, buf->update_buf, 0, 0));

    // 4.2.4 打一个 BgTask 丢进 bg_tasks 队列
    auto *task = new BgTask();
    task->thread_data     = buf;
    task->writes          = std::move(writes);
    task->pages_to_unlock = std::move(pages_locked);
    task->pages_to_deref  = std::move(read_page_ref);
    task->terminate       = false;

    this->bg_tasks_submitted.fetch_add(1, std::memory_order_relaxed);
    this->bg_tasks_pending.fetch_add(1, std::memory_order_relaxed);

    this->bg_tasks.push(task);
    this->bg_tasks.push_notify_all();

  #else   // 没开后台写线程时，保持原来的同步写盘逻辑
    char *page_buf = nullptr;
    pipeann::alloc_aligned((void **) &page_buf, size_per_io, SECTOR_LEN);
    memset(page_buf, 0, size_per_io);

    {
      auto &page_mtx = page_lock(sector);
      std::unique_lock<std::mutex> sector_guard(page_mtx);

      std::vector<IORequest> reads;
      reads.emplace_back(IORequest(off, size_per_io, page_buf, 0, 0));
      std::vector<uint64_t> read_page_ref;
      reader->read_alloc(reads, ctx, &read_page_ref);
      reader->deref(&read_page_ref, ctx);

      char *node_buf = offset_to_loc(page_buf, loc);
      DiskNode<T> node(new_id,
                      offset_to_node_coords(node_buf),
                      offset_to_node_nhood(node_buf));

      memcpy(node.coords, point, this->data_dim * sizeof(T));
      node.nnbrs = static_cast<uint32_t>(new_nhood.size());
      *(node.nbrs - 1) = node.nnbrs;
      if (!new_nhood.empty()) {
        memcpy(node.nbrs, new_nhood.data(),
              new_nhood.size() * sizeof(uint32_t));
      }
      this->tags.insert_or_assign(new_id, tag);

      std::vector<IORequest> writes;
      writes.emplace_back(IORequest(off, size_per_io, page_buf, 0, 0));
      reader->write(writes, ctx, /*async=*/false);
    }

    pipeann::aligned_free(page_buf);

  #endif  // BG_IO_THREAD

    // -------- 4.3 更新 L1 里的反向边 --------
    auto *l1 = this->l1_table_;
    if (l1 != nullptr) {
      for (auto v : new_nhood) {
        l1->add_backlink(v, new_id);
      }
    }
    // -------- 4.4 清理、返回 --------
    #ifndef BG_IO_THREAD
      // 同步写盘时，插入线程自己归还 QueryBuffer
      this->push_query_buf(read_data);
    #endif
      return new_id;
  }


  // template<class T, class TagT>
  // void SSDIndex<T, TagT>::bg_io_thread() {
  //   LOG(INFO) << "bg_io_thread started.";
  //   auto ctx = reader->get_ctx();
  //   auto timer = pipeann::Timer();
  //   uint64_t n_tasks = 0;

  //   while (true) {
  //     auto task = bg_tasks.pop();
  //     while (task == nullptr) {
  //       this->bg_tasks.wait_for_push_notify();
  //       task = bg_tasks.pop();
  //     }

  //     if (unlikely(task->terminate)) {
  //       LOG(INFO) << "bg_io_thread received terminate task, exiting. "
  //                 << "processed_tasks_so_far=" << n_tasks;
  //       delete task;
  //       break;
  //     }

  //     reader->write(task->writes, ctx);
  //     aligned_free(task->thread_data->update_buf);
  //     task->thread_data->update_buf = nullptr;

  //     v2::unlockReqs(this->page_lock_table, task->pages_to_unlock);
  //     reader->deref(&task->pages_to_deref, ctx);
  //     this->push_query_buf(task->thread_data);
  //     delete task;
  //     ++n_tasks;

  //     if (timer.elapsed() >= 5000000) {
  //       LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << (double) n_tasks * 1e6 / timer.elapsed()
  //                 << " tasks/sec.";
  //       timer.reset();
  //       n_tasks = 0;
  //     }
  //   }
  // }

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

      pipeann::Timer task_timer;
      reader->write(task->writes, ctx);
      uint64_t bytes_written = 0;
      for (const auto &req : task->writes) {
        bytes_written += req.len;
      }
      bg_bytes_written.fetch_add(bytes_written, std::memory_order_relaxed);
      bg_task_total_us.fetch_add(task_timer.elapsed(), std::memory_order_relaxed);
      aligned_free(task->thread_data->update_buf);
      task->thread_data->update_buf = nullptr;

      v2::unlockReqs(this->page_lock_table, task->pages_to_unlock);
      reader->deref(&task->pages_to_deref, ctx);
      this->push_query_buf(task->thread_data);
      delete task;
      bg_tasks_completed.fetch_add(1, std::memory_order_relaxed);
      uint64_t pending_after = bg_tasks_pending.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (pending_after == 0) {
        std::unique_lock<std::mutex> lk(bg_stats_mutex);
        bg_stats_cv.notify_all();
      }
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        auto current_depth = bg_tasks_pending.load(std::memory_order_relaxed) + bg_tasks.size();
        double throughput = static_cast<double>(n_tasks) * 1e6 / static_cast<double>(timer.elapsed());
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << throughput
                  << " tasks/sec. Pending: " << bg_tasks_pending.load(std::memory_order_relaxed)
                  << " Queue depth: " << current_depth;
        timer.reset();
        n_tasks = 0;
      }
    }
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
