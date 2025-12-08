#pragma once
#include <immintrin.h>
#include <cassert>
#include <cstdint>
#include <string>
#include <set>
#include "nbr/abstract_nbr.h"
#include <omp.h>
#include <atomic>  

#include "aligned_file_reader.h"
#include "utils/concurrent_queue.h"
#include "utils/lock_table.h"
#include "utils/percentile_stats.h"
#include "nbr/pq_nbr.h"
#include "utils.h"
#include "neighbor.h"
#include "index.h"
#include "v2/l1_neighbor_table.h"

#define MAX_N_CMPS 16384
#define MAX_N_EDGES 1024

constexpr int kIndexSizeFactor = 2;

enum SearchMode { BEAM_SEARCH = 0, PAGE_SEARCH = 1, PIPE_SEARCH = 2, CORO_SEARCH = 3 };

namespace v2 {
  class L1NeighborTable;   // 前向声明
}
namespace pipeann {

  template<typename T, typename TagT = uint32_t>
  class SSDIndex {
   public:
    SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &fileReader,
             AbstractNeighbor<T> *nbr = new PQNeighbor<T>(), bool tags = false, Parameters *parameters = nullptr);

    ~SSDIndex();

    // returns region of `node_buf` containing [COORD(T)]
    inline T *offset_to_node_coords(const char *node_buf) {
      return (T *) node_buf;
    }

    // returns region of `node_buf` containing [NNBRS][NBR_ID(uint32_t)]
    inline unsigned *offset_to_node_nhood(const char *node_buf) {
      return (unsigned *) (node_buf + data_dim * sizeof(T));
    }

    // obtains region of sector containing node
    inline char *offset_to_node(const char *sector_buf, uint32_t node_id) {
      return offset_to_loc(sector_buf, id2loc(node_id));
    }

    // sector # on disk where node_id is present
    inline uint64_t node_sector_no(uint32_t node_id) {
      return loc_sector_no(id2loc(node_id));
    }

    inline uint64_t u_node_offset(uint32_t node_id) {
      return u_loc_offset(id2loc(node_id));
    }

    // unaligned offset to location
    inline uint64_t u_loc_offset(uint64_t loc) {
      return loc * max_node_len;  // compacted store.
    }

    inline uint64_t u_loc_offset_nbr(uint64_t loc) {
      return loc * max_node_len + data_dim * sizeof(T);
    }

    inline char *offset_to_loc(const char *sector_buf, uint64_t loc) {
      return (char *) sector_buf + (nnodes_per_sector == 0 ? 0 : (loc % nnodes_per_sector) * max_node_len);
    }

    // avoid integer overflow when * SECTOR_LEN.
    inline uint64_t loc_sector_no(uint64_t loc) {
      return 1 + (nnodes_per_sector > 0 ? loc / nnodes_per_sector : loc * DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    }

    inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) {
      return nnodes_per_sector == 0 ? (sector_no - 1) / DIV_ROUND_UP(max_node_len, SECTOR_LEN)  // sector_off == 0.
                                    : (sector_no - 1) * nnodes_per_sector + sector_off;
    }

    void init_metadata(const SSDIndexMetadata<T> &meta) {
      meta.print();
      this->cur_id = this->num_points = this->init_num_pts = meta.npoints;
      this->data_dim = meta.data_dim;
      this->aligned_dim = ROUND_UP(this->data_dim, 8);
      this->range = meta.range;
      this->max_node_len = meta.max_node_len;
      this->nnodes_per_sector = meta.nnodes_per_sector;  // this makes reading to zero offset correct.
      this->size_per_io = SECTOR_LEN * (nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(max_node_len, SECTOR_LEN));
      LOG(INFO) << "Size per IO: " << size_per_io;

      this->cur_loc = num_points;
      // aligned.
      if (nnodes_per_sector != 0 && num_points % nnodes_per_sector != 0) {
        cur_loc += nnodes_per_sector - (num_points % nnodes_per_sector);
      }
      LOG(INFO) << "Cur location: " << this->cur_loc;

      // update-related metadata, if not initialized in params, initialize here.
      if (l_index < meta.range) {
        // experience values.
        LOG(INFO) << "Automatically set the update-related parameters.";
        this->l_index = meta.range + 32;
        this->beam_width = 4;
        this->maxc = 384;
        this->alpha = 1.2;
        LOG(INFO) << "L_index: " << this->l_index << " beam_width: " << this->beam_width << " maxc: " << this->maxc
                  << " alpha: " << this->alpha;
      }
      medoid = meta.entry_point;
    }

    void init_query_buf(QueryBuffer<T> &buf) {
      pipeann::alloc_aligned((void **) &buf.coord_scratch, this->aligned_dim * sizeof(T), 8 * sizeof(T));
      pipeann::alloc_aligned((void **) &buf.sector_scratch, MAX_N_SECTOR_READS * size_per_io, SECTOR_LEN);
      pipeann::alloc_aligned((void **) &buf.nbr_vec_scratch,
                             MAX_N_EDGES * AbstractNeighbor<T>::MAX_BYTES_PER_NBR * sizeof(uint8_t), 256);
      pipeann::alloc_aligned((void **) &buf.nbr_ctx_scratch, ROUND_UP(nbr_handler->query_ctx_size(), 256), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_dist_scratch, MAX_N_EDGES * sizeof(float), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_query_T, this->aligned_dim * sizeof(T), 8 * sizeof(T));

      buf.visited = new tsl::robin_set<uint64_t>(4096);
      buf.page_visited = new tsl::robin_set<unsigned>(4096);

      memset(buf.sector_scratch, 0, MAX_N_SECTOR_READS * SECTOR_LEN);
      memset(buf.coord_scratch, 0, this->aligned_dim * sizeof(T));
      memset(buf.aligned_query_T, 0, this->aligned_dim * sizeof(T));
    }

    QueryBuffer<T> *pop_query_buf(const T *query) {
      QueryBuffer<T> *data = this->thread_data_queue.pop();
      while (data == this->thread_data_queue.null_T) {
        this->thread_data_queue.wait_for_push_notify();
        data = this->thread_data_queue.pop();
      }

      if (likely(query != nullptr)) {
        if (data_is_normalized) {
          // Data has been normalized. Normalize search vector too.
          float norm = pipeann::compute_l2_norm(query, this->data_dim);
          for (uint32_t i = 0; i < this->data_dim; i++) {
            data->aligned_query_T[i] = query[i] / norm;
          }
        } else {
          memcpy(data->aligned_query_T, query, this->data_dim * sizeof(T));
        }
      }
      return data;
    }

    void push_query_buf(QueryBuffer<T> *data) {
      this->thread_data_queue.push(data);
      this->thread_data_queue.push_notify_all();
    }

    // load compressed data, and obtains the handle to the disk-resident index
    int load(const char *index_prefix, uint32_t num_threads, bool new_index_format = true,
             bool use_page_search = false);

    void load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path);

    // This function loads id2loc and loc2id (i.e., page_layout), to support index reordering.
    void load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector = 0,
                          const uint64_t num_points = 0);

    void load_tags(const std::string &tag_file, size_t offset = 0);

    uint64_t return_nd();

    // search supporting update.
    size_t beam_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr,
                       tsl::robin_set<uint32_t> *deleted_nodes = nullptr, bool dyn_search_l = true);

    size_t coro_search(T **queries, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT **res_tags, float **res_dists, const uint64_t beam_width, int N);

    // read-only search algorithms.
    size_t page_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    size_t pipe_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    // deflates `vec` into PQ ids
    std::vector<uint8_t> deflate_vector(const T *vec);

    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    uint64_t max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

   protected:
    void init_buffers(uint64_t nthreads);
    void destroy_buffers();

   public:
    // data info
    uint64_t num_points = 0;
    uint64_t init_num_pts = 0;
    uint64_t data_dim = 0;
    uint64_t aligned_dim = 0;
    uint64_t size_per_io = 0;

    std::string _disk_index_file;

    std::shared_ptr<AlignedFileReader> &reader;

    AbstractNeighbor<T> *nbr_handler;

    // distance comparator
    std::shared_ptr<Distance<T>> dist_cmp;

   public:
    // in-place update.
    int insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set = nullptr);
    void do_beam_search(const T *vec, uint32_t mem_L, uint32_t Lsize, const uint32_t beam_width,
                        std::vector<Neighbor> &expanded_nodes_info, tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
                        T *coord_buf = nullptr, QueryStats *stats = nullptr,
                        tsl::robin_set<uint32_t> *exclude_nodes = nullptr, bool dyn_search_l = true,
                        std::vector<uint64_t> *passthrough_page_ref = nullptr);
    void occlude_list(std::vector<Neighbor> &pool, const tsl::robin_map<uint32_t, T *> &coord_map,
                      std::vector<Neighbor> &result, std::vector<float> &occlude_factor);
    void prune_neighbors(const tsl::robin_map<uint32_t, T *> &coord_map, std::vector<Neighbor> &pool,
                         std::vector<uint32_t> &pruned_list);
    void occlude_list_pq(std::vector<Neighbor> &pool, std::vector<Neighbor> &result, std::vector<float> &occlude_factor,
                         uint8_t *scratch);
    void prune_neighbors_pq(std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list, uint8_t *scratch);

    void set_l1_table(v2::L1NeighborTable *t) {
      l1_table_ = t;
      if (l1_table_ != nullptr) {
        // 当某个 v 的 L1[v] 达到 fanout_B_ 时，L1NeighborTable 会调用这个回调；
        // 这里直接把 v 丢到 merge 队列里，后台 merge_worker_thread
        // 会按扇区聚合后调用 merge_nodes_on_sector。
        l1_table_->set_merge_hook(
            [this](uint32_t id) {
              this->enqueue_merge_node(id);
            });
      }
    }
    v2::L1NeighborTable *get_l1_table() const { return l1_table_; }

    // 增量图合并线程及函数
    void merge_worker_thread();
    void merge_nodes_on_sector(uint64_t sector,
                              const std::vector<uint32_t> &nodes,
                              void *ctx);
                              
    // 插入线程在 L1[v] 达到合并阈值时调用
    inline void enqueue_merge_node(uint32_t id) {
      // 统计一下 L1 触发合并的次数
      auto cnt = stats_l1_merge_triggers_.fetch_add(1) + 1;

      // 每 2^16 ≈ 6.5 万次触发打一次日志
      if ((cnt & ((1u << 16) - 1)) == 0) {
        LOG(INFO) << "[MERGE] enqueue_merge_node called " << cnt << " times so far.";
      }

      // 这里不直接操作队列，而是走封装好的 push，里面会负责 push_notify_all
      push(id);
    }


    // 轻剪 L1[v]：复用 delta_prune_neighbors_pq（PQ 路径）
    // center = v, new_id = 本次刚插入的 new 节点 id
    void prune_l1_delta(uint32_t center,
                        uint32_t new_id,
                        std::vector<uint32_t> &nbrs,
                        uint8_t *scratch);

    // delta pruning.
    struct TriangleNeighbor {
      unsigned id;
      float tgt_dis;
      float distance;

      inline bool operator<(const TriangleNeighbor &other) const {
        return (distance < other.distance) || (distance == other.distance && id < other.id);
      }
      inline bool operator==(const TriangleNeighbor &other) const {
        return (id == other.id);
      }
    };
    void delta_prune_neighbors_pq(std::vector<TriangleNeighbor> &pool, std::vector<uint32_t> &pruned_list,
                                  uint8_t *scratch, int tgt_idx);
    void reload(const char *index_prefix, uint32_t num_threads);


    // background I/O commit.
    struct BgTask {
      QueryBuffer<T> *thread_data;
      std::vector<IORequest> writes;
      std::vector<uint64_t> pages_to_unlock;
      std::vector<uint64_t> pages_to_deref;
      bool terminate = false;
    };
    // its concurrency should not be the bottleneck.
    ConcurrentQueue<BgTask *> bg_tasks = ConcurrentQueue<BgTask *>(nullptr);
    void bg_io_thread();
    static constexpr int kBgIOThreads = 1;
    std::thread *bg_io_thread_[kBgIOThreads]{nullptr};

    // test the estimation efficacy.
    uint32_t beam_width, l_index, range, maxc;
    float alpha;
    // assumed max thread, only the first nthreads are initialized.
    v2::SparseLockTable<uint64_t> page_lock_table, vec_lock_table, page_idx_lock_table, idx_lock_table;
    std::shared_mutex merge_lock;  // serve search during merge.

    // if ID == tag, then it is not stored.
    libcuckoo::cuckoohash_map<uint32_t, TagT> tags;
    TagT id2tag(uint32_t id) {
#ifdef NO_MAPPING
      return id;  // use ID to replace tags.
#else
      TagT ret;
      if (tags.find(id, ret)) {
        return ret;
      } else {
        return id;
      }
#endif
    }

    int get_vector_by_id(const uint32_t &id, T *vector);
    static constexpr uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kAllocatedID = std::numeric_limits<uint32_t>::max() - 1;

    void lock_vec(v2::SparseLockTable<uint64_t> &lock_table, uint32_t target, const std::vector<uint32_t> &neighbors,
                  bool rd = false) {
      std::vector<uint32_t> to_lock;
      to_lock.assign(neighbors.begin(), neighbors.end());
      if (target != kInvalidID)
        to_lock.push_back(target);
      std::sort(to_lock.begin(), to_lock.end());
      for (auto &id : to_lock) {
        rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
      }
    }

    void unlock_vec(v2::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                    const std::vector<uint32_t> &neighbors) {
      if (target != kInvalidID)
        lock_table.unlock(target);
      for (auto &id : neighbors) {
        lock_table.unlock(id);
      }
    }

   private:
    std::vector<uint32_t> get_to_lock_idx(uint32_t target, const std::vector<uint32_t> &neighbors) {
      std::vector<uint32_t> to_lock;
      to_lock.assign(neighbors.begin(), neighbors.end());
      if (target != kInvalidID) {
        to_lock.push_back(target);
      }
      // sort and deduplicate
      std::sort(to_lock.begin(), to_lock.end());
      auto last = std::unique(to_lock.begin(), to_lock.end());
      to_lock.erase(last, to_lock.end());
      return to_lock;
    }

   public:
    // lock the mapping for target/page if use_page_search == false/true.
    std::vector<uint32_t> lock_idx(v2::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                   const std::vector<uint32_t> &neighbors, bool rd = false) {
#ifndef READ_ONLY_TESTS
      std::vector<uint32_t> to_lock = get_to_lock_idx(target, neighbors);
      for (auto &id : to_lock) {
        rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
      }
      return to_lock;
#else
      return std::vector<uint32_t>();
#endif
    }

    void unlock_idx(v2::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock) {
#ifndef READ_ONLY_TESTS
      for (auto &id : to_lock) {
        lock_table.unlock(id);
      }
#endif
    }

    void unlock_idx(v2::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock) {
#ifndef READ_ONLY_TESTS
      lock_table.unlock(to_lock);
#endif
    }

    // two-level, as id2page may change before and after grabbing the lock.
    std::vector<uint32_t> lock_page_idx(v2::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                        const std::vector<uint32_t> &neighbors, bool rd = false) {
#ifndef READ_ONLY_TESTS
      if (!use_page_search_) {
        return std::vector<uint32_t>();
      }
      std::vector<uint32_t> to_lock(neighbors.begin(), neighbors.end());
      if (target != kInvalidID) {
        to_lock.push_back(target);
      }

      for (size_t i = 0; i < to_lock.size(); ++i) {
        to_lock[i] = id2page(to_lock[i]);
      }

      // sort and deduplicate
      std::sort(to_lock.begin(), to_lock.end());
      auto last = std::unique(to_lock.begin(), to_lock.end());
      to_lock.erase(last, to_lock.end());

      for (auto &id : to_lock) {
        rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
      }
      return to_lock;
#else
      return std::vector<uint32_t>();
#endif
    }

    void unlock_page_idx(v2::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock) {
#ifndef READ_ONLY_TESTS
      if (!use_page_search_) {
        return;
      }
      for (auto &id : to_lock) {
        lock_table.unlock(id);
      }
#endif
    }

    // in-memory navigation graph
    std::unique_ptr<Index<T, uint32_t>> mem_index_;

    // page search
    bool use_page_search_ = true;

    // Concurrency control is done in lock_idx.
    // Only resize should be protected.
    std::vector<uint32_t> id2loc_;
    v2::ReaderOptSharedMutex id2loc_resize_mu_;

    uint32_t id2loc(uint32_t id) {
#ifdef NO_MAPPING
      return id;
#else
      id2loc_resize_mu_.lock_shared();
      if (unlikely(id >= id2loc_.size())) {
        LOG(ERROR) << "id " << id << " is out of range " << id2loc_.size();
        crash();
        return kInvalidID;
      }
      uint32_t ret = id2loc_[id];
      id2loc_resize_mu_.unlock_shared();
      return ret;
#endif
    }

    void set_id2loc(uint32_t id, uint32_t loc) {
#ifdef NO_MAPPING
      return;
#else
      if (unlikely(id >= id2loc_.size())) {
        id2loc_resize_mu_.lock();
        if (likely(id >= id2loc_.size())) {
          id2loc_.resize(1.5 * id);
          LOG(INFO) << "Resize id2loc_ to " << id2loc_.size();
        }
        id2loc_resize_mu_.unlock();
      }
      // Here, we do not grab any locks. But no matter:
      // Here, the id2loc_.size() must > id (as it only increases).
      // So, we only need to ensure that no concurrent resize happens (use-after-free).
      id2loc_resize_mu_.lock_shared();
      id2loc_[id] = loc;
      id2loc_resize_mu_.unlock_shared();
#endif
    }

    uint64_t id2page(uint32_t id) {
      uint32_t loc = id2loc(id);
      if (loc == kInvalidID) {
        return kInvalidID;
      }
      return loc_sector_no(loc);
    }

    // If nnodes_per_sector >= 1, page_layout[i * nnodes_per_sector + j] is the id of the j-th node in the i-th page.
    // ElseIf nnodes_per_sector == 0, page_layout[i] is the id of the i-th node (starting from loc_sector_no(i)).
    std::vector<uint32_t> loc2id_;
    v2::ReaderOptSharedMutex loc2id_resize_mu_;
    std::mutex alloc_lock;
    ConcurrentQueue<uint32_t> empty_pages = ConcurrentQueue<uint32_t>(kInvalidID);

    using PageArr = std::vector<uint32_t>;

    PageArr get_page_layout(uint32_t page_no) {
      loc2id_resize_mu_.lock_shared();
      PageArr ret;
      auto st = sector_to_loc(page_no, 0);
      auto ed = nnodes_per_sector == 0 ? st + 1 : st + nnodes_per_sector;
      for (uint32_t i = st; i < ed; ++i) {
        ret.push_back(loc2id_[i]);
      }
      loc2id_resize_mu_.unlock_shared();
      return ret;
    }

    uint32_t loc2id(uint32_t loc) {
      loc2id_resize_mu_.lock_shared();
      if (unlikely(loc > loc2id_.size())) {
        LOG(ERROR) << "loc " << loc << " is out of range " << loc2id_.size();
        crash();
        return kInvalidID;
      }
      uint32_t ret = loc2id_[loc];
      loc2id_resize_mu_.unlock_shared();
      return ret;
    }

    void set_loc2id(uint32_t loc, uint32_t id) {
      if (unlikely(loc >= loc2id_.size())) {
        loc2id_resize_mu_.lock();
        if (likely(loc >= loc2id_.size())) {
          loc2id_.resize(1.5 * loc);
          LOG(INFO) << "Resize loc2id_ to " << loc2id_.size();
        }
        loc2id_resize_mu_.unlock();
      }
      loc2id_resize_mu_.lock_shared();
      loc2id_[loc] = id;
      loc2id_resize_mu_.unlock_shared();
    }

    void erase_loc2id(uint32_t loc) {
      loc2id_resize_mu_.lock_shared();
      loc2id_[loc] = kInvalidID;
      uint32_t st = sector_to_loc(loc_sector_no(loc), 0);
      bool empty = true;
      for (uint32_t i = st; i < st + nnodes_per_sector; ++i) {
        if (loc2id_[i] != kInvalidID) {
          empty = false;
          break;
        }
      }
      if (empty) {
        empty_pages.push(loc_sector_no(loc));
      }
      uint32_t page = loc_sector_no(loc);
      uint32_t offset = loc % nnodes_per_sector;
      loc2id_resize_mu_.unlock_shared();
    }

    void erase_and_set_loc(const std::vector<uint64_t> &old_locs, const std::vector<uint64_t> &new_locs,
                           const std::vector<uint32_t> &new_ids) {
      std::lock_guard<std::mutex> lock(alloc_lock);
      for (uint32_t i = 0; i < new_locs.size(); ++i) {
        set_loc2id(new_locs[i], new_ids[i]);
      }
      for (auto &l : old_locs) {
        erase_loc2id(l);
      }
    }

    // returns <loc, need_read>
    std::vector<uint64_t> alloc_loc(int n, const std::vector<uint64_t> &hint_pages,
                                    std::set<uint64_t> &page_need_to_read) {
      std::lock_guard<std::mutex> lock(alloc_lock);
      std::vector<uint64_t> ret;
      int cur = 0;
      // reuse.
      uint32_t threshold = (nnodes_per_sector + kIndexSizeFactor - 1) / kIndexSizeFactor;

      // 1. use empty pages.
      uint32_t empty_page = kInvalidID;
      while ((empty_page = empty_pages.pop()) != kInvalidID) {
#ifdef NO_POLLUTE_ORIGINAL
        if (empty_page < loc_sector_no(init_num_pts)) {
          continue;
        }
#endif
        auto st = sector_to_loc(empty_page, 0);
        auto ed = nnodes_per_sector == 0 ? st + 1 : st + nnodes_per_sector;
        for (uint32_t i = st; i < ed; ++i) {
          if (unlikely(loc2id_[i] != kInvalidID)) {
            LOG(ERROR) << "Page " << empty_page << " is not empty " << i << " " << loc2id_[i];
            crash();
          }
          loc2id_[i] = kAllocatedID;
          ret.push_back(i);
          ++cur;
          if (cur == n) {
            return ret;
          }
        }
      }

      // 2. use hint pages.
      for (auto &p : hint_pages) {
#ifdef NO_POLLUTE_ORIGINAL
        if (p < loc_sector_no(init_num_pts)) {
          continue;
        }
#endif
        // first, see the number of holes.
        uint32_t cnt = 0;
        auto st = sector_to_loc(p, 0);
        auto ed = nnodes_per_sector == 0 ? st + 1 : st + nnodes_per_sector;
        for (uint32_t i = st; i < ed; ++i) {
          if (loc2id_[i] == kInvalidID) {
            cnt++;
          }
        }
        if (cnt < threshold) {
          continue;
        }
        // second, allocate them.
        if (cnt < nnodes_per_sector) {
          page_need_to_read.insert(p);
        }
        for (uint32_t i = st; i < ed; ++i) {
          if (loc2id_[i] == kInvalidID) {
            loc2id_[i] = kAllocatedID;
            ret.push_back(i);
            ++cur;
            if (cur == n) {
              return ret;
            }
          }
        }
      }

      // 3. use new pages.
      int remaining = n - cur;
      for (int i = 0; i < remaining; i++) {
        set_loc2id(cur_loc + i, kAllocatedID);  // auto resize.
        ret.push_back(cur_loc + i);
      }

      // ensure that cur_loc is aligned.
      // the hole will eventually be recycled using either empty page queue or hint pages.
      cur_loc += remaining;
      while (nnodes_per_sector != 0 && cur_loc % nnodes_per_sector != 0) {
        set_loc2id(cur_loc++, kInvalidID);  // auto resize.
      }
      return ret;
    }

    void verify_id2loc() {
      // verify id -> loc -> id map.
      LOG(INFO) << "ID2loc size: " << id2loc_.size() << ", cur_loc: " << cur_loc.load() << ", cur_id: " << cur_id
                << ", nnodes_per_sector: " << nnodes_per_sector;
      for (uint32_t i = 0; i < cur_id; ++i) {
        auto loc = id2loc(i);
        if (unlikely(loc >= cur_loc.load())) {
          LOG(ERROR) << "ID2loc inconsistency at ID: " << i << ", loc: " << loc << ", cur_loc: " << cur_loc.load();
          crash();
        }
        if (unlikely(loc2id(loc) != i)) {
          LOG(ERROR) << "ID2loc inconsistency at ID: " << i << ", loc: " << id2loc(i)
                     << ", loc2id: " << loc2id(id2loc(i));
          crash();
        }
      }

      LOG(INFO) << "ID2loc consistency check passed.";

      // verify loc2id do not contain duplicate ids.
      for (uint32_t i = 0; i < cur_loc; ++i) {
        auto id = loc2id(i);
        if (id != kInvalidID && id != kAllocatedID) {
          uint32_t loc = id2loc(id);
          if (unlikely(loc != i)) {
            LOG(ERROR) << "loc2id inconsistency at loc: " << i << ", id: " << id << ", loc: " << loc;
          }
        }
      }
      LOG(INFO) << "loc2ID consistency check passed.";
    }
    std::atomic<uint64_t> cur_id, cur_loc;

    // merge deletes (NOTE: index read-only during merge.)
    void merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                       const std::vector<TagT> &deleted_nodes, const tsl::robin_set<TagT> &deleted_nodes_set,
                       uint32_t nthreads, const uint32_t &n_sampled_nbrs);

    void write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                               const uint64_t &new_npoints, const uint64_t &new_medoid,
                               std::vector<TagT> *new_tags = nullptr);
    void copy_index(const std::string &prefix_in, const std::string &prefix_out);

   private:
    // Are we dealing with normalized data? This will be true
    // if distance == COSINE and datatype == float. Required
    // because going forward, we will normalize vectors when
    // asked to search with COSINE similarity. Of course, this
    // will be done only for floating point vectors.
    v2::L1NeighborTable *l1_table_ = nullptr;   // 默认没有 L1

    // 用于触发 L1 合并的队列：插入线程 push 节点 id，后台线程 pop
    void push(uint32_t);
    bool try_pop(uint32_t &);      // 非阻塞
    void pop_blocking(uint32_t &); // 阻塞直到拿到一个 id

    // merge 队列的“空值”，不能作为真实的节点 id 使用
    static constexpr uint32_t MERGE_NULL_ID      = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t MERGE_TERMINATE_ID = std::numeric_limits<uint32_t>::max() - 1;

    // 用 ConcurrentQueue 实现的阻塞队列
    ConcurrentQueue<uint32_t> merge_queue_;

    // 结束标志，析构时用于通知 merge 线程退出
    std::atomic<bool> merge_stop_{false};

    // ---------- merge 调试统计 ----------
    // L1 触发合并（enqueue_merge_node 被调用）的总次数
    std::atomic<uint64_t> stats_l1_merge_triggers_{0};
    // merge_worker_thread 从队列成功 pop 出来的 id 次数
    std::atomic<uint64_t> stats_merge_pop_{0};
    // 去重后真正参与 merge_nodes_on_sector 的节点个数累积和
    std::atomic<uint64_t> stats_merge_nodes_merged_{0};

    bool data_is_normalized = false;

    // medoid/start info
    uint32_t medoid = 0;  // 1 entry point.

    // thread-specific scratch
    ConcurrentQueue<QueryBuffer<T> *> thread_data_queue;
    std::vector<QueryBuffer<T> *> thread_data_bufs;  // pre-allocated thread data
    uint64_t max_nthreads;

    bool load_flag = false;    // already loaded.
    bool enable_tags = false;  // support for tags and dynamic indexing
  };
}  // namespace pipeann
