#include "v2/dynamic_index.h"

#include <index.h>
#include <cstddef>
#include <future>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include "utils/timer.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MERGE_ROUND 20
#define MERGE_IO_THRESHOLD 1.2
#define NUM_INSERT_THREADS 10
#define NUM_MERGE_THREADS 20
#define NUM_DELETE_THREADS 64
#define NUM_SEARCH_THREADS 32
#define DeleteQPS 1000

int begin_time = 0;
pipeann::Timer globalTimer;

// acutually also shows disk size
void ShowMemoryStatus(const std::string &filename) {
  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  int tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;

  struct stat st;
  memset(&st, 0, sizeof(struct stat));
  std::string index_file_name = filename + "_disk.index";
  stat(index_file_name.c_str(), &st);

  std::cout << " memory current time: " << current_time << " RSS : " << rss << " KB " << index_file_name
            << " Index size " << (st.st_size / (1 << 20)) << " MB" << std::endl;
}

std::string convertFloatToString(const float value, const int precision = 0) {
  std::stringstream stream{};
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string GetTruthFileName(std::string &truthFilePrefix, int l_start) {
  std::string fileName(truthFilePrefix);
  fileName = fileName + "/gt_" + std::to_string(l_start) + ".bin";
  LOG(INFO) << "Truth file name: " << fileName;
  return fileName;
}

template<typename T>
inline uint64_t save_bin_test(const std::string &filename, T *id, float *dist, size_t npts, size_t ndims,
                              size_t offset = 0) {
  std::ofstream writer;
  open_file_to_write(writer, filename);

  std::cout << "Writing bin: " << filename.c_str() << std::endl;
  writer.seekp(offset, writer.beg);
  int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &ndims_i32, sizeof(int));
  std::cout << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B" << std::endl;

  for (int i = 0; i < npts; i++) {
    for (int j = 0; j < ndims; j++) {
      writer.write((char *) (id + i * ndims + j), sizeof(T));
      writer.write((char *) (dist + i * ndims + j), sizeof(float));
    }
  }
  writer.close();
  std::cout << "Finished writing bin." << std::endl;
  return bytes_written;
}

template<typename T, typename TagT>
void sync_search_kernel(T *query, size_t query_num, size_t query_dim, const int recall_at, uint64_t L,
                        uint32_t beam_width, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::string &truthset_file,
                        bool merged, bool calRecall, double &disk_io) {
  unsigned *gt_ids = NULL;
  float *gt_dists = NULL;
  size_t gt_num, gt_dim;

  if (!file_exists(truthset_file)) {
    calRecall = false;
  }

  if (calRecall) {
    std::cout << "current truthfile: " << truthset_file << std::endl;
    pipeann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
  }

  float *query_result_dists = new float[recall_at * query_num];
  TagT *query_result_tags = new TagT[recall_at * query_num];

  for (uint32_t q = 0; q < query_num; q++) {
    for (uint32_t r = 0; r < (uint32_t) recall_at; r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max();
    }
  }

  std::vector<double> latency_stats(query_num, 0);
  pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18) << "Mean Lat" << std::setw(12)
            << "50 Lat" << std::setw(12) << "90 Lat" << std::setw(12) << "95 Lat" << std::setw(12) << "99 Lat"
            << std::setw(12) << "99.9 Lat" << std::setw(12) << recall_string << std::setw(12) << "Disk IOs"
            << std::endl;
  std::cout << "==============================================================="
               "==============="
            << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS) schedule(dynamic)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    auto qs = std::chrono::high_resolution_clock::now();
    // stats[i].n_current_used = 8;
    sync_index.search(query + i * query_dim, recall_at, 0, L, beam_width, query_result_tags + i * recall_at,
                      query_result_dists + i * recall_at, stats + i, true);

    auto qe = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = qe - qs;
    latency_stats[i] = diff.count() * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float qps = (query_num / diff.count());
  float recall = 0;

  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
  if (calRecall) {
    recall = pipeann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim, query_result_tags, recall_at, recall_at);
    delete[] gt_ids;
  }

  std::cout << "search current time: " << current_time << std::endl;

  float mean_ios =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
            << ((float) std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0f)) / (float) query_num
            << std::setw(12) << (float) latency_stats[(uint64_t) (0.50 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.90 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.95 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.99 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.999 * ((double) query_num))] << std::setw(12) << recall
            << std::setw(12) << mean_ios << std::endl;
  disk_io = mean_ios;

  delete[] query_result_dists;
  delete[] query_result_tags;
  delete[] stats;
}

template<typename T, typename TagT>
void deletion_kernel(T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::vector<TagT> &delete_vec,
                     size_t aligned_dim) {
  pipeann::Timer timer;
  size_t npts = delete_vec.size();
  std::vector<double> delete_latencies(npts, 0);
  std::cout << "Begin Delete" << std::endl;
  for (int64_t i = 0; i < (int64_t) delete_vec.size(); i++) {
    sync_index.lazy_delete(delete_vec[i]);
  }
#pragma omp parallel for num_threads(NUM_DELETE_THREADS)
  for (int64_t i = 0; i < (int64_t) delete_vec.size(); i++) {
    // LOG(INFO) << i;
    pipeann::Timer delete_timer;
    pipeann::QueryStats stats;
    sync_index.wolverine_delete(delete_vec[i]);
    delete_latencies[i] = ((double) delete_timer.elapsed());
  }
  std::sort(delete_latencies.begin(), delete_latencies.end());
  std::cout << "10p deletion time : " << delete_latencies[(size_t) (0.10 * ((double) npts))] << " ms" << std::endl
            << "50p deletion time : " << delete_latencies[(size_t) (0.5 * ((double) npts))] << " ms" << std::endl
            << "90p deletion time : " << delete_latencies[(size_t) (0.90 * ((double) npts))] << " ms" << std::endl
            << "99p deletion time : " << delete_latencies[(size_t) (0.99 * ((double) npts))] << " ms" << std::endl
            << "99.9p deletion time : " << delete_latencies[(size_t) (0.999 * ((double) npts))] << " ms" << std::endl;
  sync_index.save_del_set();
}

template<typename T, typename TagT = uint32_t>
void get_trace(std::string data_bin, uint64_t l_start, uint64_t n, std::vector<TagT> &delete_tags) {
  for (uint64_t i = l_start; i < l_start + n; ++i) {
    delete_tags.push_back(i);
  }
}

template<typename T, typename TagT>
void update(const std::string &data_bin, const unsigned L_disk, int step, const unsigned nodes_to_cache,
            std::string &save_path, const std::string &query_file, std::string &truthset_file, const int recall_at,
            std::vector<uint64_t> Lsearch, const unsigned beam_width, pipeann::Distance<T> *dist_cmp) {
  pipeann::Parameters paras;
  paras.set(0, L_disk, 384, 1.2, 128, true, beam_width);  // 128 threads for less contention.
  std::vector<T> data_load;
  size_t dim{}, aligned_dim{};

  pipeann::Timer timer;

  std::cout << "Loading queries " << std::endl;
  T *query = NULL;
  size_t query_num, query_dim;
  pipeann::load_bin<T>(query_file, query, query_num, query_dim);

  dim = query_dim;
  aligned_dim = query_dim;
  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, TagT> sync_index(paras, save_path, save_path + "_merge", dist_cmp, metric);

  std::cout << "Searching before delete: " << std::endl;

  uint64_t res = 0;

  std::string currentFileName = GetTruthFileName(truthset_file, res);
  begin_time = globalTimer.elapsed() / 1.0e6f;
  ShowMemoryStatus(sync_index._disk_index_prefix_in);

  std::vector<double> ref_diskio;
  for (uint64_t j = 0; j < Lsearch.size(); ++j) {
    double diskio = 0;
    sync_search_kernel(query, query_num, query_dim, recall_at, Lsearch[j], beam_width, sync_index, currentFileName,
                       false, true, diskio);
    ref_diskio.push_back(diskio);
  }

  int batch = 10;
  // int batch = 1;
  int inMemorySize = 0;
  std::future<void> merge_future;
  uint64_t index_npts = sync_index._disk_index->num_points;
  // uint64_t vecs_per_step = index_npts / 10 / step;
  uint64_t vecs_per_step = 1000;
  for (int i = 0; i < batch; i++) {
    std::cout << "Batch: " << i << " Total Batch : " << batch << std::endl;
    std::vector<unsigned> delete_vec;

    /**Prepare for update*/
    uint64_t st = vecs_per_step * i;
    get_trace<T, TagT>(data_bin, st, vecs_per_step, delete_vec);

    std::future<void> delete_future = std::async(std::launch::async, deletion_kernel<T, TagT>, data_load.data(),
                                                 std::ref(sync_index), std::ref(delete_vec), aligned_dim);

    int total_queries = 0;
    std::future_status delete_status;
    do {
      delete_status = delete_future.wait_for(std::chrono::seconds(5));
      if ( delete_status == std::future_status::deferred) {
        std::cout << "deferred\n";
      } else if ( delete_status == std::future_status::timeout) {
        
      }
      if (delete_status == std::future_status::ready) {
        std::cout << "Deletions complete!\n";
      }
    } while ( delete_status != std::future_status::ready);

    ShowMemoryStatus(sync_index._disk_index_prefix_in);

    res += vecs_per_step;
    currentFileName = GetTruthFileName(truthset_file, res);
    std::cout << "Search after update, current vector number: " << res << std::endl;

    std::vector<double> disk_ios;
    for (uint64_t j = 0; j < Lsearch.size(); ++j) {
      double diskio = 0;
      sync_search_kernel(query, query_num, query_dim, recall_at, Lsearch[j], beam_width, sync_index, currentFileName,
                         false, true, diskio);
      disk_ios.push_back(diskio);
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 11) {
    std::cout << "Correct usage: " << argv[0] << " <type[int8/uint8/float]> <data_bin> <L_disk> "
              << " <indice_path> <query_file> <truthset_prefix> <recall@>"
              << " <#beam_width> <step> <Lsearch> <L2>" << std::endl;
    exit(-1);
  }

  int arg_no = 2;
  std::string data_bin = std::string(argv[arg_no++]);
  unsigned L_disk = (unsigned) atoi(argv[arg_no++]);
  std::string save_path(argv[arg_no++]);

  std::string query_file(argv[arg_no++]);
  std::string truthset(argv[arg_no++]);
  int recall_at = (int) std::atoi(argv[arg_no++]);
  unsigned beam_width = (unsigned) std::atoi(argv[arg_no++]);
  int step = (int) std::atoi(argv[arg_no++]);
  std::vector<uint64_t> Lsearch;
  for (int i = arg_no; i < argc; ++i) {
    Lsearch.push_back(std::atoi(argv[i]));
  }

  unsigned nodes_to_cache = 0;

  if (std::string(argv[1]) == std::string("int8")) {
    pipeann::DistanceL2Int8 dist_cmp;
    update<int8_t, unsigned>(data_bin, L_disk, step, nodes_to_cache, save_path, query_file, truthset, recall_at,
                             Lsearch, beam_width, &dist_cmp);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    pipeann::DistanceL2UInt8 dist_cmp;
    update<uint8_t, unsigned>(data_bin, L_disk, step, nodes_to_cache, save_path, query_file, truthset, recall_at,
                              Lsearch, beam_width, &dist_cmp);
  } else if (std::string(argv[1]) == std::string("float")) {
    pipeann::DistanceL2 dist_cmp;
    update<float, unsigned>(data_bin, L_disk, step, nodes_to_cache, save_path, query_file, truthset, recall_at, Lsearch,
                            beam_width, &dist_cmp);
  } else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}