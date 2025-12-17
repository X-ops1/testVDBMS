#include <math_utils.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#include "utils/cached_io.h"
#include "index.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include "utils/tsl/robin_map.h"

#include <cassert>
#include "partition.h"
#include "partition_tree.h"
#include "knn.h"

#define MAX_BLOCK_SIZE 16384  // 64MB for 1024-dim float vectors, 2MB for 128-dim uint8 vectors.

template<typename T>
void gen_random_slice(const std::string base_file, const std::string output_prefix, double sampling_rate,
                      size_t offset) {
    std::ifstream base_reader(base_file.c_str());
    base_reader.seekg(offset, std::ios::beg);

    std::ofstream sample_writer(std::string(output_prefix + "_data.bin").c_str(), std::ios::binary);
    std::ofstream sample_id_writer(std::string(output_prefix + "_ids.bin").c_str(), std::ios::binary);

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    auto x = rd();
    std::mt19937 generator(x);  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> distribution(0, 1);

    size_t npts, nd;
    uint32_t npts_u32, nd_u32;
    uint32_t num_sampled_pts_u32 = 0;
    uint32_t one_const = 1;

    base_reader.read((char *) &npts_u32, sizeof(uint32_t));
    base_reader.read((char *) &nd_u32, sizeof(uint32_t));
    LOG(INFO) << "Loading base " << base_file << ". #points: " << npts_u32 << ". #dim: " << nd_u32 << ".";
    sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
    sample_writer.write((char *) &nd_u32, sizeof(uint32_t));
    sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
    sample_id_writer.write((char *) &one_const, sizeof(uint32_t));

    npts = npts_u32;
    nd = nd_u32;
    std::unique_ptr<T[]> cur_row = std::make_unique<T[]>(nd);

    for (size_t i = 0; i < npts; i++) {
        float sample = distribution(generator);
        if (sample < (float) sampling_rate) {
            base_reader.read((char *) cur_row.get(), sizeof(T) * nd);
            sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
            uint32_t cur_i_u32 = (uint32_t) i;
            sample_id_writer.write((char *) &cur_i_u32, sizeof(uint32_t));
            num_sampled_pts_u32++;
        } else {
            base_reader.seekg(sizeof(T) * nd, base_reader.cur);  // skip this vector
        }
    }

    if (num_sampled_pts_u32 == 0) {
        // We have read something from file, so write it.
        sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
        num_sampled_pts_u32 = 1;
    }
    sample_writer.seekp(0, std::ios::beg);
    sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
    sample_id_writer.seekp(0, std::ios::beg);
    sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
    sample_writer.close();
    sample_id_writer.close();
    LOG(INFO) << "Wrote " << num_sampled_pts_u32 << " points to sample file: " << output_prefix + "_data.bin";
}

// streams data from the file, and samples each vector with probability p_val
// and returns a matrix of size slice_size* ndims as floating point type.
// the slice_size and ndims are set inside the function.

template<typename T>
void gen_random_slice(const std::string data_file, double p_val, std::unique_ptr<float[]> &sampled_data,
                      size_t &slice_size, size_t &ndims) {
    float *sampled_ptr = sampled_data.get();
    gen_random_slice<T>(data_file, p_val, sampled_ptr, slice_size, ndims);
    sampled_data.reset(sampled_ptr);
}

template<typename T>
void gen_random_slice(const std::string data_file, double p_val, float *&sampled_data, size_t &slice_size,
                      size_t &ndims) {
    size_t npts;
    uint32_t npts32, ndims32;
    std::vector<std::vector<float>> sampled_vectors;

    // amount to read in one shot
    uint64_t read_blk_size = 64 * 1024 * 1024;
    std::ifstream base_reader(data_file.c_str());

    // metadata: npts, ndims
    base_reader.read((char *) &npts32, sizeof(unsigned));
    base_reader.read((char *) &ndims32, sizeof(unsigned));
    npts = npts32;
    ndims = ndims32;

    std::unique_ptr<T[]> cur_vector_T = std::make_unique<T[]>(ndims);
    p_val = p_val < 1 ? p_val : 1;

    std::random_device rd;  // Will be used to obtain a seed for the random number
    size_t x = rd();
    std::mt19937 generator((unsigned) x);
    std::uniform_real_distribution<float> distribution(0, 1);

    for (size_t i = 0; i < npts; i++) {
        float rnd_val = distribution(generator);
        if (rnd_val < (float) p_val) {
            base_reader.read((char *) cur_vector_T.get(), ndims * sizeof(T));
            std::vector<float> cur_vector_float;
            for (size_t d = 0; d < ndims; d++)
                cur_vector_float.push_back(cur_vector_T[d]);
            sampled_vectors.push_back(cur_vector_float);
        } else {
            base_reader.seekg(ndims * sizeof(T), base_reader.cur);  // skip this vector
        }
    }
    slice_size = sampled_vectors.size();
    if (slice_size == 0) {
        slice_size = 1;
        std::vector<float> cur_vector_float(cur_vector_T.get(), cur_vector_T.get() + ndims);
        sampled_vectors.push_back(cur_vector_float);
    }
    sampled_data = new float[slice_size * ndims];

    for (size_t i = 0; i < slice_size; i++) {
        for (size_t j = 0; j < ndims; j++) {
            sampled_data[i * ndims + j] = sampled_vectors[i][j];
        }
    }
}

template<typename T>
int estimate_cluster_sizes(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                           const size_t k_base, std::vector<size_t> &cluster_sizes) {
    cluster_sizes.clear();

    size_t num_test, test_dim;
    float *test_data_float;
    double sampling_rate = 0.01;

    gen_random_slice<T>(data_file, sampling_rate, test_data_float, num_test, test_dim);

    if (test_dim != dim) {
        LOG(INFO) << "Error. dimensions dont match for pivot set and base set";
        return -1;
    }

    size_t *shard_counts = new size_t[num_centers];

    for (size_t i = 0; i < num_centers; i++) {
        shard_counts[i] = 0;
    }

    size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_test);
    size_t num_points = 0, num_dim = 0;
    pipeann::get_bin_metadata(data_file, num_points, num_dim);
    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    uint32_t *block_closest_centers = new uint32_t[block_size * k_base];
    float *block_data_float;

    size_t num_blocks = DIV_ROUND_UP(num_test, block_size);

    for (size_t block = 0; block < num_blocks; block++) {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_test);
        size_t cur_blk_size = end_id - start_id;

        block_data_float = test_data_float + start_id * test_dim;

        math_utils::compute_closest_centers(block_data_float, cur_blk_size, dim, pivots, num_centers, k_base,
                                            block_closest_centers);

        for (size_t p = 0; p < cur_blk_size; p++) {
            for (size_t p1 = 0; p1 < k_base; p1++) {
                size_t shard_id = block_closest_centers[p * k_base + p1];
                shard_counts[shard_id]++;
            }
        }
    }

    LOG(INFO) << "Estimated cluster sizes: ";
    for (size_t i = 0; i < num_centers; i++) {
        uint32_t cur_shard_count = (uint32_t) shard_counts[i];
        cluster_sizes.push_back(size_t(((double) cur_shard_count) * (1.0 / sampling_rate)));
        std::cerr << cur_shard_count * (1.0 / sampling_rate) << " ";
    }
    std::cerr << "\n";
    delete[] shard_counts;
    delete[] block_closest_centers;
    return 0;
}

template<typename T>
int shard_data_into_clusters(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                             const size_t k_base, std::string prefix_path) {
    uint64_t read_blk_size = 64 * 1024 * 1024;
    //  uint64_t write_blk_size = 64 * 1024 * 1024;
    // create cached reader + writer
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *) &npts32, sizeof(uint32_t));
    base_reader.read((char *) &basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    if (basedim32 != dim) {
        LOG(INFO) << "Error. dimensions dont match for train set and base set";
        return -1;
    }

    std::unique_ptr<size_t[]> shard_counts = std::make_unique<size_t[]>(num_centers);
    std::vector<std::ofstream> shard_data_writer(num_centers);
    std::vector<std::ofstream> shard_idmap_writer(num_centers);
    uint32_t dummy_size = 0;
    uint32_t const_one = 1;

    for (size_t i = 0; i < num_centers; i++) {
        std::string data_filename = prefix_path + "_subshard-" + std::to_string(i) + ".bin";
        std::string idmap_filename = prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
        shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
        shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
        shard_data_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
        shard_data_writer[i].write((char *) &basedim32, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *) &const_one, sizeof(uint32_t));
        shard_counts[i] = 0;
    }

    size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_points);
    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size * k_base);
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    for (size_t block = 0; block < num_blocks; block++) {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *) block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
        pipeann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

        math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
                                            block_closest_centers.get());

        for (size_t p = 0; p < cur_blk_size; p++) {
            for (size_t p1 = 0; p1 < k_base; p1++) {
                size_t shard_id = block_closest_centers[p * k_base + p1];
                uint32_t original_point_map_id = (uint32_t) (start_id + p);
                shard_data_writer[shard_id].write((char *) (block_data_T.get() + p * dim), sizeof(T) * dim);
                shard_idmap_writer[shard_id].write((char *) &original_point_map_id, sizeof(uint32_t));
                shard_counts[shard_id]++;
            }
        }
    }

    size_t total_count = 0;
    LOG(INFO) << "Actual shard sizes: ";
    for (size_t i = 0; i < num_centers; i++) {
        uint32_t cur_shard_count = (uint32_t) shard_counts[i];
        total_count += cur_shard_count;
        LOG(INFO) << cur_shard_count << " ";
        shard_data_writer[i].seekp(0);
        shard_data_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
        shard_data_writer[i].close();
        shard_idmap_writer[i].seekp(0);
        shard_idmap_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
        shard_idmap_writer[i].close();
    }

    LOG(INFO) << "\n Partitioned " << num_points << " with replication factor " << k_base << " to get " << total_count
              << " points across " << num_centers << " shards ";
    return 0;
}

template<typename T>
int partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                              size_t graph_degree, const std::string prefix_path, size_t k_base) {
    size_t train_dim;
    size_t num_train;
    float *train_data_float;
    size_t max_k_means_reps = 20;

    int num_parts = 3;
    bool fit_in_ram = false;

    gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train, train_dim);

    float *pivot_data = nullptr;

    std::string cur_file = std::string(prefix_path);
    std::string output_file;

    // kmeans_partitioning on training data

    //  cur_file = cur_file + "_kmeans_partitioning-" + std::to_string(num_parts);
    output_file = cur_file + "_centroids.bin";

    while (!fit_in_ram) {
        fit_in_ram = true;

        double max_ram_usage = 0;
        if (pivot_data != nullptr)
            delete[] pivot_data;

        pivot_data = new float[num_parts * train_dim];
        // Process Global k-means for kmeans_partitioning Step
        LOG(INFO) << "Processing global k-means (kmeans_partitioning Step)";
        kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim, pivot_data, num_parts);

        kmeans::run_elkan(train_data_float, num_train, train_dim, pivot_data, num_parts, max_k_means_reps, NULL, NULL);

        // now pivots are ready. need to stream base points and assign them to
        // closest clusters.

        std::vector<size_t> cluster_sizes;
        estimate_cluster_sizes<T>(data_file, pivot_data, num_parts, train_dim, k_base, cluster_sizes);

        for (auto &p: cluster_sizes) {
            double cur_shard_ram_estimate = pipeann::estimate_ram_usage(p, train_dim, sizeof(T), graph_degree);

            if (cur_shard_ram_estimate > max_ram_usage)
                max_ram_usage = cur_shard_ram_estimate;
        }
        LOG(INFO) << "With " << num_parts << " parts, max estimated RAM usage: " << max_ram_usage / (1024 * 1024 * 1024)
                  << "GB, budget given is " << ram_budget;
        if (max_ram_usage > 1024 * 1024 * 1024 * ram_budget) {
            fit_in_ram = false;
            num_parts++;
        }
    }

    LOG(INFO) << "Saving global k-center pivots";
    pipeann::save_bin<float>(output_file.c_str(), pivot_data, (size_t) num_parts, train_dim);

    shard_data_into_clusters<T>(data_file, pivot_data, num_parts, train_dim, k_base, prefix_path);
    delete[] pivot_data;
    delete[] train_data_float;
    return num_parts;
}

template<typename T>
float split_into_two_clusters(float *data, float *pivots, const size_t num, const size_t dim,
                              std::vector<size_t> &cluster_sizes, std::vector<float *> &cluster_data, int level,
                              int nid) {
    std::unique_ptr<size_t[]> shard_counts = std::make_unique<size_t[]>(2);
    for (size_t i = 0; i < 2; i++) {
        shard_counts[i] = 0;
    }

    size_t num_boundary = 0;
    std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(num * 2);
    std::unique_ptr<float[]> beta = std::make_unique<float[]>(num);

    std::vector<std::pair<float, unsigned>> ratios;

    math_utils::compute_closest_centers(data, num, dim, pivots, 2, 2, block_closest_centers.get(), beta.get());

    for (size_t p = 0; p < num; p++) {
        size_t shard_id = block_closest_centers[p * 2];
        shard_counts[shard_id]++;
    }

    std::vector<int> flags(num, 0);

    uint32_t k = 16;
//  float theta_deg = 40.0f;
//  float theta_rad = theta_deg * M_PI / 180.0f;
//  float cos_threshold = std::cos(theta_rad);
    int *closest_points = new int[num * k];
    float *dist_closest_points = new float[num * k];
    exact_knn(dim, k, closest_points, dist_closest_points, num, data, num, data);
    std::vector<std::vector<int>> cluster_neighbor_size(num, std::vector<int>(2, 0));
    for (uint32_t i = 0; i < num; i++) {
        uint32_t closest_cluster_id = block_closest_centers[i * 2];
//    float theda = math_utils::calc_angle(pivots, pivots+dim, data + i*dim, dim);
//    if(theda > cos_threshold){
//      continue;
//    }
        for (uint32_t j = 0; j < k; j++) {
            uint32_t neighbor = closest_points[i * k + j];
            uint32_t cluster_id = block_closest_centers[neighbor * 2];
            cluster_neighbor_size[i][cluster_id]++;
        }
        ratios.emplace_back(beta[i], i);
        float other_ratio = (float) cluster_neighbor_size[i][closest_cluster_id] / (float) k;
        if (other_ratio <= 0.7f) {
            num_boundary++;
        }
    }
    std::cout << num_boundary << std::endl;

    if (ratios.size() > 0) {
        std::nth_element(ratios.begin(), ratios.begin() + num_boundary, ratios.end());
        std::sort(ratios.begin(), ratios.begin() + num_boundary,
                  [](const auto &a, const auto &b) { return a.second < b.second; });
    }
    for (size_t p = 0; p < num_boundary; p++) {
        size_t id = ratios[p].second;
        size_t shard_id = block_closest_centers[id * 2 + 1];
        shard_counts[shard_id]++;
    }
    for (size_t i = 0; i < 2; i++) {
        auto temp = new float[shard_counts[i] * dim];
        cluster_data[i] = temp;
        cluster_sizes[i] = shard_counts[i];
    }

    int cur_pos = 0;
    int first_pos = 0;
    int second_pos = 0;
    for (size_t p = 0; p < num; p++) {
        size_t shard_id = block_closest_centers[p * 2];
        int start = (shard_id == 0) ? first_pos++ : second_pos++;
        for (size_t i = 0; i < dim; i++) {
            cluster_data[shard_id][start * dim + i] = data[p * dim + i];
        }

        if (ratios.size() == 0 || p < ratios[cur_pos].second) {
            continue;
        }
        shard_id = block_closest_centers[p * 2 + 1];
        start = (shard_id == 0) ? first_pos++ : second_pos++;
        for (size_t i = 0; i < dim; i++) {
            cluster_data[shard_id][start * dim + i] = data[p * dim + i];
        }
        cur_pos++;
    }

    delete[] closest_points;
    delete[] dist_closest_points;
    return (ratios.size() > 0) ? ratios[num_boundary].first : 0.0f;
}

template<typename T>
void create_partition_tree(float *data, size_t num, size_t dim, const double sampling_rate, double budget,
                           size_t graph_degree, partitionNode *root) {
    assert(!root->check_leaf());
    size_t max_k_means_reps = 20;
    std::vector<float> pivot_data;
    pivot_data.resize(2 * dim);

    LOG(INFO) << "Processing global k-means (kmeans_partitioning Step)";
    kmeans::kmeanspp_selecting_pivots(data, num, dim, pivot_data.data(), 2);

    kmeans::run_elkan(data, num, dim, pivot_data.data(), 2, max_k_means_reps, NULL, NULL);

    std::vector<size_t> cluster_sizes(2);
    std::vector<float *> cluster_data(2);

    float boundary_factor = split_into_two_clusters<T>(data, pivot_data.data(), num, dim, cluster_sizes, cluster_data,
                                                       root->get_level(), root->get_id());

    partitionNode *left_node = nullptr;
    partitionNode *right_node = nullptr;

    {
        double cur_shard_ram_estimate =
                pipeann::estimate_ram_usage(cluster_sizes[0] * (1.0 / sampling_rate), dim, sizeof(T), graph_degree);
        int id = root->get_id() * 2;
        int level = root->get_level() + 1;

        if (cur_shard_ram_estimate > budget) {
            left_node = new partitionNode(false, level, id);
            create_partition_tree<T>(cluster_data[0], cluster_sizes[0], dim, sampling_rate, budget, graph_degree,
                                     left_node);
        } else {
            left_node = new partitionNode(true, level, id);
            left_node->make_leaf(cluster_sizes[0]);
            delete[] cluster_data[0];
        }
    }

    {
        double cur_shard_ram_estimate =
                pipeann::estimate_ram_usage(cluster_sizes[1] * (1.0 / sampling_rate), dim, sizeof(T), graph_degree);
        int id = root->get_id() * 2 + 1;
        int level = root->get_level() + 1;

        if (cur_shard_ram_estimate > budget) {
            right_node = new partitionNode(false, level, id);
            create_partition_tree<T>(cluster_data[1], cluster_sizes[1], dim, sampling_rate, budget, graph_degree,
                                     right_node);
        } else {
            right_node = new partitionNode(true, level, id);
            right_node->make_leaf(cluster_sizes[1]);
            delete[] cluster_data[1];
        }
    }

    std::unique_ptr<partitionNode> left(left_node);
    std::unique_ptr<partitionNode> right(right_node);
    delete[] data;
    root->make_internal(boundary_factor, pivot_data, std::move(left), std::move(right));
}

template<typename T>
int shard_data_into_clusters_with_tree(const std::string data_file, const std::string idmap_file, partitionNode *root,
                                       std::string prefix_path) {
    uint64_t read_blk_size = 64 * 1024 * 1024;
    bool is_root = (idmap_file == "");

    if (root->check_leaf()) {
        return 1;
    }

    int level = root->get_level();
    int nid = root->get_id();
    float boundary_factor = root->get_internal()->boundary_factor;
    std::vector<float> pivots = root->get_internal()->centroids;

    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *) &npts32, sizeof(uint32_t));
    base_reader.read((char *) &basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    size_t dim = basedim32;

    cached_ifstream base_id_reader;
    if (!is_root) {
        base_id_reader.open(idmap_file, read_blk_size, 0);
        uint32_t t1, t2;
        base_id_reader.read((char *) &t1, sizeof(uint32_t));
        base_id_reader.read((char *) &t2, sizeof(uint32_t));
    }

    std::unique_ptr<size_t[]> shard_counts = std::make_unique<size_t[]>(2);
    std::vector<std::ofstream> shard_data_writer(2);
    std::vector<std::ofstream> shard_idmap_writer(2);
    std::vector<std::string> shard_data_files(2);
    std::vector<std::string> shard_idmap_files(2);
    uint32_t dummy_size = 0;
    uint32_t const_one = 1;

    for (size_t i = 0; i < 2; i++) {
        int id = nid * 2 + i;
        std::string data_filename =
                prefix_path + "_level-" + std::to_string(level + 1) + "_subshard-" + std::to_string(id) + ".bin";
        std::string idmap_filename =
                prefix_path + "_level-" + std::to_string(level + 1) + "_subshard-" + std::to_string(id) +
                "_ids_uint32.bin";
        shard_data_files[i] = data_filename;
        shard_idmap_files[i] = idmap_filename;
        shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
        shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
        shard_data_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
        shard_data_writer[i].write((char *) &basedim32, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *) &const_one, sizeof(uint32_t));
        shard_counts[i] = 0;
    }

    size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_points);
    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size * 2);
    std::unique_ptr<float[]> beta = std::make_unique<float[]>(block_size);
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);
    std::unique_ptr<uint32_t[]> block_idmap = std::make_unique<uint32_t[]>(block_size);

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    int count = 0;
    for (size_t block = 0; block < num_blocks; block++) {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *) block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
        pipeann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

        if (!is_root) {
            base_id_reader.read((char *) block_idmap.get(), sizeof(uint32_t) * cur_blk_size);
        }

        math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots.data(), 2, 2,
                                            block_closest_centers.get(), beta.get());


        for (size_t p = 0; p < cur_blk_size; p++) {
            size_t shard_id = block_closest_centers[p * 2];
            uint32_t original_point_map_id = (is_root) ? (uint32_t) start_id + p : block_idmap[p];
            shard_data_writer[shard_id].write((char *) (block_data_T.get() + p * dim), sizeof(T) * dim);
            shard_idmap_writer[shard_id].write((char *) &original_point_map_id, sizeof(uint32_t));
            shard_counts[shard_id]++;

            assert(beta[p] > 1 + 1e-7);
            if (beta[p] <= boundary_factor) {
                count++;
                shard_id = block_closest_centers[p * 2 + 1];
                shard_data_writer[shard_id].write((char *) (block_data_T.get() + p * dim), sizeof(T) * dim);
                shard_idmap_writer[shard_id].write((char *) &original_point_map_id, sizeof(uint32_t));
                shard_counts[shard_id]++;
            }
        }
    }

    size_t total_count = 0;
    LOG(INFO) << "Actual level " << level << " nid " << nid << " shard sizes: ";
    for (size_t i = 0; i < 2; i++) {
        uint32_t cur_shard_count = (uint32_t) shard_counts[i];
        std::cerr << cur_shard_count << " ";
        total_count += cur_shard_count;
        shard_data_writer[i].seekp(0);
        shard_data_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
        shard_data_writer[i].close();
        shard_idmap_writer[i].seekp(0);
        shard_idmap_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
        shard_idmap_writer[i].close();
    }
    std::cerr << "\n";

    std::cout << "level: " << level << " node: " << nid << " bf: " << boundary_factor << " number: " << num_points
              << " boundary number: " << count << " ratio: " << (float) count / (float) num_points << std::endl;

    if (level != 0) {
        std::remove(data_file.c_str());
        std::remove(idmap_file.c_str());
    }

    int num_l = shard_data_into_clusters_with_tree<T>(shard_data_files[0], shard_idmap_files[0],
                                                      root->get_internal()->left.get(), prefix_path);
    int num_r = shard_data_into_clusters_with_tree<T>(shard_data_files[1], shard_idmap_files[1],
                                                      root->get_internal()->right.get(), prefix_path);
    return num_l + num_r;
}

template<typename T>
int hierarchical_partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                                           size_t graph_degree, const std::string prefix_path) {
    size_t train_dim;
    size_t num_train;
    float *train_data_float;
    double budget = 1024 * 1024 * 1024 * ram_budget;

    gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train, train_dim);

    auto root = new partitionNode(false, 0, 0);

    create_partition_tree<T>(train_data_float, num_train, train_dim, sampling_rate, budget, graph_degree, root);

    int num_parts = shard_data_into_clusters_with_tree<T>(data_file, "", root, prefix_path);

    int shard_id = 0;
    root->rename_shard(prefix_path, shard_id);

    assert(shard_id == num_parts);

    delete[] train_data_float;
    return num_parts;
}

// Instantations of supported templates
template void gen_random_slice<int8_t>(const std::string data_file, double p_val,
                                       std::unique_ptr<float[]> &sampled_data, size_t &slice_size, size_t &ndims);

template void gen_random_slice<uint8_t>(const std::string data_file, double p_val,
                                        std::unique_ptr<float[]> &sampled_data, size_t &slice_size, size_t &ndims);

template void gen_random_slice<float>(const std::string data_file, double p_val, std::unique_ptr<float[]> &sampled_data,
                                      size_t &slice_size, size_t &ndims);

template void gen_random_slice<int8_t>(const std::string base_file, const std::string output_prefix,
                                       double sampling_rate, size_t offset);

template void gen_random_slice<uint8_t>(const std::string base_file, const std::string output_prefix,
                                        double sampling_rate, size_t offset);

template void gen_random_slice<float>(const std::string base_file, const std::string output_prefix,
                                      double sampling_rate, size_t offset);

template void gen_random_slice<float>(const std::string data_file, double p_val, float *&sampled_data,
                                      size_t &slice_size, size_t &ndims);

template void gen_random_slice<uint8_t>(const std::string data_file, double p_val, float *&sampled_data,
                                        size_t &slice_size, size_t &ndims);

template void gen_random_slice<int8_t>(const std::string data_file, double p_val, float *&sampled_data,
                                       size_t &slice_size, size_t &ndims);

template int partition_with_ram_budget<int8_t>(const std::string data_file, const double sampling_rate,
                                               double ram_budget, size_t graph_degree, const std::string prefix_path,
                                               size_t k_base);

template int partition_with_ram_budget<uint8_t>(const std::string data_file, const double sampling_rate,
                                                double ram_budget, size_t graph_degree, const std::string prefix_path,
                                                size_t k_base);

template int partition_with_ram_budget<float>(const std::string data_file, const double sampling_rate,
                                              double ram_budget, size_t graph_degree, const std::string prefix_path,
                                              size_t k_base);

template int hierarchical_partition_with_ram_budget<int8_t>(const std::string data_file, const double sampling_rate,
                                                            double ram_budget, size_t graph_degree,
                                                            const std::string prefix_path);

template int hierarchical_partition_with_ram_budget<uint8_t>(const std::string data_file, const double sampling_rate,
                                                             double ram_budget, size_t graph_degree,
                                                             const std::string prefix_path);

template int hierarchical_partition_with_ram_budget<float>(const std::string data_file, const double sampling_rate,
                                                           double ram_budget, size_t graph_degree,
                                                           const std::string prefix_path);