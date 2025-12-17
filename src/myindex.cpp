#include <iostream>
#include <omp.h>

#include "myindex.h"
#include "utils/timer.h"
#include "utils/lock_table.h"

namespace pipeann {

    template<typename T, typename TagT>
    myIndex<T, TagT>::myIndex(Metric m, const size_t dim, const size_t max_points, const bool dynamic_index,
                              const bool save_index_in_one_file, const bool enable_tags)
            : Index<T, TagT>(m, dim, max_points, dynamic_index, save_index_in_one_file, enable_tags) {
        const size_t total_internal_points = this->_max_points + this->_num_frozen_pts;
        this->weights_.resize(total_internal_points);
        this->wws_.resize(total_internal_points, 0.0f);
    }

    template<typename T, typename TagT>
    void myIndex<T, TagT>::mybuild(const char *filename, const size_t num_points_to_load, Parameters &parameters,
                                   const std::vector<TagT> &tags) {
        if (!file_exists(filename)) {
            LOG(ERROR) << "Data file " << filename << " does not exist!!! Exiting....";
            crash();
        }

        size_t file_num_points, file_dim;
        if (filename == nullptr) {
            LOG(INFO) << "Starting with an empty index.";
            this->_nd = 0;
        } else {
            pipeann::get_bin_metadata(filename, file_num_points, file_dim);
            if (file_num_points > this->_max_points || num_points_to_load > file_num_points) {
                LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points and file has "
                           << file_num_points << " points, but "
                           << "index can support only " << this->_max_points << " points as specified in constructor.";
                crash();
            }
            if (file_dim != this->_dim) {
                LOG(ERROR) << "ERROR: Driver requests loading " << this->_dim << " dimension,"
                           << "but file has " << file_dim << " dimension.";
                crash();
            }

            copy_aligned_data_from_file<T>(std::string(filename), this->_data, file_num_points, file_dim,
                                           this->_aligned_dim);

            LOG(INFO) << "Loading only first " << num_points_to_load << " from file.. ";
            this->_nd = num_points_to_load;

            if (this->_enable_tags && tags.size() != num_points_to_load) {
                LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points from file,"
                           << "but tags vector is of size " << tags.size() << ".";
                crash();
            }
            if (this->_enable_tags) {
                for (size_t i = 0; i < tags.size(); ++i) {
                    this->_tag_to_location[tags[i]] = (unsigned) i;
                    this->_location_to_tag[(unsigned) i] = tags[i];
                }
            }
        }

        this->generate_frozen_point();
        this->mylink(parameters);  // Primary func for creating nsg graph

        size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
        for (size_t i = 0; i < this->_nd; i++) {
            auto &pool = this->_final_graph[i];
            max = (std::max)(max, pool.size());
            min = (std::min)(min, pool.size());
            total += pool.size();
            if (pool.size() < 2)
                cnt++;
        }
        if (min > max)
            min = max;
        if (this->_nd > 0) {
            LOG(INFO) << "Index built with degree: max:" << max
                      << "  avg:" << (float) total / (float) (this->_nd + this->_num_frozen_pts) << "  min:" << min
                      << "  count(deg<2):" << cnt;
        }
        this->_width = (std::max)((unsigned) max, this->_width);
        this->_has_built = true;
    }

    template<typename T, typename TagT>
    void myIndex<T, TagT>::mylink(Parameters &parameters) {
        unsigned num_threads = parameters.num_threads;
        this->_saturate_graph = parameters.saturate_graph;
        unsigned L = parameters.L;  // Search list size
        const unsigned range = parameters.R;

        LOG(INFO) << "Parameters: "
                  << "L: " << L << ", R: " << range << ", saturate_graph: "
                  << (this->_saturate_graph ? "true" : "false")
                  << ", num_threads: " << num_threads << ", alpha: " << parameters.alpha;
        if (num_threads != 0)
            omp_set_num_threads(num_threads);

        int64_t n_vecs_to_visit = this->_nd + this->_num_frozen_pts;
        this->_ep = this->_num_frozen_pts > 0 ? this->_max_points : this->calculate_entry_point();

        std::vector<unsigned> init_ids;
        init_ids.emplace_back(this->_ep);
//    this->weights_.resize(this->_max_points + this->_num_frozen_pts);
//    this->wws_.resize(this->_max_points + this->_num_frozen_pts);

        pipeann::Timer link_timer;
#pragma omp parallel for schedule(dynamic)
        for (int64_t node = 0; node < n_vecs_to_visit; node++) {
            // search.
            std::vector<Neighbor> pool;
            tsl::robin_set<unsigned> visited;
            pool.reserve(2 * L);
            visited.reserve(2 * L);
            this->get_expanded_nodes(node, L, init_ids, pool, visited);
            // remove the node itself from pool.
            for (auto it = pool.begin(); it != pool.end();) {
                if (it->id == node) {
                    it = pool.erase(it);
                } else {
                    ++it;
                }
            }
            // prune neighbors.
            std::vector<unsigned> pruned_list;
            std::vector<float> weight_list;
            this->my_prune_neighbors(node, pool, parameters, pruned_list, weight_list);

            {
                // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, node);
                v2::LockGuard guard(this->_locks->wrlock(node));
                this->_final_graph[node].assign(pruned_list.begin(), pruned_list.end());
                this->weights_[node].clear();
                this->weights_[node].assign(weight_list.begin(), weight_list.end());
            }

            this->my_inter_insert(node, pruned_list, parameters);

            if (node % 100000 == 0) {
                std::cerr << "\r" << (100.0 * node) / (n_vecs_to_visit) << "% of index build completed.";
            }
        }

        if (this->_nd > 0) {
            LOG(INFO) << "Starting final cleanup..";
        }
#pragma omp parallel for schedule(dynamic, 65536)
        for (int64_t node_ctr = 0; node_ctr < n_vecs_to_visit; node_ctr++) {
            auto node = node_ctr;
            if (this->_final_graph[node].size() > range) {
                tsl::robin_set<unsigned> dummy_visited(0);
                std::vector<Neighbor> dummy_pool(0);
                std::vector<unsigned> new_out_neighbors;
                std::vector<float> new_out_weights;

                for (auto cur_nbr: this->_final_graph[node]) {
                    if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node) {
                        float dist = this->_distance->compare(this->_data + this->_aligned_dim * (size_t) node,
                                                              this->_data + this->_aligned_dim * (size_t) cur_nbr,
                                                              (unsigned) this->_aligned_dim);
                        dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
                        dummy_visited.insert(cur_nbr);
                    }
                }
                this->my_prune_neighbors(node, dummy_pool, parameters, new_out_neighbors, new_out_weights);

                this->_final_graph[node].clear();
                this->weights_[node].clear();
                for (size_t i = 0; i < new_out_neighbors.size(); i++) {
                    this->_final_graph[node].emplace_back(new_out_neighbors[i]);
                    this->weights_[node].emplace_back(new_out_weights[i]);
                }
            }
        }
        if (this->_nd > 0) {
            LOG(INFO) << "done. Link time: " << ((double) link_timer.elapsed() / (double) 1000000) << "s";
        }
    }

    template<typename T, typename TagT>
    void myIndex<T, TagT>::my_prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                                              const Parameters &parameter, std::vector<unsigned> &pruned_list,
                                              std::vector<float> &weight_list) {
        unsigned range = parameter.R;
        unsigned maxc = parameter.C;
        float alpha = parameter.alpha;

        if (pool.size() == 0) {
            crash();
        }

        this->_width = (std::max)(this->_width, range);

        std::sort(pool.begin(), pool.end());

        std::vector<std::pair<Neighbor, float>> result;
        result.reserve(range);
        std::vector<float> occlude_factor(pool.size(), 0);

        this->my_occlude_list(pool, alpha, range, maxc, result, occlude_factor, location);

        pruned_list.clear();
        weight_list.clear();
        assert(result.size() <= range);
        for (auto iter: result) {
            if (iter.first.id != location) {
                pruned_list.emplace_back(iter.first.id);
                weight_list.emplace_back(iter.second);
            }
        }

        if (this->_saturate_graph && alpha > 1) {
            for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
                if ((std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) == pruned_list.end()) &&
                    pool[i].id != location) {
                    pruned_list.emplace_back(pool[i].id);
                    weight_list.emplace_back(1.0f);
                }
            }
        }
    }

    template<typename T, typename TagT>
    void myIndex<T, TagT>::my_occlude_list(std::vector<Neighbor> &pool, const float alpha, const unsigned degree,
                                           const unsigned maxc, std::vector<std::pair<Neighbor, float>> &result,
                                           std::vector<float> &occlude_factor, const unsigned location) {
        if (pool.empty())
            return;
        assert(std::is_sorted(pool.begin(), pool.end()));
        assert(!pool.empty());

        float cur_alpha = 1;
        while (cur_alpha <= alpha && result.size() < degree) {
            unsigned start = 0;

            while (result.size() < degree && (start) < pool.size() && start < maxc) {
                auto &p = pool[start];
                if (occlude_factor[start] > cur_alpha) {
                    start++;
                    continue;
                }
                occlude_factor[start] = std::numeric_limits<float>::max();
                float weight = 1.0f;
                auto tmp_iter = std::find(this->_final_graph[location].begin(), this->_final_graph[location].end(),
                                          p.id);
                if (tmp_iter != this->_final_graph[location].end()) {
                    weight = this->weights_[location][tmp_iter - this->_final_graph[location].begin()];
                }
                for (unsigned t = start + 1; t < pool.size() && t < maxc; t++) {
                    if (occlude_factor[t] > alpha)
                        continue;
                    float djk =
                            this->_distance->compare(this->_data + this->_aligned_dim * (size_t) pool[t].id,
                                                     this->_data + this->_aligned_dim * (size_t) p.id,
                                                     (unsigned) this->_aligned_dim);

                    if (djk == 0.0) {
                        occlude_factor[t] = std::numeric_limits<float>::max();
                        weight += 1.0f;
                        {
                            v2::LockGuard guard(this->_locks->wrlock(pool[t].id));
                            this->wws_[pool[t].id] += 1.0f;
                        }
                    } else {
                        float ratio_dist = pool[t].distance / djk;
                        if (occlude_factor[t] <= cur_alpha && ratio_dist > cur_alpha) {
                            weight += 1.0f;
                            {
                                v2::LockGuard guard(this->_locks->wrlock(pool[t].id));
                                this->wws_[pool[t].id] += 1.0f;
                            }
                        }
                        occlude_factor[t] = (std::max)(occlude_factor[t], ratio_dist);
                    }
                }
                start++;
                result.emplace_back(std::make_pair(p, weight));
            }
            cur_alpha *= 1.2f;
        }
    }

    template<typename T, typename TagT>
    void
    myIndex<T, TagT>::my_inter_insert(unsigned n, std::vector<unsigned> &pruned_list, const Parameters &parameter) {
        const auto range = parameter.R;
        assert(n >= 0 && n < _nd + _num_frozen_pts);

        const auto &src_pool = pruned_list;

        assert(!src_pool.empty());

        for (auto des: src_pool) {
            /* des.id is the id of the neighbors of n */
            assert(des >= 0 && des < _max_points + _num_frozen_pts);
            /* des_pool contains the neighbors of the neighbors of n */
            auto &des_pool = this->_final_graph[des];
            std::vector<unsigned> copy_of_neighbors;
            bool prune_needed = false;
            {
                // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, des);
                v2::LockGuard guard(this->_locks->wrlock(des));
                if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
                    if (des_pool.size() < (uint64_t) (SLACK_FACTOR * range)) {
                        des_pool.emplace_back(n);
                        this->weights_[des].push_back(1.0f);
                        prune_needed = false;
                    } else {
                        copy_of_neighbors = des_pool;
                        prune_needed = true;
                    }
                }
            }  // des lock is released by this point

            if (prune_needed) {
                copy_of_neighbors.push_back(n);
                tsl::robin_set<unsigned> dummy_visited(0);
                std::vector<Neighbor> dummy_pool(0);

                size_t reserveSize = (size_t) (std::ceil(1.05 * SLACK_FACTOR * range));
                dummy_visited.reserve(reserveSize);
                dummy_pool.reserve(reserveSize);

                for (auto cur_nbr: copy_of_neighbors) {
                    if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != des) {
                        float dist = this->_distance->compare(this->_data + this->_aligned_dim * (size_t) des,
                                                              this->_data + this->_aligned_dim * (size_t) cur_nbr,
                                                              (unsigned) this->_aligned_dim);
                        dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
                        dummy_visited.insert(cur_nbr);
                    }
                }
                std::vector<unsigned> new_out_neighbors;
                std::vector<float> new_out_weights;
                this->my_prune_neighbors(des, dummy_pool, parameter, new_out_neighbors, new_out_weights);
                {
                    // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, des);
                    v2::LockGuard guard(this->_locks->wrlock(des));
                    this->_final_graph[des].assign(new_out_neighbors.begin(), new_out_neighbors.end());
                    this->weights_[des].assign(new_out_weights.begin(), new_out_weights.end());
                }
            }
        }
    }

    template<typename T, typename TagT>
    void myIndex<T, TagT>::mysave(const char *filename) {
        auto start = std::chrono::high_resolution_clock::now();
        std::unique_lock<std::shared_timed_mutex> lock(this->_update_lock);
        this->_change_lock.lock();

        this->compact_frozen_point();
        if (!this->_save_as_one_file) {
            std::string graph_file = std::string(filename);
            std::string tags_file = std::string(filename) + ".tags";
            std::string data_file = std::string(filename) + ".data";
            std::string delete_list_file = std::string(filename) + ".del";
            std::string weights_file = std::string(filename) + ".weights";

            delete_file(graph_file);
            this->save_graph(graph_file);
            delete_file(data_file);
            this->save_data(data_file);
            delete_file(tags_file);
            this->save_tags(tags_file);
            delete_file(delete_list_file);
            this->save_delete_list(delete_list_file);
            delete_file(weights_file);
            this->my_save_weights(weights_file);
        } else {
            delete_file(filename);
            std::vector<size_t> cumul_bytes(6, 0);
            cumul_bytes[0] = METADATA_SIZE;
            cumul_bytes[1] = cumul_bytes[0] + this->save_graph(std::string(filename), cumul_bytes[0]);
            cumul_bytes[2] = cumul_bytes[1] + this->save_data(std::string(filename), cumul_bytes[1]);
            cumul_bytes[3] = cumul_bytes[2] + this->save_tags(std::string(filename), cumul_bytes[2]);
            cumul_bytes[4] = cumul_bytes[3] + this->save_delete_list(filename, cumul_bytes[3]);
            cumul_bytes[5] = cumul_bytes[4] + this->my_save_weights(filename, cumul_bytes[4]);
            pipeann::save_bin<uint64_t>(filename, cumul_bytes.data(), cumul_bytes.size(), 1, 0);

            LOG(INFO) << "Saved index as one file to " << filename << " of size " << cumul_bytes[cumul_bytes.size() - 1]
                      << "B.";
        }

        this->reposition_frozen_point_to_end();

        this->_change_lock.unlock();
        auto stop = std::chrono::high_resolution_clock::now();
        auto timespan = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
        LOG(INFO) << "Time taken for save: " << timespan.count() << "s.";
    }

    template<typename T, typename TagT>
    uint64_t myIndex<T, TagT>::my_save_weights(std::string graph_file, size_t file_offset) {
        std::ofstream out;
        open_file_to_write(out, graph_file);

        out.seekp(file_offset, out.beg);

        float max_weight = 0.0f;
        out.write((char *) &max_weight, sizeof(float));
        uint64_t num_edges = 0;
        out.write((char *) &num_edges, sizeof(uint64_t));
        out.write((char *) &this->_nd, sizeof(uint64_t));
        for (unsigned i = 0; i < this->_nd; i++) {
            unsigned out_degree = (unsigned) this->weights_[i].size();
            out.write((char *) &out_degree, sizeof(unsigned));
            out.write((char *) this->weights_[i].data(), out_degree * sizeof(float));
            max_weight = std::max(max_weight, *std::max_element(this->weights_[i].begin(), this->weights_[i].end()));
            num_edges += out_degree;
        }
        out.seekp(file_offset, out.beg);
        out.write((char *) &max_weight, sizeof(float));
        out.write((char *) &num_edges, sizeof(uint64_t));
        out.close();
        LOG(INFO) << "saving weights..."
                  << ", #edges:" << num_edges << ", #vertice:" << this->_nd << ", max weight:" << max_weight;
        return num_edges;
    }

    // EXPORTS
    template
    class myIndex<float, uint32_t>;

    template
    class myIndex<int8_t, uint32_t>;

    template
    class myIndex<uint8_t, uint32_t>;
}  // namespace pipeann