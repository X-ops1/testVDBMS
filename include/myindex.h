#pragma once

#include "index.h"

namespace pipeann {

    template<typename T, typename TagT = uint32_t>
    class myIndex : public Index<T, TagT> {
    public:
        myIndex(Metric m, const size_t dim, const size_t max_points = 1, const bool dynamic_index = false,
                const bool save_index_in_one_file = false, const bool enable_tags = false);

        void mybuild(const char *filename, const size_t num_points_to_load, Parameters &parameters,
                     const std::vector<TagT> &tags = std::vector<TagT>());

        void mylink(Parameters &parameters);

        void my_prune_neighbors(const unsigned location, std::vector<Neighbor> &pool, const Parameters &parameter,
                                std::vector<unsigned> &pruned_list, std::vector<float> &weight_list);

        void my_occlude_list(std::vector<Neighbor> &pool, const float alpha, const unsigned degree, const unsigned maxc,
                             std::vector<std::pair<Neighbor, float>> &result, std::vector<float> &occlude_factor,
                             const unsigned location);

        void my_inter_insert(unsigned n, std::vector<unsigned> &pruned_list, const Parameters &parameter);

        void mysave(const char *filename);

        uint64_t my_save_weights(std::string filename, size_t file_offset = 0);

        std::vector<std::vector<float>> weights_;
        std::vector<float> wws_;
    };

}  // namespace pipeann