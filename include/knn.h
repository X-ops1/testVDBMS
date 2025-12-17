//
// Created by owlet on 2025/11/28.
//

#ifndef PIPEANN_KNN_H
#define PIPEANN_KNN_H

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <queue>
#include <cblas.h>
#include <stdlib.h>

#include "omp.h"
#include "utils.h"

template<class T>
T div_round_up(const T numerator, const T denominator) {
    return (numerator % denominator == 0) ? (numerator / denominator) : 1 + (numerator / denominator);
}

using pairIF = std::pair<int, float>;

struct cmpmaxstruct {
    bool operator()(const pairIF &l, const pairIF &r) {
        return l.second < r.second;
    };
};

using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;


inline bool custom_dist(const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b) {
    return a.second < b.second;
}

void compute_l2sq(float *const points_l2sq, const float *const matrix, const int64_t num_points, const int dim) {
    assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
    for (int64_t d = 0; d < num_points; ++d)
        points_l2sq[d] =
                cblas_sdot(dim, matrix + (ptrdiff_t) d * (ptrdiff_t) dim, 1, matrix + (ptrdiff_t) d * (ptrdiff_t) dim,
                           1);
}

void distsq_to_points(const size_t dim,
                      float *dist_matrix,  // Col Major, cols are queries, rows are points
                      size_t npoints, const float *const points,
                      const float *const points_l2sq,  // points in Col major
                      size_t nqueries, const float *const queries,
                      const float *const queries_l2sq,  // queries in Col major
                      float *ones_vec = NULL)           // Scratchspace of num_data size and init to 1.0
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL) {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float) 1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float) -2.0, points, dim, queries,
                dim,
                (float) 0.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float) 1.0, points_l2sq, npoints,
                ones_vec, nqueries, (float) 1.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float) 1.0, ones_vec, npoints,
                queries_l2sq, nqueries, (float) 1.0, dist_matrix, npoints);
    if (ones_vec_alloc)
        delete[] ones_vec;
}

void exact_knn(const size_t dim, const size_t k,
               int *const closest_points,         // k * num_queries preallocated, col
        // major, queries columns
               float *const dist_closest_points,  // k * num_queries
        // preallocated, Dist to
        // corresponding closes_points
               size_t npoints,
               const float *const points,  // points in Col major
               size_t nqueries,
               const float *const queries)  // queries in Col major
{
    float *points_l2sq = new float[npoints];
    float *queries_l2sq = new float[nqueries];
    compute_l2sq(points_l2sq, points, npoints, dim);
    compute_l2sq(queries_l2sq, queries, nqueries, dim);

    size_t q_batch_size = (1 << 9);
    float *dist_matrix = new float[(size_t) q_batch_size * (size_t) npoints];

    for (uint64_t b = 0; b < div_round_up(nqueries, q_batch_size); ++b) {
        int64_t q_b = b * q_batch_size;
        int64_t q_e = ((b + 1) * q_batch_size > nqueries) ? nqueries : (b + 1) * q_batch_size;

        distsq_to_points(dim, dist_matrix, npoints, points, points_l2sq, q_e - q_b,
                         queries + (ptrdiff_t) q_b * (ptrdiff_t) dim, queries_l2sq + q_b);
        std::cout << "Computed distances for queries: [" << q_b << "," << q_e << ")" << std::endl;

#pragma omp parallel for schedule(dynamic, 16)
        for (long long q = q_b; q < q_e; q++) {
            maxPQIFCS point_dist;
            for (uint64_t p = 0; p < k; p++)
                point_dist.emplace(p, dist_matrix[(ptrdiff_t) p + (ptrdiff_t) (q - q_b) * (ptrdiff_t) npoints]);
            for (uint64_t p = k; p < npoints; p++) {
                if (point_dist.top().second > dist_matrix[(ptrdiff_t) p + (ptrdiff_t) (q - q_b) * (ptrdiff_t) npoints])
                    point_dist.emplace(p, dist_matrix[(ptrdiff_t) p + (ptrdiff_t) (q - q_b) * (ptrdiff_t) npoints]);
                if (point_dist.size() > k)
                    point_dist.pop();
            }
            for (ptrdiff_t l = 0; l < (ptrdiff_t) k; ++l) {
                closest_points[(ptrdiff_t) (k - 1 - l) + (ptrdiff_t) q * (ptrdiff_t) k] = point_dist.top().first;
                dist_closest_points[(ptrdiff_t) (k - 1 - l) + (ptrdiff_t) q * (ptrdiff_t) k] = point_dist.top().second;
                point_dist.pop();
            }
            assert(std::is_sorted(dist_closest_points + (ptrdiff_t) q * (ptrdiff_t) k,
                                  dist_closest_points + (ptrdiff_t) (q + 1) * (ptrdiff_t) k));
        }
        std::cout << "Computed exact k-NN for queries: [" << q_b << "," << q_e << ")" << std::endl;
    }

    delete[] dist_matrix;

    delete[] points_l2sq;
    delete[] queries_l2sq;
}


#endif  // PIPEANN_KNN_H
