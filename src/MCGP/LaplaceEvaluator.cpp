#include "LaplaceEvaluator.hpp"
#include <igl/AABB.h>
#include <igl/point_mesh_squared_distance.h>
#include <parlay/primitives.h>
#include <chrono>
#include "Bucket.hpp"


void laplace_evaluate(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& BV,
    const Eigen::MatrixXd& SV,
    Eigen::VectorXd& EV,
    const int64_t sample_num,
    const int64_t walk_step)
{
    using Clock = std::chrono::high_resolution_clock;
    auto start_time = Clock::now();
    std::cout << "Laplace evaluation work: " << static_cast<int>(SV.rows()) << " * " << sample_num
              << " * " << walk_step << " * log(" << static_cast<int>(F.rows()) << ")" << std::endl;
    std::cout << "Laplace evaluation span: " << 1 << " * " << 1 << " * " << walk_step << " * log("
              << static_cast<int>(F.rows()) << ")" << std::endl;

    igl::AABB<Eigen::MatrixXd, 3> bvh;
    bvh.init(V, F);

    Eigen::MatrixXd C;
    Eigen::VectorXd R, I;
    EV.resize(SV.rows());

    const auto& evaluate_val = [&](int64_t i) { return BV(F(i, 0)); };

    const std::function<double(Eigen::MatrixXd&, int64_t)> laplace_wos =
        [&](Eigen::MatrixXd& P, int64_t deep_num) -> double {
        // compute nearest point on the mesh
        // std::cout << "dis_be";
        bvh.squared_distance(V, F, P, R, I, C); // R: distance, I: face index, C: nearest points

        // std::cout << "1";
        int64_t id = I(0);
        double distance = R(0);

        // std::cout << "2";

        if (deep_num <= 0) {
            return evaluate_val(id);
        }

        // Walk on sphere
        Eigen::MatrixXd R_Sphere_vec;
        Eigen::VectorXd vecR(1);
        vecR(0) = distance;
        // std::cout << "3";

        sampling_on_sphere(R_Sphere_vec, vecR); // returns Nx3
        // std::cout << "4" << std::endl;
        P = P + R_Sphere_vec; // single row shifted

        return laplace_wos(P, deep_num - 1);
    };


    parlay::parallel_for(0, static_cast<int>(SV.rows()), [&](int64_t i) {
        Eigen::MatrixXd P(1, 3);
        P << SV(i, 0), SV(i, 1), SV(i, 2);
        double evaluate_val_sum = 0;
        for (int64_t j = 0; j < sample_num; j++) {
            Eigen::MatrixXd P_copy = P;
            evaluate_val_sum += laplace_wos(P_copy, walk_step);
        }
        EV(i) = evaluate_val_sum / static_cast<double>(sample_num);
    });

    // for (int64_t i = 0; i < EV.size(); i++) {
    //     Eigen::MatrixXd P(1, 3);
    //     P << SV(i, 0), SV(i, 1), SV(i, 2);
    //     double evaluate_val = 0;
    //     for (int64_t j = 0; j < sample_num; j++) {
    //         Eigen::MatrixXd P_copy = P;
    //         evaluate_val += laplace_wos(P_copy, walk_step);
    //     }
    //     evaluate_val /= 1.0 * sample_num;
    //     EV(i) = evaluate_val;


    //     // process bar
    //     // int barWidth = 50;
    //     // double progress = (i + 1.0) / EV.size();
    //     // std::cout << "[";
    //     // int pos = barWidth * progress;
    //     // for (int j = 0; j < barWidth; ++j) {
    //     //     if (j < pos)
    //     //         std::cout << "=";
    //     //     else if (j == pos)
    //     //         std::cout << ">";
    //     //     else
    //     //         std::cout << " ";
    //     // }
    //     // std::cout << "] " << int(progress * 100.0) << " %\r";
    //     // std::cout.flush();
    // }

    auto end_time = Clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Laplace evaluation completed in " << elapsed.count() << " seconds.\n";
}

void laplace_evaluate_improved(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& BV,
    const Eigen::MatrixXd& SV,
    Eigen::VectorXd& EV,
    const int64_t sample_num,
    const int64_t walk_step,
    const double epsilon,
    const int64_t sample_on_sphere_num)
{
    using Clock = std::chrono::high_resolution_clock;
    auto start_time = Clock::now();
    std::cout << "Laplace evaluation work: " << static_cast<int>(SV.rows()) << " * " << sample_num
              << " * " << walk_step << " * log(" << static_cast<int>(F.rows()) << ") * "
              << sample_on_sphere_num << std::endl;
    std::cout << "Laplace evaluation span: " << 1 << " * " << 1 << " * " << walk_step << " * log("
              << static_cast<int>(F.rows()) << ") * " << 1 << std::endl;
    igl::AABB<Eigen::MatrixXd, 3> bvh;
    bvh.init(V, F);

    Eigen::MatrixXd C;
    Eigen::VectorXd R, I;
    EV.resize(SV.rows());

    const auto& evaluate_val = [&](int64_t i) { return BV(F(i, 0)); };

    const std::function<double(Eigen::MatrixXd, int64_t, int64_t, double)> laplace_wos =
        [&](Eigen::MatrixXd P, int64_t deep_num, int64_t id, double distance) -> double {
        if (distance < 0) {
            bvh.squared_distance(V, F, P, R, I, C);
            id = I(0);
            distance = R(0);
        }

        if (deep_num <= 0 || distance <= epsilon) {
            return evaluate_val(id);
        }

        std::vector<Eigen::MatrixXd> new_samples(sample_on_sphere_num);
        std::vector<int64_t> new_sample_faceid(sample_on_sphere_num);
        std::vector<double> new_sample_dists(sample_on_sphere_num);

        parlay::parallel_for(0, sample_on_sphere_num, [&](int k) {
            Eigen::MatrixXd R_Sphere_vec;
            Eigen::VectorXd vecR(1);
            vecR(0) = distance;
            sampling_on_sphere(R_Sphere_vec, vecR);
            Eigen::MatrixXd new_sample_P = P + R_Sphere_vec;
            new_samples[k] = new_sample_P;

            Eigen::VectorXd R_temp;
            Eigen::VectorXi I_temp;
            Eigen::MatrixXd C_temp;
            bvh.squared_distance(V, F, new_sample_P, R_temp, I_temp, C_temp);
            new_sample_faceid[k] = I_temp(0);
            new_sample_dists[k] = R_temp(0);

        });

        // reduce求最小距离对应的index
        int best_k = parlay::reduce(
                         parlay::delayed_tabulate(
                             sample_on_sphere_num,
                             [&](int k) { return std::make_pair(new_sample_dists[k], k); }),
                         parlay::minm<std::pair<double, int>>())
                         .second;

        return laplace_wos(new_samples[best_k], deep_num - 1, new_sample_faceid[best_k],new_sample_dists[best_k]);
    };


    parlay::parallel_for(0, static_cast<int>(SV.rows()), [&](int64_t i) {
        Eigen::MatrixXd P(1, 3);
        P << SV(i, 0), SV(i, 1), SV(i, 2);

        std::vector<double> results(sample_num);
        parlay::parallel_for(0, static_cast<int>(sample_num), [&](int64_t j) {
            Eigen::MatrixXd P_copy = P;
            results[j] = laplace_wos(P_copy, walk_step, -1, -1.0);
        });
        double evaluate_val_sum = std::accumulate(results.begin(), results.end(), 0.0);
        EV(i) = evaluate_val_sum / static_cast<double>(sample_num);
    });


    auto end_time = Clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Laplace evaluation completed in " << elapsed.count() << " seconds.\n";
}

void laplace_evaluate_load_balance(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& BV,
    const Eigen::MatrixXd& SV,
    Eigen::VectorXd& EV,
    const int64_t sample_num,
    const int64_t walk_step,
    const double epsilon,
    const int64_t sample_on_sphere_num)
{
    int num_threads = parlay::num_workers();
    std::cout << "Parlaylib num threads = " << num_threads << std::endl;
    igl::AABB<Eigen::MatrixXd, 3> bvh;
    bvh.init(V, F);

    Eigen::MatrixXd C;
    Eigen::VectorXd R, I;
    EV.resize(SV.rows());

    bvh.squared_distance(V, F, SV, R, I, C);

    auto scores = parlay::tabulate(SV.rows(), [&](size_t i) { return std::make_pair(i, R(i)); });

    scores = parlay::sort(scores, [](auto a, auto b) { return a.second > b.second; });

    std::vector<Bucket> buckets(num_threads);

    // 贪心分配
    for (auto [idx, score] : scores) {
        auto& bucket =
            *std::min_element(buckets.begin(), buckets.end(), [](const Bucket& a, const Bucket& b) {
                return a.total_score < b.total_score;
            });
        bucket.point_index.push_back(idx);
        bucket.total_score += score;
    }


    using Clock = std::chrono::high_resolution_clock;
    auto start_time = Clock::now();
    std::cout << "Laplace evaluation work: " << static_cast<int>(SV.rows()) << " * " << sample_num
              << " * " << walk_step << " * log(" << static_cast<int>(F.rows()) << ") * "
              << sample_on_sphere_num << std::endl;
    std::cout << "Laplace evaluation span: " << 1 << " * " << 1 << " * " << walk_step << " * log("
              << static_cast<int>(F.rows()) << ") * " << 1 << std::endl;


    const auto& evaluate_val = [&](int64_t i) { return BV(F(i, 0)); };

    const std::function<double(Eigen::MatrixXd, int64_t, int64_t, double)> laplace_wos =
        [&](Eigen::MatrixXd P, int64_t deep_num, int64_t id, double distance) -> double {
        if (distance < 0) {
            bvh.squared_distance(V, F, P, R, I, C);
            id = I(0);
            distance = R(0);
        }

        if (deep_num <= 0 || distance <= epsilon) {
            return evaluate_val(id);
        }

        std::vector<Eigen::MatrixXd> new_samples(sample_on_sphere_num);
        std::vector<int64_t> new_sample_faceid(sample_on_sphere_num);
        std::vector<double> new_sample_dists(sample_on_sphere_num);

        parlay::parallel_for(0, sample_on_sphere_num, [&](int k) {
            Eigen::MatrixXd R_Sphere_vec;
            Eigen::VectorXd vecR(1);
            vecR(0) = distance;
            sampling_on_sphere(R_Sphere_vec, vecR);
            Eigen::MatrixXd new_sample_P = P + R_Sphere_vec;
            new_samples[k] = new_sample_P;

            Eigen::VectorXd R_temp;
            Eigen::VectorXi I_temp;
            Eigen::MatrixXd C_temp;
            bvh.squared_distance(V, F, new_sample_P, R_temp, I_temp, C_temp);
            new_sample_faceid[k] = I_temp(0);
            new_sample_dists[k] = R_temp(0);
        });

        // reduce求最小距离对应的index
        int best_k = parlay::reduce(
                         parlay::delayed_tabulate(
                             sample_on_sphere_num,
                             [&](int k) { return std::make_pair(new_sample_dists[k], k); }),
                         parlay::minm<std::pair<double, int>>())
                         .second;

        return laplace_wos(new_samples[best_k], deep_num - 1, new_sample_faceid[best_k],new_sample_dists[best_k]);
    };

    parlay::parallel_for(0, num_threads, [&](int t) {
        parlay::parallel_for(0, static_cast<int>(buckets[t].point_index.size()), [&](int p) {
            int idx = buckets[t].point_index[p];

            Eigen::MatrixXd P(1, 3);
            P << SV(idx, 0), SV(idx, 1), SV(idx, 2);

            std::vector<double> results(sample_num);
            parlay::parallel_for(0, static_cast<int>(sample_num), [&](int64_t j) {
                Eigen::MatrixXd P_copy = P;
                results[j] = laplace_wos(P_copy, walk_step, -1, -1.0);
            });
            double evaluate_val_sum = std::accumulate(results.begin(), results.end(), 0.0);
            EV(idx) = evaluate_val_sum / static_cast<double>(sample_num);
        });
    });
    auto end_time = Clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Laplace evaluation completed in " << elapsed.count() << " seconds.\n";
}