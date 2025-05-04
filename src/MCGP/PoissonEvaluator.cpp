#include "PoissonEvaluator.hpp"
#include <igl/AABB.h>
#include <parlay/primitives.h>
#include <chrono>
#include "Bucket.hpp"

double green_func(const Eigen::Vector3d& x, const Eigen::Vector3d& y, double R)
{
    double norm_y = y.norm();
    Eigen::Vector3d y_star = (R * R / (norm_y * norm_y)) * y;

    double G =
        1.0 / (4.0 * M_PI) * (1.0 / (x - y).norm() - (1.0 / (x - y_star).norm()) * (R / norm_y));
    return G;
}


void poisson_evaluate(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& BV,
    const Eigen::MatrixXd& SV,
    Eigen::VectorXd& EV,
    const std::function<double(const Eigen::Vector3d&)>& source_term_func,
    const int64_t sample_num,
    const int64_t walk_step)
{
    using Clock = std::chrono::high_resolution_clock;
    auto start_time = Clock::now();

    double x_max = V.col(0).maxCoeff();
    double x_min = V.col(0).minCoeff();

    igl::AABB<Eigen::MatrixXd, 3> bvh;
    bvh.init(V, F);

    Eigen::MatrixXd C;
    Eigen::VectorXd R, I;
    EV.resize(SV.rows());

    const auto& evaluate_val = [&](int64_t i) { return BV(F(i, 0)); };

    const std::function<double(Eigen::MatrixXd&, int64_t)> poisson_wos =
        [&](Eigen::MatrixXd& P, int64_t deep_num) -> double {
        // compute nearest point on the mesh
        // std::cout << "dis_be";
        bvh.squared_distance(V, F, P, R, I, C); // R: distance, I: face index, C: nearest points

        int64_t id = I(0);
        double distance = R(0);

        if (deep_num <= 0) {
            return evaluate_val(id);
        }

        Eigen::MatrixXd R_Sphere_vec;
        Eigen::VectorXd vecR(1);
        vecR(0) = distance;
        sampling_on_sphere(R_Sphere_vec, vecR);
        Eigen::MatrixXd P_boundary = P + R_Sphere_vec; // single row shifted

        sampling_in_sphere(R_Sphere_vec, vecR);
        Eigen::MatrixXd P_inside = P + R_Sphere_vec;

        double term1 = poisson_wos(P_boundary, deep_num - 1);
        double term2 = 4.0 / 3.0 * MY_PI * std::pow(distance, 3) *
                       source_term_func(P_inside.row(0).transpose()) *
                       green_func(P.row(0).transpose(), P_inside.row(0).transpose(), distance);
        // std::cout << deep_num << ":" << term1 << "+" << term2 << ":"
        //           << source_term_func(x_min, x_max, P_inside.row(0).transpose()) << std::endl;
        // std::cout << deep_num << ":" << term1 << "+" << term2 << std::endl;
        return term1 + term2;
    };

    parlay::parallel_for(0, static_cast<int>(SV.rows()), [&](int64_t i) {
        Eigen::MatrixXd P(1, 3);
        P << SV(i, 0), SV(i, 1), SV(i, 2);
        double evaluate_val_sum = 0;
        for (int64_t j = 0; j < sample_num; j++) {
            Eigen::MatrixXd P_copy = P;
            evaluate_val_sum += poisson_wos(P_copy, walk_step);
        }
        EV(i) = evaluate_val_sum / static_cast<double>(sample_num);
    });

    // for (int64_t i = 0; i < EV.size(); i++) {
    //     Eigen::MatrixXd P(1, 3);
    //     P << SV(i, 0), SV(i, 1), SV(i, 2);
    //     double evaluate_val = 0;
    //     for (int64_t j = 0; j < sample_num; j++) {
    //         Eigen::MatrixXd P_copy = P;
    //         evaluate_val += poisson_wos(P_copy, walk_step);
    //     }
    //     evaluate_val /= 1.0 * sample_num;
    //     EV(i) = evaluate_val;


    //     // process bar
    //     int barWidth = 50;
    //     double progress = (i + 1.0) / EV.size();
    //     std::cout << "[";
    //     int pos = barWidth * progress;
    //     for (int j = 0; j < barWidth; ++j) {
    //         if (j < pos)
    //             std::cout << "=";
    //         else if (j == pos)
    //             std::cout << ">";
    //         else
    //             std::cout << " ";
    //     }
    //     std::cout << "] " << int(progress * 100.0) << " %\r";
    //     std::cout.flush();
    // }

    auto end_time = Clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Poisson evaluation completed in " << elapsed.count() << " seconds.\n";
}

void poisson_evaluate_improved(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& BV,
    const Eigen::MatrixXd& SV,
    Eigen::VectorXd& EV,
    const std::function<double(const Eigen::Vector3d&)>& source_term_func,
    const int64_t sample_num,
    const int64_t walk_step,
    const double epsilon,
    const int64_t sample_on_sphere_num)
{
    using Clock = std::chrono::high_resolution_clock;
    auto start_time = Clock::now();
    std::cout << "Poisson evaluation work: " << static_cast<int>(SV.rows()) << " * " << sample_num
              << " * " << walk_step << " * log(" << static_cast<int>(F.rows()) << ") * "
              << sample_on_sphere_num << std::endl;
    std::cout << "Poisson evaluation span: " << 1 << " * " << 1 << " * " << walk_step << " * log("
              << static_cast<int>(F.rows()) << ") * " << 1 << std::endl;
    double x_max = V.col(0).maxCoeff();
    double x_min = V.col(0).minCoeff();

    igl::AABB<Eigen::MatrixXd, 3> bvh;
    bvh.init(V, F);

    Eigen::MatrixXd C;
    Eigen::VectorXd R, I;
    EV.resize(SV.rows());

    const auto& evaluate_val = [&](int64_t i) { return BV(F(i, 0)); };

    const std::function<double(Eigen::MatrixXd&, int64_t, double)> poisson_wos =
        [&](Eigen::MatrixXd& P, int64_t deep_num, double distance) -> double {
        if (distance < 0) {
            bvh.squared_distance(V, F, P, R, I, C);
            distance = R(0);
        }

        int64_t id = I(0);

        if (deep_num <= 0|| distance <= epsilon) {
            return evaluate_val(id);
        }

        std::vector<Eigen::MatrixXd> new_samples(sample_on_sphere_num);
        std::vector<double> new_sample_dists(sample_on_sphere_num);

        parlay::parallel_for(0, sample_on_sphere_num, [&](int k) {
            Eigen::MatrixXd R_Sphere_vec;
            Eigen::VectorXd vecR(1);
            vecR(0) = distance;
            sampling_on_sphere(R_Sphere_vec, vecR);
            Eigen::MatrixXd new_sample_P_boundary = P + R_Sphere_vec; // single row shifted
            new_samples[k] = new_sample_P_boundary;

            Eigen::VectorXd R_temp;
            Eigen::VectorXi I_temp;
            Eigen::MatrixXd C_temp;
            bvh.squared_distance(V, F, new_sample_P_boundary, R_temp, I_temp, C_temp);
            new_sample_dists[k] = R_temp(0);
        });

        int best_k = parlay::reduce(
                         parlay::delayed_tabulate(
                             sample_on_sphere_num,
                             [&](int k) { return std::make_pair(new_sample_dists[k], k); }),
                         parlay::minm<std::pair<double, int>>())
                         .second;

        double term1 = poisson_wos(new_samples[best_k], deep_num - 1, new_sample_dists[best_k]);

        Eigen::MatrixXd R_Sphere_vec;
        Eigen::VectorXd vecR(1);
        vecR(0) = new_sample_dists[best_k];

        sampling_in_sphere(R_Sphere_vec, vecR);
        Eigen::MatrixXd P_inside = P + R_Sphere_vec;

        double term2 = 4.0 / 3.0 * MY_PI * std::pow(new_sample_dists[best_k], 3) *
                       source_term_func(P_inside.row(0).transpose()) *
                       green_func(P.row(0).transpose(), P_inside.row(0).transpose(), new_sample_dists[best_k]);

        return term1 + term2;
    };
    parlay::parallel_for(0, static_cast<int>(SV.rows()), [&](int64_t i) {
        Eigen::MatrixXd P(1, 3);
        P << SV(i, 0), SV(i, 1), SV(i, 2);

        std::vector<double> results(sample_num);
        parlay::parallel_for(0, static_cast<int>(sample_num), [&](int64_t j) {
            Eigen::MatrixXd P_copy = P;
            results[j] = poisson_wos(P_copy, walk_step, -1.0);
        });
        double evaluate_val_sum = std::accumulate(results.begin(), results.end(), 0.0);
        EV(i) = evaluate_val_sum / static_cast<double>(sample_num);
    });

    auto end_time = Clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Poisson evaluation completed in " << elapsed.count() << " seconds.\n";
}


void poisson_evaluate_load_balance(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& BV,
    const Eigen::MatrixXd& SV,
    Eigen::VectorXd& EV,
    const std::function<double(const Eigen::Vector3d&)>& source_term_func,
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
    std::cout << "Poisson evaluation work: " << static_cast<int>(SV.rows()) << " * " << sample_num
              << " * " << walk_step << " * log(" << static_cast<int>(F.rows()) << ") * "
              << sample_on_sphere_num << std::endl;
    std::cout << "Poisson evaluation span: " << 1 << " * " << 1 << " * " << walk_step << " * log("
              << static_cast<int>(F.rows()) << ") * " << 1 << std::endl;
    double x_max = V.col(0).maxCoeff();
    double x_min = V.col(0).minCoeff();

    const auto& evaluate_val = [&](int64_t i) { return BV(F(i, 0)); };

    const std::function<double(Eigen::MatrixXd&, int64_t, double)> poisson_wos =
        [&](Eigen::MatrixXd& P, int64_t deep_num, double distance) -> double {
        if (distance < 0) {
            bvh.squared_distance(V, F, P, R, I, C);
            distance = R(0);
        }

        int64_t id = I(0);

        if (deep_num <= 0|| distance <= epsilon) {
            return evaluate_val(id);
        }

        std::vector<Eigen::Vector3d> new_samples(sample_on_sphere_num);
        std::vector<double> new_sample_dists(sample_on_sphere_num);

        parlay::parallel_for(0, sample_on_sphere_num, [&](int k) {
            Eigen::MatrixXd R_Sphere_vec;
            Eigen::VectorXd vecR(1);
            vecR(0) = distance;
            sampling_on_sphere(R_Sphere_vec, vecR);
            Eigen::Vector3d new_sample_P = P.row(0) + R_Sphere_vec.row(0); // single row shifted
            new_samples[k] = new_sample_P;

            Eigen::VectorXd R_temp;
            Eigen::VectorXi I_temp;
            Eigen::MatrixXd C_temp;
            bvh.squared_distance(V, F, new_sample_P.transpose(), R_temp, I_temp, C_temp);
            new_sample_dists[k] = R_temp(0);
        });

        int best_k = parlay::reduce(
                         parlay::delayed_tabulate(
                             sample_on_sphere_num,
                             [&](int k) { return std::make_pair(new_sample_dists[k], k); }),
                         parlay::minm<std::pair<double, int>>())
                         .second;

        P.row(0) = new_samples[best_k];
        double best_dist = new_sample_dists[best_k];

        double term1 = poisson_wos(P, deep_num - 1, best_dist);

        Eigen::MatrixXd R_Sphere_vec_inside;
        Eigen::VectorXd vecR(1);
        vecR(0) = best_dist;

        sampling_in_sphere(R_Sphere_vec_inside, vecR);
        Eigen::MatrixXd P_inside = P + R_Sphere_vec_inside;

        double term2 = 4.0 / 3.0 * MY_PI * std::pow(best_dist, 3) *
                       source_term_func(P_inside.row(0).transpose()) *
                       green_func(P.row(0).transpose(), P_inside.row(0).transpose(), best_dist);

        return term1 + term2;
    };
    parlay::parallel_for(0, num_threads, [&](int t) {
        parlay::parallel_for(0, static_cast<int>(buckets[t].point_index.size()), [&](int p) {
            int idx = buckets[t].point_index[p];

            Eigen::MatrixXd P(1, 3);
            P << SV(idx, 0), SV(idx, 1), SV(idx, 2);

            std::vector<double> results(sample_num);
            parlay::parallel_for(0, static_cast<int>(sample_num), [&](int64_t j) {
                Eigen::MatrixXd P_copy = P;
                results[j] = poisson_wos(P_copy, walk_step, -1.0);
            });
            double evaluate_val_sum = std::accumulate(results.begin(), results.end(), 0.0);
            EV(idx) = evaluate_val_sum / static_cast<double>(sample_num);
        });
    });

    auto end_time = Clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Poisson evaluation completed in " << elapsed.count() << " seconds.\n";
}