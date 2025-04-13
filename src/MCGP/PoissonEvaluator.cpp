#include "PoissonEvaluator.hpp"
#include <igl/AABB.h>
#include <parlay/primitives.h>
#include <chrono>

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