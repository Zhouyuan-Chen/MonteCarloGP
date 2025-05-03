#include "Sampling.hpp"

/**
 * V input mesh's vertices
 * F input mesh's faces
 * BV boundary condition
 * SV sampling vertices
 * EV evaluate value
 */

void laplace_evaluate(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& BV,
    const Eigen::MatrixXd& SV,
    Eigen::VectorXd& EV,
    const int64_t sample_num = 10,
    const int64_t walk_step = 10);

void laplace_evaluate_improved(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& BV,
    const Eigen::MatrixXd& SV,
    Eigen::VectorXd& EV,
    const int64_t sample_num = 10,
    const int64_t walk_step = 10,
    const double epsilon = 0.1,
    const int64_t sample_on_sphere_num = 8);