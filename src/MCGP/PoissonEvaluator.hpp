#include "Sampling.hpp"

/**
 * x evaluate point
 * y source point
 */
double green_func(const Eigen::Vector3d& x, const Eigen::Vector3d& y, double R);

/**
 * V input mesh's vertices
 * F input mesh's faces
 * BV boundary condition
 * SV sampling vertices
 * EV evaluate value
 * source_term_func
 */

void poisson_evaluate(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& BV,
    const Eigen::MatrixXd& SV,
    Eigen::VectorXd& EV,
    const std::function<double(const Eigen::Vector3d&)>& source_term_func,
    const int64_t sample_num = 10,
    const int64_t walk_step = 10);

void poisson_evaluate_improved(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& BV,
    const Eigen::MatrixXd& SV,
    Eigen::VectorXd& EV,
    const std::function<double(const Eigen::Vector3d&)>& source_term_func,
    const int64_t sample_num = 10,
    const int64_t walk_step = 10,
    const double epsilon = 0.1,
    const int64_t sample_on_sphere_num = 8);