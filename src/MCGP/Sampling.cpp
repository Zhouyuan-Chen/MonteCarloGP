#include "Sampling.hpp"
#include <igl/fast_winding_number.h>
#include <igl/slice_mask.h>
#include <random>

void sampling_on_sphere(Eigen::MatrixXd& SP, const Eigen::VectorXd& R)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::uniform_real_distribution<double> dist_azimuth(0.0, 2 * MY_PI);
    std::uniform_real_distribution<double> dist_cos_theta(-1.0, 1.0);

    int N = R.size();
    SP.resize(N, 3);

    for (int i = 0; i < N; ++i) {
        double theta = acos(dist_cos_theta(generator)); // polar angle
        double phi = dist_azimuth(generator); // azimuthal angle

        double x = R(i) * sin(theta) * cos(phi);
        double y = R(i) * sin(theta) * sin(phi);
        double z = R(i) * cos(theta);

        SP.row(i) = Eigen::Vector3d(x, y, z);
    }
}

void sampling_in_sphere(Eigen::MatrixXd& P, const Eigen::VectorXd& R)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::uniform_real_distribution<double> dist_azimuth(0.0, 2 * MY_PI);
    std::uniform_real_distribution<double> dist_cos_theta(-1.0, 1.0);
    std::uniform_real_distribution<double> dist_radius(0.0, 1.0);

    int N = R.size();
    P.resize(N, 3);

    for (int i = 0; i < N; ++i) {
        double u = dist_radius(generator);
        double r = std::cbrt(u) * R(i); // scale radius to be uniform in volume

        double cos_theta = dist_cos_theta(generator);
        double theta = std::acos(cos_theta);
        double phi = dist_azimuth(generator);

        double x = r * std::sin(theta) * std::cos(phi);
        double y = r * std::sin(theta) * std::sin(phi);
        double z = r * cos_theta;

        P.row(i) = Eigen::Vector3d(x, y, z);
    }
}


void generate_sample(
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& SV,
    int64_t sample_num)
{
    Eigen::Vector3d min = V.colwise().minCoeff();
    Eigen::Vector3d max = V.colwise().maxCoeff();
    Eigen::Vector3d dig = max - min;

    double lower_bound = 0;
    double upper_bound = 1;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    Eigen::MatrixXd QV(sample_num, 3);
    int64_t valid_sample_num = 0;
    for (int64_t i = 0; i < sample_num; i++) {
        double re_x_ratio = unif(re);
        double re_y_ratio = unif(re);
        double re_z_ratio = unif(re);

        double x = re_x_ratio * dig.x() + min.x();
        double y = re_y_ratio * dig.y() + min.y();
        double z = re_z_ratio * dig.z() + min.z();
        QV(i, 0) = x;
        QV(i, 1) = y;
        QV(i, 2) = z;
    }

    Eigen::VectorXd S;
    igl::FastWindingNumberBVH bvh;
    igl::fast_winding_number(V.cast<float>().eval(), F, 2, bvh);
    igl::fast_winding_number(bvh, 2, QV.cast<float>().eval(), S);
    igl::slice_mask(QV, S.array() > 0.5, 1, SV);
}