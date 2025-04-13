#include <igl/colormap.h>
#include <igl/jet.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <Eigen/Core>

#include <MCGP/LaplaceEvaluator.hpp>
#include <MCGP/PoissonEvaluator.hpp>
#include <MCGP/Sampling.hpp>

int main(int argc, char* argv[])
{
    Eigen::MatrixXd V, SV, C;
    Eigen::MatrixXi F;
    Eigen::VectorXd C_s;
    int64_t sample_num = 1000;
    igl::read_triangle_mesh(argv[1], V, F);

    // set boundary condition
    Eigen::VectorXd BV_raw = V.col(1);
    double y_min = BV_raw.minCoeff();
    double y_max = BV_raw.maxCoeff();
    Eigen::VectorXd BV = (BV_raw.array() - y_min) / (y_max - y_min);

    // set source term
    // auto source_term_func = [&](double x_min, double x_max, const Eigen::Vector3d P) -> double {
    //     double norm_x = (P.x() - x_min) / (x_max - x_min);
    //     double c = 1;

    //     // linear
    //     return norm_x * c;

    //     // Gaussian
    //     // double center = 0.5;
    //     // double sigma = 0.1;
    //     // return std::exp(-std::pow(norm_x - center, 2) / (2 * sigma * sigma));

    //     // sin/cos
    //     // return std::cos(M_PI * norm_x);
    //     // return std::cos(M_PI * P.x());
    // };
    Eigen::Vector3d center = V.colwise().mean();
    double radius = (V.rowwise() - center.transpose()).rowwise().norm().maxCoeff();
    auto source_term_func = [&](const Eigen::Vector3d& P) -> double {
        double dist = (P - center).norm();
        double sigma = radius * 0.2;
        return std::exp(-(dist * dist) / (2.0 * sigma * sigma));
        // return 0;
    };


    igl::opengl::glfw::Viewer viewer;
    Eigen::MatrixXd Vtemp;

    viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer&, unsigned int key, int mod) {
        switch (key) {
        case 's':
            std::cout << "Sampling..." << std::endl;
            generate_sample(V, F, SV, sample_num);
            C.resize(SV.rows() + 1, 3);
            C.setZero();
            C(SV.rows(), 0) = 1;
            Vtemp.resize(SV.rows() + 1, 3);
            Vtemp.topRows(SV.rows()) = SV;
            Vtemp.bottomRows(1) = center.transpose();
            SV = Vtemp;
            break;
        case '+':
            std::cout << "Increasing Sampling..." << std::endl;
            sample_num *= 2;
            generate_sample(V, F, SV, sample_num);
            C.resize(SV.rows(), 3);
            C.setZero();
            break;
        case '-':
            std::cout << "Decreasing Sampling..." << std::endl;
            sample_num /= 2;
            generate_sample(V, F, SV, sample_num);
            C.resize(SV.rows(), 3);
            C.setZero();
            break;
        case 'l': {
            std::cout << "Laplace" << std::endl;
            C.resize(SV.rows(), 3);
            C.setZero();
            C_s.resize(SV.rows());
            BV = (BV_raw.array() - y_min) / (y_max - y_min);
            std::cout << "Begin" << std::endl;
            laplace_evaluate(V, F, BV, SV, C_s);
            std::cout << "Finished" << std::endl;

            igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, C_s, C_s.minCoeff(), C_s.maxCoeff(), C);

            // std::cout << C_s;
        } break;
        case 'p':
            std::cout << "Poisson" << std::endl;
            C.resize(SV.rows(), 3);
            C.setZero();
            C_s.resize(SV.rows());
            C_s.setZero();
            BV.setZero();
            std::cout << "Begin" << std::endl;
            poisson_evaluate(V, F, BV, SV, C_s, source_term_func);
            std::cout << "Finished" << std::endl;

            igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, C_s, C_s.minCoeff(), C_s.maxCoeff(), C);
            // std::cout << C << std::endl;
            break;
        default: break;
        }
        viewer.data().clear();
        viewer.data().set_points(SV, C);
        return true;
    };

    viewer.data().set_mesh(V, F);
    viewer.data().show_lines = true;
    viewer.launch();

    return EXIT_SUCCESS;
}