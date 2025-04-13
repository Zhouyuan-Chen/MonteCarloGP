#pragma once

#include <igl/AABB.h>
#include <igl/point_mesh_squared_distance.h>

#define MY_PI 3.1415926

void sampling_on_sphere(Eigen::MatrixXd& SP, const Eigen::VectorXd& R);

void sampling_in_sphere(Eigen::MatrixXd& P, const Eigen::VectorXd& R);

void generate_sample(
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& SV,
    int64_t sample_num = 1000);