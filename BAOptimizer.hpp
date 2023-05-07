#include <iostream>
#include <string>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "sophus/se3.hpp"

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> Vecs2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> Vecs3d;

class BAOptimizer
{
public:
    Eigen::Matrix3d K;
    double fx, fy, cx, cy;

public:
    BAOptimizer(const cv::Mat& cvK) {
        cv::cv2eigen(cvK, K);
        fx = K(0, 0);
        fy = K(1, 1);
        cx = K(0, 2);
        cy = K(1, 2);
    }
    ~BAOptimizer() {}
    
    // 使用DLT求解PnP问题
    void DLT(const Vecs3d &points_3d, const Vecs2d &points_2d, Sophus::SE3d& pose){
        int n_points = points_3d.size();
        Eigen::MatrixXd A(2*n_points, 12);

        // 构造A矩阵
        for (int i = 0; i < n_points; ++i) {
            A.row(2*i) << points_3d[i][0], points_3d[i][1], points_3d[i][2], 1.0, 0.0, 0.0, 0.0, 0.0, -points_2d[i][0]*points_3d[i][0], -points_2d[i][0]*points_3d[i][1], -points_2d[i][0]*points_3d[i][2], -points_2d[i][0];
            A.row(2*i+1) << 0.0, 0.0, 0.0, 0.0, points_3d[i][0], points_3d[i][1], points_3d[i][2], 1.0, -points_2d[i][1]*points_3d[i][0], -points_2d[i][1]*points_3d[i][1], -points_2d[i][1]*points_3d[i][2], -points_2d[i][1];
        }

        // 奇异值分解
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::VectorXd p = svd.matrixV().col(11);
        
        // 从p中恢复位姿
        Eigen::Matrix<double, 3, 3> R;
        R << p(0), p(1), p(2),
            p(4), p(5), p(6),
            p(8), p(9), p(10);
        Eigen::Quaterniond q(R);
        Eigen::Vector3d t(p(3), p(7), p(11));
        pose = Sophus::SE3d(q, t);

        // 计算累计重投影误差
        double cost = 0;
        for (int i=0; i<points_3d.size(); i++){  // 计算每组2d-3d匹配点对的损失并累加
            Eigen::Vector3d pc = pose * points_3d[i];  // 该3D点在相机坐标系下的坐标
            double inv_z = 1.0 / pc[2];  // 1/z
            double inv_z2 = inv_z * inv_z;  // 1/z^2

            Eigen::Vector2d reproj_2d(fx * pc[0] * inv_z + cx, fy * pc[1] * inv_z + cy);  // 该3D点在图像平面上的投影点
            Eigen::Vector2d e = points_2d[i] - reproj_2d;  // 该3D点在图像平面上的投影点与对应的2D点的误差
            cost += e.squaredNorm();  // 误差的平方和
        }
        std::cout << "Total reprojection error of DLT: " << cost << std::endl;
    }
    
    // 给定2d-3d匹配点对，计算BA问题的累计重投影误差
    double computeReprojErr(const Vecs3d &points_3d, const Vecs2d &points_2d, const Sophus::SE3d& pose, Eigen::Matrix<double, 6, 6>& H, Eigen::Matrix<double, 6, 1>& g){
        double cost = 0;
        for (int i=0; i<points_3d.size(); i++){  // 计算每组2d-3d匹配点对的损失并累加
            Eigen::Vector3d pc = pose * points_3d[i];  // 该3D点在相机坐标系下的坐标
            double inv_z = 1.0 / pc[2];  // 1/z
            double inv_z2 = inv_z * inv_z;  // 1/z^2

            Eigen::Vector2d reproj_2d(fx * pc[0] * inv_z + cx, fy * pc[1] * inv_z + cy);  // 该3D点在图像平面上的投影点
            Eigen::Vector2d e = points_2d[i] - reproj_2d;  // 该3D点在图像平面上的投影点与对应的2D点的误差
            cost += e.squaredNorm();  // 误差的平方和

            Eigen::Matrix<double, 2, 6> J;  // 误差对位姿的雅可比矩阵
            J << -fx * inv_z, 0, fx * pc[0] * inv_z2, fx * pc[0] * pc[1] * inv_z2, -fx - fx * pc[0] * pc[0] * inv_z2, fx * pc[1] * inv_z,
                    0, -fy * inv_z, fy * pc[1] * inv_z2, fy + fy * pc[1] * pc[1] * inv_z2, -fy * pc[0] * pc[1] * inv_z2, -fy * pc[0] * inv_z;

            H += J.transpose() * J;  // Hessian矩阵
            g += -J.transpose() * e;  // 误差项
        }
        return cost;
    }
    
    // 针对BA求解PnP问题的高斯牛顿法优化
    void GaussNewton(const Vecs3d &points_3d, const Vecs2d &points_2d, Sophus::SE3d &pose){
        std::cout << "Gauss-Newton--------------------------\n";
        const int iterations = 100;  // 迭代次数
        double cost = 0, lastCost = 0;
        double eps = 1e-6;  // 迭代停止条件：当误差变化小于eps时停止迭代
        
        // 高斯牛顿迭代
        for (int iter=0; iter<iterations; iter++){
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();  // Hessian矩阵
            Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();  // 误差项

            // 计算损失
            cost = computeReprojErr(points_3d, points_2d, pose, H, g);

            // 求解线性方程 H * dx = g
            Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(g);

            if (isnan(dx[0])) {
                std::cout << "result is nan!" << std::endl;
                break;
            }

            if (iter > (iterations/2) && cost >= lastCost) {  // 损失增长，停止更新
                std::cout << "iter: " << iter << ", cost: " << cost << ", last cost: " << lastCost << std::endl;
                break;
            }

            // 更新估计位姿
            pose = Sophus::SE3d::exp(dx) * pose;
            lastCost = cost;
            std::cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << std::endl;
            
            if (dx.norm() < eps) {  // 更新量足够小，停止迭代
                break;
            }
        }

        std::cout << "Final total reprojection error of G-N: " << cost << std::endl;
        std::cout << "Final average reprojection error of G-N: " << cost / points_2d.size()  << std::endl;

    }

    // 针对BA求解PnP问题的LM法优化
    void LevenbergMarquardt(const Vecs3d &points_3d, const Vecs2d &points_2d, Sophus::SE3d &pose){
        std::cout << "Levenberg-Marquardt--------------------------\n";
        const int iterations = 100;  // 迭代次数
        double cost = 0, lastCost = 0;
        double lambda = 10;  // 缩放系数
        double eps = 1e-6;  // 迭代停止条件：当误差变化小于eps时停止迭代

        // LM迭代
        for (int iter=0; iter<iterations; iter++){
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();  // Hessian矩阵
            Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();  // 误差项

            // 计算损失
            cost = computeReprojErr(points_3d, points_2d, pose, H, g);

            // 求解线性方程 H * dx = g，其中 lambda 是缩放系数
            Eigen::Matrix<double, 6, 6> H_lm = H + lambda * Eigen::Matrix<double, 6, 6>::Identity();
            Eigen::Matrix<double, 6, 1> dx = H_lm.ldlt().solve(g);  // 求解线性方程 H * dx = g

            if (isnan(dx[0])) {
                std::cout << "result is nan!" << std::endl;
                break;
            }

            if (iter > (iterations/2) && cost >= lastCost) {  // 损失增长，停止更新
                std::cout << "iter: " << iter << ", cost: " << cost << ", last cost: " << lastCost << std::endl;
                break;
            }

            // 更新估计位姿
            Sophus::SE3d pose_new = Sophus::SE3d::exp(dx) * pose;
            double cost_new = computeReprojErr(points_3d, points_2d, pose_new, H, g);  // 计算新的损失

            // 根据误差和更新量调整lambda的大小
            if (cost_new < cost) {
                lambda /= 2;
                pose = pose_new;
                lastCost = cost;
                std::cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << std::endl;
            } else {
                lambda *= 2;
            }
        }

        std::cout << "Final total reprojection error of LM: " << cost << std::endl;
        std::cout << "Final average reprojection error of LM: " << cost / points_2d.size()  << std::endl;

    }

};