#include <iostream>
#include <string>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> Vecs2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> Vecs3d;


// 处理SE3变换的自定义局部参数化
class SE3Parameterization : public ceres::LocalParameterization {
public:
    virtual ~SE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const {
        Eigen::Map<const Eigen::Vector3d> trans(x + 4);
        Eigen::Map<const Eigen::Quaterniond> quat(x);

        Eigen::Map<const Eigen::Vector3d> delta_trans(delta + 3);
        Eigen::Map<const Eigen::Quaterniond> delta_quat(delta);

        Eigen::Map<Eigen::Quaterniond> quat_plus_delta(x_plus_delta);
        Eigen::Map<Eigen::Vector3d> trans_plus_delta(x_plus_delta + 4);

        quat_plus_delta = (quat * delta_quat).normalized();
        trans_plus_delta = trans + delta_trans;

        return true;
    }

    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const {
        ceres::MatrixRef(jacobian, 7, 6) = ceres::Matrix::Identity(7, 6);
        return true;
    }

    virtual int GlobalSize() const { return 7; }
    virtual int LocalSize() const { return 6; }
};

// BA 优化求解器
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
    
    //! -------------------------------------- 手写求解器 -------------------------------------- //
    
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
        double eps = 1e-3;  // 迭代停止条件：当误差变化小于eps时停止迭代
        
        // 高斯牛顿迭代
        int iter;
        for (iter=0; iter<iterations; iter++){
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
            // std::cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << std::endl;
            
            if (dx.norm() < eps) {  // 更新量足够小，停止迭代
                break;
            }
        }

        std::cout << "Final total and average error after " << iter << " iterations of G-N: " << cost << ", " << cost / points_2d.size() << std::endl;
    }

    // 针对BA求解PnP问题的LM法优化
    void LevenbergMarquardt(const Vecs3d &points_3d, const Vecs2d &points_2d, Sophus::SE3d &pose){
        std::cout << "Levenberg-Marquardt--------------------------\n";
        const int iterations = 100;  // 迭代次数
        double cost = 0, lastCost = 0;
        double lambda = 10;  // 缩放系数
        double eps = 1e-3;  // 迭代停止条件：当误差变化小于eps时停止迭代

        // LM迭代
        int iter;
        for (iter=0; iter<iterations; iter++){
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
                // std::cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << std::endl;
            } else {
                lambda *= 2;
            }
            
            if (dx.norm() < eps) {  // 更新量足够小，停止迭代
                break;
            }
        }

        std::cout << "Final total and average error after " << iter << " iterations of LM: " << cost << ", " << cost / points_2d.size() << std::endl;

    }

    //! -------------------------------------- 基于Ceres的求解器 -------------------------------------- //

    // 定义结构体作为代价函数
    struct PnPResidual {
        const Eigen::Vector2d point_2d;
        const Eigen::Vector3d point_3d;
        double fx, fy, cx, cy;

        PnPResidual(const Eigen::Vector2d& observed, const Eigen::Vector3d& world, double fx, double fy, double cx, double cy)
            : point_2d(observed), point_3d(world), fx(fx), fy(fy), cx(cx), cy(cy) {}

        template <typename T>
        bool operator()(const T* const camera, T* residuals) const {
            // 提取平移和旋转
            Eigen::Matrix<T, 3, 1> trans(camera[0], camera[1], camera[2]);
            Eigen::Quaternion<T> q(camera[6], camera[3], camera[4], camera[5]); // 注意这里的四元数初始化是(w, x, y, z)
            Eigen::Quaternion<T> rot = q.normalized();

            // 构建 SE3 对象
            Sophus::SE3<T> pose(rot, trans);

            // 将 3D 点投影到 2D 点
            Eigen::Matrix<T, 3, 1> pc = pose * point_3d.template cast<T>();  // 该3D点在相机坐标系下的坐标
            T inv_z = 1.0 / pc[2];  // 1/z
            T inv_z2 = inv_z * inv_z;  // 1/z^2
            Eigen::Matrix<T, 2, 1> reproj_2d(fx * pc[0] * inv_z + cx, fy * pc[1] * inv_z + cy);  // 该3D点在图像平面上的投影点
            
            // 计算残差
            residuals[0] = reproj_2d[0] - T(point_2d[0]);
            residuals[1] = reproj_2d[1] - T(point_2d[1]);

            return true;
        }
    };

    // 使用 Ceres 实现两种算法求解 PnP 问题
    void CeresSolver(const Vecs3d& points_3d, const Vecs2d& points_2d, Sophus::SE3d& pose, const int flag=0){
        const int num_points = points_3d.size();
        ceres::Problem problem;  // 创建 ceres 问题

        // 添加残差块
        for (int i = 0; i < num_points; ++i) {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<PnPResidual, 2, 7>(
                    new PnPResidual(points_2d[i], points_3d[i], fx, fy, cx, cy));

            ceres::LocalParameterization* se3_parameterization =
                new SE3Parameterization();

            problem.AddParameterBlock(pose.data(), 7, se3_parameterization);
            problem.AddResidualBlock(cost_function, nullptr, pose.data());
        }


        // 设置 Ceres 求解器选项
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 100;
        options.function_tolerance = 1e-20;
        options.gradient_tolerance = 1e-20;

        if (flag == 1){  // flag = 1 使用LM算法
            options.minimizer_type = ceres::TRUST_REGION; 
        }
        else if (flag != 0){  // falg = 0默认使用高斯牛顿法
            std::cerr << "Ceres solver type error !\n";
        }

        // 求解 PnP 问题
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";
    
    }

};

