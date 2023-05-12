#include "Calibrator.hpp"
#include "BAOptimizer.hpp"
#include "Timer.hpp"

int main ( int argc, char** argv )
{
    // 对所有图像进行标定，得到相机内参和畸变系数
    Calibrator calibrator(6, 8, 0.03);
    calibrator.calib();
    cv::Mat K = calibrator.K.clone();
    BAOptimizer optimizer(K);
    int gn_ms = 0, lm_ms = 0, gn_ceres_ms = 0, lm_ceres_ms = 0;  // 各种算法总耗时

    // 对每张图像求解PnP问题
    for (int n=0; n<calibrator.num_images; n++){
        std::cout << "------------------------------ Image index: " << n << " ------------------------------" << std::endl;
        std::cout << "Image name: " << calibrator.img_paths[n] << std::endl;

        // 2d-3d匹配点对从cv::Point2f和cv::Point3f转换为Eigen::Vector2d和Eigen::Vector3d
        std::vector<cv::Point2f> points_2d_cv = calibrator.imagePointsList[n];
        std::vector<cv::Point3f> points_3d_cv = calibrator.objectPointsList[n];
        Vecs2d points_2d_eigen(points_2d_cv.size());
        Vecs3d points_3d_eigen(points_3d_cv.size());
        std::transform(points_2d_cv.begin(), points_2d_cv.end(), points_2d_eigen.begin(), [](const cv::Point2f p){return Eigen::Vector2d(p.x, p.y);});
        std::transform(points_3d_cv.begin(), points_3d_cv.end(), points_3d_eigen.begin(), [](const cv::Point3f p){return Eigen::Vector3d(p.x, p.y, p.z);});

        // opencv标定位姿
        Eigen::Vector3d rvec_eigen, tvec_eigen;  // 将世界坐标系下的点p_w变换到相机坐标系下的点p_c
        cv::cv2eigen(calibrator.rvecs[n], rvec_eigen);
        cv::cv2eigen(calibrator.tvecs[n], tvec_eigen);
        Sophus::SO3d rot = Sophus::SO3d::exp(rvec_eigen);
        Sophus::SE3d pose_cv(rot, tvec_eigen);  
        // std::cout << "Pose of opencv calib: \n" << pose_cv.matrix() << std::endl;
        calibrator.computeReprojectionErrors(n);  // 输出OpenCV标定重投影误差

        // 生成随机的初始相机位姿
        Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
        q.normalize();
        Eigen::Vector3d t = Eigen::Vector3d::Random();
        // Sophus::SE3d pose_gn(q, t), pose_lm(q, t), pose_gn_ceres(q, t), pose_lm_ceres(q, t);
        Sophus::SE3d pose_gn(rot, tvec_eigen), pose_lm(rot, tvec_eigen), pose_gn_ceres(rot, tvec_eigen), pose_lm_ceres(rot, tvec_eigen);

        Timer timer;  // 初始化计时器

        // 高斯牛顿法优化
        optimizer.GaussNewton(points_3d_eigen, points_2d_eigen, pose_gn);
        gn_ms += timer.update("Gauss-Newton");
        // std::cout << "Pose by Gauss-Newton: \n" << pose_gn.matrix() << std::endl;

        // LM 算法优化
        optimizer.LevenbergMarquardt(points_3d_eigen, points_2d_eigen, pose_lm);
        lm_ms += timer.update("Levenberg-Marquardt");
        // std::cout << "Pose by Levenberg-Marquardt: \n" << pose_lm.matrix() << std::endl;

        // Ceres 的高斯牛顿法优化
        optimizer.CeresSolver(points_3d_eigen, points_2d_eigen, pose_gn_ceres);
        gn_ceres_ms += timer.update("Ceres Gauss-Newton");
        // std::cout << "Pose by Ceres Gauss-Newton: \n" << pose_gn_ceres.matrix() << std::endl;

        // Ceres 的 LM 算法优化
        optimizer.CeresSolver(points_3d_eigen, points_2d_eigen, pose_lm_ceres, 1);
        lm_ceres_ms += timer.update("Ceres Levenberg-Marquardt");
        // std::cout << "Pose by Ceres Levenberg-Marquardt: \n" << pose_lm_ceres.matrix() << std::endl;

        // // DLT求解PnP问题
        // Sophus::SE3d pose_dlt;
        // optimizer.DLT(points_3d_eigen, points_2d_eigen, pose_dlt);
        // std::cout << "Pose of DLT: \n" << pose_dlt.matrix() << std::endl;

        // // 以DLT求解结果作为BA位姿初值
        // optimizer.LevenbergMarquardt(points_3d_eigen, points_2d_eigen, pose_dlt);
        // std::cout << "Pose by Gauss-Newton based on DLT: \n" << pose_dlt.matrix() << std::endl;

    }

    std::cout << "Gauss-Newton total time: " << gn_ms << " ms\n";
    std::cout << "Levenberg-Marquardt total time: " << lm_ms << " ms\n";
    std::cout << "Ceres Gauss-Newton total time: " << gn_ceres_ms << " ms\n";
    std::cout << "Ceres Levenberg-Marquardt total time: " << lm_ceres_ms << " ms\n";

}