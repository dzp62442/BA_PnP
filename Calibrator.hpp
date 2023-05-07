#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

class Calibrator
{
public:
    int num_images = 0;
    cv::Size imageSize;
    std::vector<std::string> img_paths, img_names;
    cv::Size boardSize;  // 内角点行列数
    float squareSize;  // 棋盘格方格大小
    
    std::string calib_imgs_dir = "../imgs/calib_imgs/";
    std::string draw_imgs_dir = "../imgs/draw_imgs/";
    cv::Mat K, distCoeffs;  // 内参和畸变系数
    std::vector<cv::Mat> rvecs, tvecs;  // 外参
    std::vector<std::vector<cv::Point2f>> imagePointsList;  // 每张图像的2D像素点坐标
    std::vector<std::vector<cv::Point3f>> objectPointsList;  // 每张图像对应的3D空间点坐标

public:
    Calibrator(int rows, int cols, float distance): boardSize(rows, cols), squareSize(distance) {}
    ~Calibrator() {}
    
    // 获取文件夹中的文件名列表和文件数目
    bool readCalibDir(){
        std::filesystem::path dir(calib_imgs_dir);
        if (!std::filesystem::exists(dir)){
            std::cout << "Error: dir not exists!" << std::endl;
            return false;
        }
        if (!std::filesystem::is_directory(dir)){
            std::cout << "Error: dir is not a directory!" << std::endl;
            return false;
        }
        for (const auto & entry : std::filesystem::directory_iterator(dir)){
            img_paths.push_back(entry.path().string());
            img_names.push_back(entry.path().filename().string());
            num_images++;
        }
        return true;
    }

    // 标定相机
    bool calib(){
        // 准备标定板上的世界坐标系三维点坐标
        std::vector<cv::Point3f> objectPoints;
        for (int i = 0; i < boardSize.height; ++i) {
            for (int j = 0; j < boardSize.width; ++j) {
                objectPoints.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
            }
        }

        // 准备标定图像和对应的角点
        readCalibDir();        
        for (int i=0; i<num_images; ++i){
            // std::cout << img_paths[i] << std::endl;
            cv::Mat image = cv::imread(img_paths[i]);  // 读取棋盘格图像
            imageSize = image.size();

            // 检测角点
            std::vector<cv::Point2f> corners;
            bool found = cv::findChessboardCorners(image, boardSize, corners);

            if (found) {
                // 提取角点的亚像素坐标
                cv::Mat gray;
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
                cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
                cv::drawChessboardCorners(image, boardSize, corners, found);
                cv::imwrite(draw_imgs_dir + img_names[i], image);

                // 保存图像点和对应的三维点坐标
                imagePointsList.push_back(corners);
                objectPointsList.push_back(objectPoints);
            }
            else{
                std::cout << "Error: cannot find chessboard corners: " << img_paths[i] << std::endl;
                return false;
            }
        }
        
        // 标定相机
        cv::calibrateCamera(objectPointsList, imagePointsList, imageSize, K, distCoeffs, rvecs, tvecs);

        // 输出标定结果
        std::cout << "K = " << K << std::endl;
        std::cout << "distCoeffs = " << distCoeffs << std::endl;
        return true;
    }

    // 计算指定图像的重投影误差
    void computeReprojectionErrors(int img_idx){
        std::vector<cv::Point2f> reprojected_points;  // 重投影后的2d点
        cv::Mat D = cv::Mat::zeros(5, 1, CV_64F);
        cv::projectPoints(objectPointsList[img_idx], rvecs[img_idx], tvecs[img_idx], K, D, reprojected_points);
        double totalErr = 0;  // 该图像的总误差
        for (int i=0; i<reprojected_points.size(); i++){
            double err = cv::norm(imagePointsList[img_idx][i] - reprojected_points[i]);
            totalErr += err * err;
        }
        std::cout << "Total reprojection error of opencv calib: " << totalErr << std::endl;
        std::cout << "Average reprojection error of opencv calib: " << totalErr / reprojected_points.size() << std::endl;
    }
};