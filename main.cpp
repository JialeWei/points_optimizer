#include <cmath>
#include <memory>
#include <tuple>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include<opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>


const int sample_size = 200;
const double u = 10.0;
const double v = 5.0;
const double theta = 0.5;

template<typename T>
Eigen::Matrix<T, 2, 2> getRotationMatrix(T angle) {
    Eigen::Matrix<T, 2, 2> R;
    R << cos(angle), -sin(angle),
            sin(angle), cos(angle);
    return R;
}

bool pointIsValid(int r, int c, int height, int width) {
    if (abs(r) > height) return false;
    if (abs(c) > width) return false;
    if (r < 0) return false;
    if (c < 0) return false;
    return true;
}

struct CostFunctor {
    cv::Mat mask_;
    Eigen::Vector2d p_original_;
    int height_, width_;
    Eigen::MatrixXd e_mask_;
    std::unique_ptr<ceres::Grid2D<double, 1> > grid;
    std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > > interpolator;

    CostFunctor(const cv::Mat &mask, const Eigen::Vector2d &point) {
        mask_ = mask;
        p_original_ = point;
        height_ = mask.rows;
        width_ = mask.cols;

        cv2eigen(mask_, e_mask_);


        grid = std::make_unique<ceres::Grid2D<double, 1>>(
                e_mask_.data(), 0, e_mask_.cols(), 0, e_mask_.rows());
        interpolator = std::make_unique<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > >(*grid);
    }

    bool operator()(const double *const t_and_r, double *residual) const {
        Eigen::Matrix<double, 2, 2> R = getRotationMatrix(t_and_r[2]);
        const double point_new_r =
                R(0) * (p_original_(0) - t_and_r[0]) + R(1) * (p_original_(1) - t_and_r[1]);
        const double point_new_c =
                R(2) * (p_original_(0) - t_and_r[0]) + R(3) * (p_original_(1) - t_and_r[1]);
        int pix_r = floor(point_new_r);
        int pix_c = floor(point_new_c);
        if (pointIsValid(pix_r, pix_c, height_, width_)) {
            double dist;
            interpolator->Evaluate(point_new_c, point_new_r, &dist);
//            double d;
//            cv::Mat image = cv::Mat::zeros(height_, width_, CV_8U);
//            for (int i = 0; i < height_; i++){
//                for (int j = 0; j < width_; j++){
//                    interpolator->Evaluate(j, i, &d);
//                    image.at<uchar>(i, j) = d;
//                }
//            }
//            cv::imwrite("/home/wjl/out/test.png", image);
            residual[0] = dist * dist;
        } else residual[0] = 1e4;
        return true;
    }
};

std::tuple<double, double, double> pointsOptimizer(const cv::Mat &mask, const Eigen::MatrixXi &points) {
    double t_and_r[3] = {0.0, 0.0, 0.0};
    ceres::Problem problem;
    for (int i = 0; i < sample_size; i++) {
        double r = points(0, i);
        double c = points(1, i);

        Eigen::Vector2d point(r, c);
        ceres::CostFunction *cost_function = new ceres::NumericDiffCostFunction<CostFunctor, ceres::CENTRAL, 1, 3>(
                new CostFunctor(mask, point));
        problem.AddResidualBlock(cost_function, nullptr, t_and_r);
    }
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "u:" << t_and_r[0] << std::endl;
    std::cout << "v:" << t_and_r[1] << std::endl;
    std::cout << "theta:" << t_and_r[2] * 180 / M_PI << std::endl;
    return std::make_tuple(t_and_r[0], t_and_r[1], t_and_r[2]);
}

void eval(const std::vector<cv::Point> &gt_points, const Eigen::MatrixXi &random_points, double tran_u, double tran_v,
          double rot) {
    std::vector<int> gt_points_r, gt_points_c;
    Eigen::Matrix<double, 2, 2> R = getRotationMatrix(rot);
    double mae = 0.0, rmse = 0.0;
    for (int i = 0; i < sample_size; i++) {
        double r = random_points(0, i);
        double c = random_points(1, i);
        int new_r = floor(R(0) * (r - tran_u) + R(1) * (c - tran_v));
        int new_c = floor(R(2) * (r - tran_u) + R(3) * (c - tran_v));
        int gt_r = gt_points[i].y;
        int gt_c = gt_points[i].x;
        mae = mae + fabs(new_r - gt_r) + fabs(new_c - gt_c);
        rmse = rmse + (new_r - gt_r) * (new_r - gt_r) + (new_c - gt_c) * (new_c - gt_c);
    }
    std::cout << "MAE: " << mae / sample_size << std::endl;
    std::cout << "RMSE: " << sqrt(rmse / sample_size) << std::endl;
}

cv::Mat
drawCirclesAfterOpt(const cv::Mat &src, const Eigen::MatrixXi &points, double tran_u, double tran_v, double rot) {
    Eigen::Matrix<double, 2, 2> R = getRotationMatrix(rot);
    for (int i = 0; i < sample_size; i++) {
        double r = points(0, i);
        double c = points(1, i);
        cv::Point pt;
        pt.y = floor(R(0) * (r - tran_u) + R(1) * (c - tran_v));
        pt.x = floor(R(2) * (r - tran_u) + R(3) * (c - tran_v));
        cv::circle(src, pt, 5, cv::Scalar(0, 255, 0), 1, cv::LINE_8);
    }
    return src;
}

std::tuple<cv::Mat, Eigen::MatrixXi, std::vector<cv::Point>>
generateRandomPoints(const cv::Mat &src, const std::vector<std::vector<cv::Point>> &contours) {
    cv::Mat src_bgr;
    cv::cvtColor(src, src_bgr, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point> gt_points, sampled_gt_points;
    Eigen::MatrixXi sampledPointsMatrix;

    for (auto &contour : contours) {
        for (auto &point : contour) {
            gt_points.push_back(point);
        }
    }
    if (gt_points.size() >= sample_size) {
        //sampled ground-truth points(draw with circle)
        cv::randShuffle(gt_points);
        for (int i = 0; i < sample_size; i++) {
            sampled_gt_points.push_back(gt_points[i]);
            //circle(src_bgr, sampled_gt_points[i], 5, Scalar(0, 0, 255), 1, cv::LINE_8);
        }

        //with rotation and transition
        std::vector<int> points_r, points_c;
        for (auto &sampled_gt_point : sampled_gt_points) {
            points_r.push_back(sampled_gt_point.y);
            points_c.push_back(sampled_gt_point.x);
        }
        Eigen::MatrixXi xMat = Eigen::RowVectorXi::Map(&points_r[0], points_r.size());
        Eigen::MatrixXi yMat = Eigen::RowVectorXi::Map(&points_c[0], points_c.size());
        sampledPointsMatrix.resize(2, sample_size);
        sampledPointsMatrix << xMat,
                yMat;
        Eigen::Matrix<double, 2, 2> R = getRotationMatrix(theta * M_PI / 180);
        for (int i = 0; i < sample_size; i++) {
            sampledPointsMatrix(0, i) = floor(R(0) * sampledPointsMatrix(0, i)
                                              + R(1) * sampledPointsMatrix(1, i) + u);
            sampledPointsMatrix(1, i) = floor(R(2) * sampledPointsMatrix(0, i)
                                              + R(3) * sampledPointsMatrix(1, i) + v);
        }

        //draw sampled points with circle
        for (int i = 0; i < sample_size; i++) {
            cv::Point pt;
            pt.x = sampledPointsMatrix(1, i);
            pt.y = sampledPointsMatrix(0, i);
            cv::circle(src_bgr, pt, 5, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
        }
        return std::make_tuple(src_bgr, sampledPointsMatrix, sampled_gt_points);
    } else return std::make_tuple(src, sampledPointsMatrix, sampled_gt_points);
}

cv::Mat getDistanceTransform(const cv::Mat &src) {
    cv::Mat dist, dst;
    cv::distanceTransform(src, dist, cv::DIST_L2, 3);
    cv::normalize(dist, dst, 0, 255, cv::NORM_MINMAX);
    return dst;
}

std::vector<std::vector<cv::Point> > getContours(const cv::Mat &src) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

cv::Mat drawContours(const cv::Mat &src, const std::vector<std::vector<cv::Point> > &contours) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8U);
    std::vector<cv::Vec4i> hierarchy;
    for (int i = 0; i < contours.size(); i++) {
        cv::drawContours(dst, contours, (int) i, 255, 1, cv::LINE_8, hierarchy, 0);
    }
    return dst;
}

cv::Mat drawMasks(cv::Mat src, int k) {
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8U);
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            if (int(src.at<uchar>(i, j)) == k) {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
    return dst;
}

int main() {
    std::string pic_path = "/home/wjl/Documents/0000002552.png";
    cv::Mat src = cv::imread(pic_path, 0);

    double minVal;
    double maxVal;

    cv::minMaxLoc(src, &minVal, &maxVal, nullptr, nullptr);

    for (int k = 0; k < maxVal - minVal + 1; k++) {
        cv::Mat img_mask = drawMasks(src, k);
        cv::imwrite("/home/wjl/out/masks/label" + std::to_string(k) + ".png", img_mask);

        std::vector<std::vector<cv::Point>> contours = getContours(img_mask);
        cv::Mat img_contours = drawContours(src, contours);
        cv::imwrite("/home/wjl/out/contours/label" + std::to_string(k) + ".png", img_contours);

        cv::Mat img_dt_masks = getDistanceTransform(img_mask);
        cv::imwrite("/home/wjl/out/dt_masks/label" + std::to_string(k) + ".png", img_dt_masks);

        cv::Mat img_dt_contours = getDistanceTransform(img_contours);
        cv::imwrite("/home/wjl/out/dt_contours/label" + std::to_string(k) + ".png", img_dt_contours);

        cv::Mat img_circles;
        Eigen::MatrixXi random_points;
        std::vector<cv::Point> gt_points;
        std::tie(img_circles, random_points, gt_points) = generateRandomPoints(img_dt_masks, contours);
        cv::imwrite("/home/wjl/out/circles/label" + std::to_string(k) + ".png", img_circles);

        if (random_points.size() != 0) {
            std::cout << "label" + std::to_string(k) << std::endl;
            double tran_u, tran_v, rot;
            std::tie(tran_u, tran_v, rot) = pointsOptimizer(img_dt_masks, random_points);
            cv::Mat img_points_opt = drawCirclesAfterOpt(img_circles, random_points, tran_u, tran_v, rot);
            eval(gt_points, random_points, tran_u, tran_v, rot);
            cv::imwrite("/home/wjl/out/circles/label" + std::to_string(k) + ".png", img_points_opt);
            std::cout << "....................................." << std::endl;
        }
    }

}