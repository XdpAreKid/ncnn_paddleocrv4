#include "dbNet.h"
#include "mat.h"
#include "tools.h"
#include <cstdio>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>
#include <vector>


std::vector<TextBox> inline findRsBoxes(const cv::Mat &fMapMat,
                                        const cv::Mat &norfMapMat,
                                        const float boxScoreThresh,
                                        const float unClipRatio) {
    const float minArea = 3;
    std::vector<TextBox> rsBoxes;
    rsBoxes.clear();
    // printf("norfmapmat: %d", norfMapMat.type());
    // cv::imwrite("tmp.png", norfMapMat);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(norfMapMat, contours, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i) {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox =
                getMinBoxes(contours[i], minSideLen, perimeter);
        if (minSideLen < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> clipBox = unClip(minBox, perimeter, unClipRatio);
        std::vector<cv::Point> clipMinBox =
                getMinBoxes(clipBox, minSideLen, perimeter);
        //---use clipper end---

        if (minSideLen < minArea + 2)
            continue;

        for (int j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = (clipMinBox[j].x / 1.0);
            clipMinBox[j].x =
                    (std::min)((std::max)(clipMinBox[j].x, 0), norfMapMat.cols);

            clipMinBox[j].y = (clipMinBox[j].y / 1.0);
            clipMinBox[j].y =
                    (std::min)((std::max)(clipMinBox[j].y, 0), norfMapMat.rows);
        }

        rsBoxes.emplace_back(TextBox{clipMinBox, score});
    }
    reverse(rsBoxes.begin(), rsBoxes.end());

    return rsBoxes;
}

bool dbNet::forward(cv::Mat &src, std::vector<TextBox> &results_) {
    int width = src.cols;
    int height = src.rows;
    int target_size = 640;
    int w = width;
    int h = height;
    float scale = 1.f;
    const int resizeMode = 0; // min = 0, max = 1

    if (resizeMode == 1) {
        if (w < h) {
            scale = (float) target_size / w;
            w = target_size;
            h = h * scale;
        } else {
            scale = (float) target_size / h;
            h = target_size;
            w = w * scale;
        }
    } else if (resizeMode == 0) {
        if (w > h) {
            scale = (float) target_size / w;
            w = target_size;
            h = h * scale;
        } else {
            scale = (float) target_size / h;
            w = w * scale;
            h = target_size;
        }
    }
    ncnn::Extractor extractor = net.create_extractor();

    ncnn::Mat out;
    cv::Size in_pad_size;
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;

    ncnn::Mat in_pad_;
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
            src.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    // pad to target_size rectangle

    ncnn::copy_make_border(input, in_pad_, hpad / 2, hpad - hpad / 2, wpad / 2,
                           wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    in_pad_.substract_mean_normalize(meanValues, normValues);
    in_pad_size = cv::Size(in_pad_.w, in_pad_.h);
    extractor.input("x", in_pad_);
    extractor.extract("sigmoid_0.tmp_0", out);




//    ncnn::Mat flattened_out = out.reshape(out.w * out.h * out.c);
    //-----boxThresh-----
    cv::Mat fMapMat(in_pad_size.height, in_pad_size.width, CV_32FC1, (float *) out.data);
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(norfMapMat, norfMapMat, element, cv::Point(-1, -1), 1);

    std::vector<TextBox> results =
            findRsBoxes(fMapMat, norfMapMat, boxScoreThresh,
                        unClipRatio);
    for (int i = 0; i < results.size(); i++) {
        for (int j = 0; j < results[i].boxPoint.size(); j++) {
            float x = float(results[i].boxPoint[j].x - (wpad / 2)) / scale;
            float y = float(results[i].boxPoint[j].y - (hpad / 2)) / scale;
            x = std::max(std::min(x, (float) (width - 1)), 0.f);
            y = std::max(std::min(y, (float) (height - 1)), 0.f);
            results[i].boxPoint[j].x = (int) x;
            results[i].boxPoint[j].y = (int) y;
        }
        if (abs(results[i].boxPoint[0].x - results[i].boxPoint[1].x) <= 3) {
            continue;
        }
        if (abs(results[i].boxPoint[0].y - results[i].boxPoint[3].y) <= 3) {
            continue;
        }
        results_.push_back(results[i]);
    }

    return true;
}

dbNet::dbNet() {
    net.opt.use_vulkan_compute = false;
    net.opt.lightmode = true;

}
