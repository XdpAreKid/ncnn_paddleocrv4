#ifndef __OCR_DBNET_H__
#define __OCR_DBNET_H__

#include "baseStruct.h"
#include "ncnn/net.h"
#include <memory>
#include <ncnn/mat.h>
#include <opencv2/opencv.hpp>
#include <vector>

class dbNet {
public:
    dbNet();

    ~dbNet() {};


    void initParam(std::string param) {
        net.load_param(param.data());
    };

    void initBin(std::string bin) {
        net.load_model(bin.data());
    };


    bool forward(cv::Mat &src, std::vector<TextBox> &results_);

private:
    ncnn::Net net;
    const float meanValues[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
    const float normValues[3] = {1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0,
                                 1.0 / 0.225 / 255.0};
    float boxThresh = 0.3f;
    float boxScoreThresh = 0.5f;
    float unClipRatio = 2.0f;

};

#endif //__OCR_DBNET_H__
