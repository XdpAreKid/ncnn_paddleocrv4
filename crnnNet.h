#ifndef __OCR_CRNNNET_H__
#define __OCR_CRNNNET_H__

#include "baseStruct.h"
#include "ncnn/net.h"
#include <iostream>
#include <memory>
#include <ncnn/mat.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>
#include <fstream>

class crnnNet {
public:
    crnnNet();

    ~crnnNet() {};

    void initParam(std::string param) {
        net.load_param(param.data());
    };


    void initBin(std::string bin) {
        net.load_model(bin.data());
    };

    void initKeys(std::string keyTxt) {
        std::ifstream in(keyTxt.c_str());
        std::string line;
        if (in) {
            while (getline(in, line)) {// line中不包括每行的换行符
                keys.push_back(line);
            }
        } else {
            printf("The keys.txt file was not found\n");
        }
        keys.insert(keys.begin(), "#");
        keys.emplace_back(" ");
    };


    bool forward(cv::Mat &src, TextLine &result);

    bool forward(std::vector<cv::Mat> &src, std::vector<TextLine> &results);

private:
    TextLine scoreToTextLine(const std::vector<float> &outputData, int h, int w);

private:
    ncnn::Net net;
    const int dstHeight = 48;
    const int dstWidth = 320;
    const float meanValues[3] = {127.5, 127.5, 127.5};
    const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};

    std::vector<std::string> keys;
};

#endif //__OCR_DBNET_H__