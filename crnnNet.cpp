#include "crnnNet.h"
#include "mat.h"
#include <cmath>
#include <fstream>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <thread>

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

TextLine crnnNet::scoreToTextLine(const std::vector<float> &outputData, int h,
                                  int w) {
    int keySize = keys.size();
    std::string strRes;
    std::vector<float> scores;
    int lastIndex = -1;
    int maxIndex;
    float maxValue;

    for (int i = 0; i < h; i++) {
        maxIndex = 0;
        maxValue = -1000.f;

        maxIndex =
                int(argmax(outputData.begin() + i * w, outputData.begin() + i * w + w));
        maxValue = float(
                *std::max_element(outputData.begin() + i * w,
                                  outputData.begin() + i * w + w)); // / partition;
        if (maxIndex > 0 && maxIndex < keySize && (!(maxIndex == lastIndex))) {
            scores.emplace_back(maxValue);
            strRes.append(keys[maxIndex]);
        }
        lastIndex = maxIndex;
    }
    return {strRes, scores};
}


bool crnnNet::forward(cv::Mat &src, TextLine &result) {
    int resized_w = 0;
    float ratio = src.cols / float(src.rows);


    resized_w = ceil(dstHeight * ratio);

    cv::Size tmp = cv::Size(resized_w, dstHeight);

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_BGR2RGB,
                                             src.cols, src.rows, tmp.width, tmp.height);

    input.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = net.create_extractor();
    extractor.input("in0", input);
    ncnn::Mat out;
    extractor.extract("out0", out);
    float *floatArray = (float *) out.data;
    std::vector<float> outputData(floatArray, floatArray + out.h * out.w);
    result = scoreToTextLine(outputData, out.h, out.w);

    return true;
}

bool crnnNet::forward(std::vector<cv::Mat> &src,
                      std::vector<TextLine> &results) {
    int sizeLen = src.size();
    // results.resize(sizeLen);
    for (size_t i = 0; i < sizeLen; i++) {
        TextLine textline;
        if (forward(src[i], textline)) {
            results.emplace_back(textline);
        } else {
            return false;
        }
    }
    return true;
}

crnnNet::crnnNet() {
    net.opt.use_vulkan_compute = false;
}
