#include <iostream>
#include "opencv2/freetype.hpp"
#include "crnnNet.h"
#include "dbNet.h"
#include "tools.h"

int main() {
    dbNet detectNet;
    crnnNet recNet;
    detectNet.initParam("./data/det.param");
    detectNet.initBin("./data/det.bin");
    recNet.initParam("./data/ch_recv4.ncnn.param");
    recNet.initBin("./data/ch_recv4.ncnn.bin");
    recNet.initKeys("./data/dict_chi_sim.txt");
    cv::Mat img = cv::imread("./data/R-C.jpeg");
    cv::Mat drawImg = img.clone();
    std::vector<TextBox> boxResult;
    std::vector<TextLine> recResult;
    detectNet.forward(img, boxResult);
    recResult.resize(boxResult.size());

    cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData("./data/LXGWWenKai-Light.ttf", 0);
    for (size_t i = 0; i < boxResult.size(); i++) {
        cv::Mat partImg = getRotateCropImage(img, boxResult[i].boxPoint);
        recNet.forward(partImg, recResult[i]);
        cv::polylines(drawImg, boxResult[i].boxPoint, true, cv::Scalar(0,255,0));
//        cv::putText(drawImg, recResult[i].text, boxResult[i].boxPoint[0], cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(0,0,255));
        ft2->putText(drawImg, recResult[i].text, boxResult[i].boxPoint[0],100, cv::Scalar(0, 0, 255), -1, 8, true);
    }
    cv::imwrite("result.png", drawImg);

    return 0;
}
