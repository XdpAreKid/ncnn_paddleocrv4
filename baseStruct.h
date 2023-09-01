#ifndef __OCR_STRUCT_H__
#define __OCR_STRUCT_H__
#include <algorithm>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>


struct TextBox {
  std::vector<cv::Point> boxPoint;
  float score;
};

struct TextLine {
  std::string text;
  std::vector<float> charScores;
};


#endif