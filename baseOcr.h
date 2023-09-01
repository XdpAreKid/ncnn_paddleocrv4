#ifndef OCR_BASEOCR_H
#define OCR_BASEOCR_H

#include "baseStruct.h"
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

class baseOcr {
public:
  baseOcr(){};
  virtual ~baseOcr(){};
  virtual bool init() { return false; };
  virtual bool forward(cv::Mat &img, std::vector<TextResult> &results) {
    return false;
  };
};

#endif // !OCR_BASEOCR_H