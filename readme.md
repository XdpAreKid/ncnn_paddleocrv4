# ncnn_paddleocrv4

This is a sample paddleocr ncnn project

## how to build and run
1. git clone https://github.com/XdpAreKid/ncnn_paddleocrv4.git
2. cd ncnn_paddleocrv4 && mkdir build && cd build
3. cmake .. && make -j8 
4. cp -r ../data ./
5. ./test_paddle

# screenshot
![](result.png) 

# convert model
1. dbModel use paddle->onnx->ncnn, use [this](https://github.com/Tencent/ncnn/pull/4975) to simply model
2. crnnModel use [this](https://github.com/frotms/PaddleOCR2Pytorch)
3. maybe intel gpu may get error result, see [here](https://github.com/Tencent/ncnn/issues/4986)

# according
1.https://github.com/FeiGeChuanShu/ncnn_paddleocr
2.https://github.com/frotms/PaddleOCR2Pytorch  
3.https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7  
4.https://github.com/lxgw/LxgwWenKai