// Copyright (c) Microsoft. All rights reserved.

// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#ifndef COMBO_H
#define COMBO_H
#include <string>
#include "opencv2/opencv.hpp"
#include "Classifier.h"
using namespace std;
class Combo
{
public:

	Combo();
	~Combo();
	
	void SetGPU(int no);

	bool LoadA(const char* file_A);
	bool LoadBP(const char* file_BP);
	
	void GetASize(int& width, int& height);
	void GetBPSize(int& width, int& height);
	
	void ComputeDist(Classifier& classifier_A, Classifier& classifier_B, FILE* fp_a, FILE* fp_b, const char* ff_a, const char* ff_b);

private:
	cv::Mat img_AL, img_BPL, img_AL_col, img_BPL_col;

};

#endif