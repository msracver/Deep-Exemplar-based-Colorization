// Copyright (c) Microsoft. All rights reserved.

// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include "Combo.cuh"
#include "Classifier.h"
#include <boost/filesystem.hpp>

#define MAX_LEN 1024

int ComputeCombo(int argc, char** argv)
{
	string modelDir = argv[1];
	string rootDir = argv[2];
	int sid = atoi(argv[3]);
	int eid = atoi(argv[4]);
	int gid = atoi(argv[5]);

	string fname = rootDir + "/pairs.txt";
	FILE* fp = fopen(fname.c_str(), "r");
	char name0[260], name1[260];
	float score = 0.f;
	int val = 1;
	for (int i = 0; i < sid; ++i)
	{
		val = fscanf(fp, "%s %s %f\n", name0, name1, &score);
		if (val == EOF) break;
	}

	if (val == EOF)
		return 0;

	Combo dp;

	dp.SetGPU(gid);

	string inputDir = rootDir + "/input/";
	string outputDir_flow = rootDir + "/flow/";
	string outputDir_combo = rootDir + "/combo_new/";
	if (!boost::filesystem::exists(outputDir_combo))
	{
		boost::filesystem::create_directory(outputDir_combo);
	}

	string model_file = "/vgg_19_gray_bn/deploy.prototxt";
	string trained_file = "/vgg_19_gray_bn/vgg19_bn_gray_ft_iter_150000.caffemodel";

	::google::InitGoogleLogging("ComputeCombo");

	Classifier classifier_A(modelDir + model_file, modelDir + trained_file);
	Classifier classifier_B(modelDir + model_file, modelDir + trained_file);

	for (int i = sid; i < eid; ++i)
	{
		int val = fscanf(fp, "%s %s %f\n", name0, name1, &score);

		printf("Info: Read line #%d, image1 = %s, image2 = %s.\n", i, name0, name1);
		if (val == EOF) break;

		string name0Str(name0);
		string name1Str(name1);

		int pos0 = name0Str.find_last_of(".");
		int pos1 = name1Str.find_last_of(".");
		if (name0Str.length() - pos0 <= 4)
		{
			name0Str = name0Str.substr(0, pos0) + ".jpg";
			name1Str = name1Str.substr(0, pos1) + ".jpg";
		}

		string A = inputDir + name0Str;
		string BP = inputDir + name1Str;

		pos0 = name0Str.find_last_of(".");
		pos1 = name1Str.find_last_of(".");
		string name_A = name0Str.substr(0, pos0);
		string name_B = name1Str.substr(0, pos1);

		char fileName0[260];
		char fileName1[260];
		char ffileName0[260];
		char ffileName1[260];

		// load images
		bool isOKA = dp.LoadA(A.c_str());
		bool isOKB = dp.LoadBP(BP.c_str());
		if (!isOKA)
		{
			printf("Error: Fail reading image1: %s!\n", A.c_str());
			continue;
		}

		if (!isOKB)
		{
			printf("Error: Fail reading image2: %s!\n", BP.c_str());
			continue;
		}

		int aw, ah, bw, bh;
		dp.GetASize(aw, ah);
		dp.GetBPSize(bw, bh);

		if (aw > MAX_LEN || ah > MAX_LEN)
		{
			printf("Error: Unsupported image1's size (long edge > 1024): w = %d, h = %d!\n", aw, ah);
			continue;
		}

		if (bw > MAX_LEN || bh > MAX_LEN)
		{
			printf("Error: Unsupported image size (long edge > 1024): w = %d, h = %d!\n", bw, bh);
			continue;
		}

		// first detect if flow exits
		sprintf(ffileName0, "%s/%s_%s.txt", outputDir_flow.c_str(), name_A.c_str(), name_B.c_str());

		sprintf(ffileName1, "%s/%s_%s.txt", outputDir_flow.c_str(), name_B.c_str(), name_A.c_str());

		if (!boost::filesystem::exists(ffileName0))
		{
			printf("Error: Flow %s does not exist!\n", ffileName0);
			continue;
		}

		if (!boost::filesystem::exists(ffileName1))
		{
			printf("Error: Flow %s does not exist!\n", ffileName1);
			continue;
		}

		// detect if flow is valid
		printf("Info: Flows exist.\n", i);

		sprintf(fileName0, "%s/%s_%s.combo", outputDir_combo.c_str(), name_A.c_str(), name_B.c_str());
		sprintf(fileName1, "%s/%s_%s.combo", outputDir_combo.c_str(), name_B.c_str(), name_A.c_str());

		FILE* fp_a = fopen(fileName0, "wb");
		FILE* fp_b = fopen(fileName1, "wb");

		dp.ComputeDist(classifier_A, classifier_B, fp_a, fp_b, ffileName0, ffileName1);

		fclose(fp_a);
		fclose(fp_b);

		printf("Info: Create %s and %s.\n\n", fileName0, fileName1);
	}

	google::ShutdownGoogleLogging();

	classifier_B.DeleteNet();
	classifier_A.DeleteNet();
	fclose(fp);

	return 0;
}

int main(int argc, char** argv) 
{
	ComputeCombo(argc, argv);
	return 0;
}
