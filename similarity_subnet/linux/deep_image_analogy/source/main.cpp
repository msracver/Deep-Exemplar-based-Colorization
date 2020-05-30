#include "DeepAnalogy.cuh"
#include "Classifier.h"
#include <boost/filesystem.hpp>

void run_flow(int argc, char** argv)
{
	string modelDir = argv[1];
	string rootDir = argv[2];

	int sid = atoi(argv[3]);
	int eid = atoi(argv[4]);
	int gid = atoi(argv[5]);

	string postfix = "";
	if (argc >= 8)
	{
		postfix = "_" + string(argv[7]);
	}

	string fname = rootDir + "/pairs" + postfix + ".txt";
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
		return;

	DeepAnalogy dp;
	dp.SetModel(modelDir);
	dp.SetGPU(gid);

	string inputDir = rootDir + "/input" + postfix + "/";
	string outputDir_flow = rootDir + "/flow" + postfix + "/";

	if (!boost::filesystem::exists(outputDir_flow))
	{
		boost::filesystem::create_directory(outputDir_flow);
	}

	::google::InitGoogleLogging("deepanalogy");

	string model_file = "vgg19/VGG_ILSVRC_19_layers_deploy.prototxt";
	string trained_file = "vgg19/VGG_ILSVRC_19_layers.caffemodel";

	Classifier classifier_A(modelDir + model_file, modelDir + trained_file);
	Classifier classifier_B(modelDir + model_file, modelDir + trained_file);

	for (int i = sid; i < eid; ++i)
	{
		int val = fscanf(fp, "%s %s %f\n", name0, name1, &score);
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

		string flow1 = outputDir_flow + name_A + "_" + name_B + ".txt";
		string flow2 = outputDir_flow + name_B + "_" + name_A + ".txt";

		dp.SetA(A);
		dp.SetBPrime(BP);
		dp.SetOutputDir(outputDir_flow);
		dp.SetRatio(1.0);
		dp.SetBlendWeight(2);
		dp.UsePhotoTransfer(false);
		dp.LoadInputs();
		dp.ComputeAnn(classifier_A, classifier_B);
	}

	google::ShutdownGoogleLogging();

	classifier_A.DeleteNet();
	classifier_B.DeleteNet();
	fclose(fp);
}

int main(int argc, char** argv) 
{
	run_flow(argc, argv);
	return 0;
}
