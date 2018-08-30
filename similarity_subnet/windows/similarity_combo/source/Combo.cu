// Copyright (c) Microsoft. All rights reserved.

// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include "GeneralizedPatchMatch.cuh"
#include "Combo.cuh"

#define ENABLE_VIS 0

struct Parameters
{
	std::vector<std::string> layers; //which layers used as content
	std::vector<std::string> layernames; //which layers used as content
	std::vector<std::string> datanames; //which layers used as content

	int patch_size0;
	int iter;

};

Combo::Combo()
{
}

Combo::~Combo()
{
}

void Combo::SetGPU(int no)
{
	int devCount;
	cudaGetDeviceCount(&devCount);
	wcout << "CUDA Devices: " << endl << endl;

	for (int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		size_t totalMem = 0;
		size_t freeMem = 0;
		cudaSetDevice(i);
		cudaMemGetInfo(&freeMem, &totalMem);
		wcout << "GPU " << i << ", Name = " << props.name << ", free  = " << freeMem << ", total = " << totalMem << endl;
	}

	cudaSetDevice(no);
	int num = -1;
	size_t totalMem = 0;
	size_t freeMem = 0;
	cudaGetDevice(&num);
	cudaMemGetInfo(&freeMem, &totalMem);
	wcout << "Current GPU = " << num << ", free  = " << freeMem << ", total = " << totalMem << endl;
}

bool Combo::LoadA(const char* file_A)
{
	img_AL_col = imread(file_A);
	if (img_AL_col.empty())
	{
		cout << "Error: Source image cannot read!" << endl;
		waitKey();
		return false;
	}

	img_AL = Mat::zeros(img_AL_col.size(), CV_8UC3);
	
	// convert to grayscale image
	{
		Mat gray(img_AL_col.size(), CV_8UC3);
		cvtColor(img_AL_col, gray, cv::COLOR_BGR2Lab);

#pragma omp parallel for
		for (int r = 0; r < img_AL.rows; ++r)
		{
			for (int c = 0; c < img_AL.cols; ++c)
			{
				uchar g = gray.at<Vec3b>(r, c)[0];
				img_AL.at<Vec3b>(r, c) = Vec3b(g, g, g);
			}
		}
	}

	return true;
}

bool Combo::LoadBP(const char* file_BP)
{
	img_BPL_col = imread(file_BP);
	if (img_BPL_col.empty())
	{
		cout << "Error: Reference image cannot read!" << endl;
		waitKey();
		return false;
	}

	img_BPL = Mat::zeros(img_BPL_col.size(), CV_8UC3);
	
	// convert to grayscale image
	{
		Mat gray(img_BPL_col.size(), CV_8UC3);
		cvtColor(img_BPL_col, gray, cv::COLOR_BGR2Lab);

#pragma omp parallel for
		for (int r = 0; r < img_BPL.rows; ++r)
		{
			for (int c = 0; c < img_BPL.cols; ++c)
			{
				uchar g = gray.at<Vec3b>(r, c)[0];
				img_BPL.at<Vec3b>(r, c) = Vec3b(g, g, g);
			}
		}
	}

	return true;
}

void Combo::GetASize(int& width, int& height)
{
	width = img_AL.cols;
	height = img_AL.rows;
}

void Combo::GetBPSize(int& width, int& height)
{
	width = img_BPL.cols;
	height = img_BPL.rows;
}

void Combo::ComputeDist(Classifier& classifier_A, Classifier& classifier_B, 
FILE* fp_a, FILE* fp_b, const char* ff_a, const char* ff_b) 
{
	if (img_BPL.empty())
	{
		printf("Error: Image2 is empty!\n");
		return;
	}
	if(img_AL.empty())
	{
		printf("Error: Image1 is empty!\n");
		return;
	}

	const int param_size = 9;
	int aw = img_AL.cols;
	int ah = img_AL.rows;
	int bw = img_BPL.cols;
	int bh = img_BPL.rows;

	int *params_host, *params_device_AB, *params_device_BA;
	unsigned int *ann_device_AB, *ann_host_AB, *ann_device_BA, *ann_host_BA;
	unsigned int *rann_device_AB, *rann_device_BA;
	float *annd_device_AB, *annd_host_AB, *annd_device_BA, *annd_host_BA;
	float *rannd_device_AB, *rannd_device_BA;
	unsigned char* bgr_device_AB, *bgr_device_BA, *bgr_host_AB, *bgr_host_BA;

	//set parameters
	Parameters params;
	params.layers.push_back("conv5_1/bn");
	params.layers.push_back("conv4_1/bn");
	params.layers.push_back("conv3_1/bn");
	params.layers.push_back("conv2_1/bn");
	params.layers.push_back("conv1_1/bn");

	std::vector<int> sizes;
	sizes.push_back(3);
	sizes.push_back(3);
	sizes.push_back(3);
	sizes.push_back(3);
	sizes.push_back(3);

	//scale and enhance
	Mat img_BP = img_BPL.clone();
	Mat img_A = img_AL.clone();

	std::vector<float *> data_A;
	data_A.resize(params.layers.size());
	std::vector<Dim> data_A_size;
	data_A_size.resize(params.layers.size());

	classifier_A.Predict(img_A, params.layers, data_A, data_A_size);

	std::vector<float *> data_B;
	data_B.resize(params.layers.size());
	std::vector<Dim> data_B_size;
	data_B_size.resize(params.layers.size());

	classifier_B.Predict(img_BP, params.layers, data_B, data_B_size);

	int full_ann_size_AB = aw * ah;
	int full_ann_size_BA = bw * bh;
	params_host = (int *)malloc(param_size * sizeof(int));

	ann_host_AB = (unsigned int *)malloc(full_ann_size_AB * sizeof(unsigned int));
	annd_host_AB = (float *)malloc(full_ann_size_AB * sizeof(float));
	bgr_host_AB = (unsigned char*)malloc(full_ann_size_AB * sizeof(unsigned char)* 3);

	ann_host_BA = (unsigned int *)malloc(full_ann_size_BA * sizeof(unsigned int));
	annd_host_BA = (float *)malloc(full_ann_size_BA * sizeof(float));
	bgr_host_BA = (unsigned char*)malloc(full_ann_size_BA * sizeof(unsigned int)* 3);

	cudaMalloc(&params_device_AB, param_size * sizeof(int));
	cudaMalloc(&ann_device_AB, full_ann_size_AB * sizeof(unsigned int));
	cudaMalloc(&rann_device_AB, full_ann_size_AB * sizeof(unsigned int));
	cudaMalloc(&annd_device_AB, full_ann_size_AB * sizeof(float));
	cudaMalloc(&rannd_device_AB, full_ann_size_AB * sizeof(float));
	cudaMalloc(&bgr_device_AB, full_ann_size_AB * sizeof(unsigned int));

	cudaMalloc(&params_device_BA, param_size * sizeof(int));
	cudaMalloc(&ann_device_BA, full_ann_size_BA * sizeof(unsigned int));
	cudaMalloc(&rann_device_BA, full_ann_size_BA * sizeof(unsigned int));
	cudaMalloc(&annd_device_BA, full_ann_size_BA * sizeof(float));
	cudaMalloc(&rannd_device_BA, full_ann_size_BA * sizeof(float));
	cudaMalloc(&bgr_device_BA, full_ann_size_BA * sizeof(unsigned char));

	int numlayer = params.layers.size();

	ifstream aflow_input;
	aflow_input.open(ff_a);
	for (int y = 0; y < ah; y++)
	{
		for (int x = 0; x < aw; x++)
		{
			int dx = 0, dy = 0;
			aflow_input >> dx;
			aflow_input >> dy;
			int xbest = x + dx;
			int ybest = y + dy;
			ann_host_AB[y * aw + x] = XY_TO_INT(xbest, ybest);
		}
	}
	aflow_input.close();

	ifstream bflow_input;
	bflow_input.open(ff_b);
	for (int y = 0; y < bh; y++)
	{
		for (int x = 0; x < bw; x++)
		{
			int dx = 0, dy = 0;
			bflow_input >> dx;
			bflow_input >> dy;
			int xbest = x + dx;
			int ybest = y + dy;
			ann_host_BA[y * bw + x] = XY_TO_INT(xbest, ybest);
		}
	}
	bflow_input.close();

	cudaMemcpy(ann_device_AB, ann_host_AB, full_ann_size_AB * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(ann_device_BA, ann_host_BA, full_ann_size_BA * sizeof(unsigned int), cudaMemcpyHostToDevice);

	dim3 blocksPerGridAB(aw / 20 + 1, ah / 20 + 1, 1);
	dim3 blocksPerGridBA(bw / 20 + 1, bh / 20 + 1, 1);
	dim3 threadsPerBlock(20, 20, 1);

	reverse_flow << <blocksPerGridAB, threadsPerBlock >> >(ann_device_AB, ann_device_BA, rann_device_AB, ah, aw, bh, bw);
	reverse_flow << <blocksPerGridBA, threadsPerBlock >> >(ann_device_BA, ann_device_AB, rann_device_BA, bh, bw, ah, aw);

	Mat result_AB = reconstruct_avg(img_AL_col, img_BPL_col, ann_host_AB, sizes[numlayer - 1]);
	cudaMemcpy(ann_host_AB, rann_device_AB, full_ann_size_AB * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	Mat reverse_AB = reconstruct_avg(img_AL_col, img_AL_col, ann_host_AB, sizes[numlayer - 1]);

	fwrite(&ah, sizeof(int), 1, fp_a);
	fwrite(&aw, sizeof(int), 1, fp_a);
	fwrite(&bh, sizeof(int), 1, fp_b);
	fwrite(&bw, sizeof(int), 1, fp_b);

	cv::vector<uchar> buf;
	imencode(".png", result_AB, buf);
	int sz = buf.size();
	fwrite(&sz, sizeof(int), 1, fp_a);
	fwrite(&(buf[0]), sizeof(uchar), sz, fp_a);

	imencode(".png", reverse_AB, buf);
	sz = buf.size();
	fwrite(&sz, sizeof(int), 1, fp_a);
	fwrite(&(buf[0]), sizeof(uchar), buf.size(), fp_a);

	Mat result_BA = reconstruct_avg(img_BPL_col, img_AL_col, ann_host_BA, sizes[numlayer - 1]);
	cudaMemcpy(ann_host_BA, rann_device_BA, full_ann_size_BA * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	Mat reverse_BA = reconstruct_avg(img_BPL_col, img_BPL_col, ann_host_BA, sizes[numlayer - 1]);

	imencode(".png", result_BA, buf);
	sz = buf.size();
	fwrite(&sz, sizeof(int), 1, fp_b);
	fwrite(&(buf[0]), sizeof(uchar), sz, fp_b);

	imencode(".png", reverse_BA, buf);
	sz = buf.size();
	fwrite(&sz, sizeof(int), 1, fp_b);
	fwrite(&(buf[0]), sizeof(uchar), buf.size(), fp_b);

	// compute feature distance for each layer
	for (int curr_layer = 0; curr_layer < numlayer; curr_layer++)//from 32 to 512
	{
		//set parameters	
		params_host[0] = data_A_size[curr_layer].channel;//channels
		params_host[1] = data_A_size[curr_layer].height;
		params_host[2] = data_A_size[curr_layer].width;
		params_host[3] = data_A_size[curr_layer].height;
		params_host[4] = data_A_size[curr_layer].width;
		params_host[5] = sizes[curr_layer];
		params_host[6] = params.iter;
		
		//copy to device
		cudaMemcpy(params_device_AB, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

		//set parameters			
		params_host[0] = data_B_size[curr_layer].channel;//channels
		params_host[1] = data_B_size[curr_layer].height;
		params_host[2] = data_B_size[curr_layer].width;
		params_host[3] = data_B_size[curr_layer].height;
		params_host[4] = data_B_size[curr_layer].width;

		//copy to device
		cudaMemcpy(params_device_BA, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

		int scale = pow(2, 4 - curr_layer);

		// error ba
		//set parameters	
		params_host[0] = data_A_size[curr_layer].channel;//channels
		params_host[1] = data_A_size[curr_layer].height;
		params_host[2] = data_A_size[curr_layer].width;
		params_host[3] = data_B_size[curr_layer].height;
		params_host[4] = data_B_size[curr_layer].width;
		params_host[5] = sizes[curr_layer];
		params_host[6] = params.iter;
		
		//copy to device
		cudaMemcpy(params_device_AB, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

		//set parameters			
		params_host[0] = data_B_size[curr_layer].channel;//channels
		params_host[1] = data_B_size[curr_layer].height;
		params_host[2] = data_B_size[curr_layer].width;
		params_host[3] = data_A_size[curr_layer].height;
		params_host[4] = data_A_size[curr_layer].width;

		//copy to device
		cudaMemcpy(params_device_BA, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

		compute_dist_norm << <blocksPerGridAB, threadsPerBlock >> >(ann_device_AB, annd_device_AB, data_A[curr_layer], data_B[curr_layer], params_device_AB, aw, ah, scale);
		compute_dist_norm << <blocksPerGridBA, threadsPerBlock >> >(ann_device_BA, annd_device_BA, data_B[curr_layer], data_A[curr_layer], params_device_BA, bw, bh, scale);

		convert_float2bgr << <blocksPerGridAB, threadsPerBlock >> >(annd_device_AB, bgr_device_AB, aw, ah);
		convert_float2bgr << <blocksPerGridBA, threadsPerBlock >> >(annd_device_BA, bgr_device_BA, bw, bh);

		cudaMemcpy(bgr_host_AB, bgr_device_AB, full_ann_size_AB * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaMemcpy(bgr_host_BA, bgr_device_BA, full_ann_size_BA * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		Mat ebgrAB(ah, aw, CV_8UC1, bgr_host_AB);
		Mat ebgrBA(bh, bw, CV_8UC1, bgr_host_BA);

		imencode(".png", ebgrAB, buf);
		int sz = buf.size();
		fwrite(&sz, sizeof(int), 1, fp_a);
		fwrite(&(buf[0]), sizeof(uchar), sz, fp_a);

		imencode(".png", ebgrBA, buf);
		sz = buf.size();
		fwrite(&sz, sizeof(int), 1, fp_b);
		fwrite(&(buf[0]), sizeof(uchar), buf.size(), fp_b);

		// error ab
		reverse_dist << <blocksPerGridAB, threadsPerBlock >> >(ann_device_AB, rannd_device_AB, annd_device_BA, aw, ah, bw, bh);
		reverse_dist << <blocksPerGridBA, threadsPerBlock >> >(ann_device_BA, rannd_device_BA, annd_device_AB, bw, bh, aw, ah);

		convert_float2bgr << <blocksPerGridAB, threadsPerBlock >> >(rannd_device_AB, bgr_device_AB, aw, ah);
		convert_float2bgr << <blocksPerGridBA, threadsPerBlock >> >(rannd_device_BA, bgr_device_BA, bw, bh);

		cudaMemcpy(bgr_host_AB, bgr_device_AB, full_ann_size_AB * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaMemcpy(bgr_host_BA, bgr_device_BA, full_ann_size_BA * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		Mat rbgrAB(ah, aw, CV_8UC1, bgr_host_AB);
		Mat rbgrBA(bh, bw, CV_8UC1, bgr_host_BA);

		imencode(".png", rbgrAB, buf);
		sz = buf.size();
		fwrite(&sz, sizeof(int), 1, fp_a);
		fwrite(&(buf[0]), sizeof(uchar), buf.size(), fp_a);

		imencode(".png", rbgrBA, buf);
		sz = buf.size();
		fwrite(&sz, sizeof(int), 1, fp_b);
		fwrite(&(buf[0]), sizeof(uchar), buf.size(), fp_b);
	}

	cudaFree(params_device_AB);
	cudaFree(ann_device_AB);
	cudaFree(rann_device_AB);
	cudaFree(annd_device_AB);
	cudaFree(rannd_device_AB);

	cudaFree(params_device_BA);
	cudaFree(ann_device_BA);
	cudaFree(rann_device_BA);
	cudaFree(annd_device_BA);
	cudaFree(rannd_device_BA);

	cudaFree(bgr_device_AB);
	cudaFree(bgr_device_BA);

	free(ann_host_AB);
	free(ann_host_BA);
	free(annd_host_AB);
	free(annd_host_BA);
	free(params_host);
	free(bgr_host_AB);
	free(bgr_host_BA);

	for (int i = 0; i < numlayer; i++)
	{
		cudaFree(data_A[i]);
		cudaFree(data_B[i]);
	}
}
