#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

using namespace cv;
using namespace std;

float GetIntensity(Mat img)//интенсивность SIMM
{
	float Intens = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b bgr = img.at<Vec3b>(i, j);
			Intens += (bgr[0] + bgr[1] + bgr[2]) / 3;
		}
	Intens = Intens / (img.rows * img.cols);
	return Intens;
}

float GetContrast(Mat img)//констрастность SIMM
{
	float Contrast = 0;
	float Intensity = GetIntensity(img);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b bgr = img.at<Vec3b>(i, j);
			Contrast += pow((bgr[0] + bgr[1] + bgr[2]) / 3 - Intensity, 2);
		}
	Contrast = sqrt(Contrast / (img.rows * img.cols) - 1);
	return Contrast;
}

float SSIM(const Mat img1, const Mat img2) {
	float l = 0, c = 0, s = 0;
	float Intensity1 = GetIntensity(img1);
	float Intensity2 = GetIntensity(img2);
	float Contrast1 = GetContrast(img1);
	float Contrast2 = GetContrast(img2);
	int c1 = 5, c2 = 2, c3 = 1;
	l = (2 * Intensity1 * Intensity2 + c1) / (Intensity1 * Intensity1 + Intensity2 * Intensity2 + c1);
	c = (2 * Contrast1 * Contrast2 + c2) / (Contrast1 * Contrast1 + Contrast2 * Contrast2 + c2);
	//s = (GetCov(img1, img2) + c3) / (Contrast1 + Contrast2 + c3);
	return l + c - 1;
}

Mat MidpointFilter(const Mat img) {
	Mat filter = img.clone();
	int i = 0, j = 0;
	int size = 3;
	float mid = 0, max, min;
	while (i < img.rows) {
		while (j < img.cols) {
			max = 0, min = 255;
			for (int i1 = i; i1 < std::min(i + size, filter.rows); i1++) {
				for (int j1 = j; j1 < std::min(j + size, filter.cols); j1++) {
					Vec3b bgr = img.at<Vec3b>(i1, j1);
					if (bgr[0] > max) {
						max = bgr[0];
					}
					if (bgr[0] < min) {
						min = bgr[0];
					}
				}
			}
			mid = (max + min) / 2;
			for (int i1 = i; i1 < std::min(i + size, filter.rows); i1++) {
				for (int j1 = j; j1 < std::min(j + size, filter.cols); j1++) {
					filter.at<Vec3b>(i1, j1)[0] = mid;
					filter.at<Vec3b>(i1, j1)[1] = mid;
					filter.at<Vec3b>(i1, j1)[2] = mid;
				}
			}
			j += 3;
		}
		j = 0;
		i += 3;
	}
	return filter;
}

Mat Salt_pepper(const Mat img) {
	Mat noise = img.clone();
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int rnd = rand() % 1000;
			if (rnd > 990) {
				if (rnd > 995) {
					noise.at<Vec3b>(i, j)[0] = 255;
					noise.at<Vec3b>(i, j)[1] = 255;
					noise.at<Vec3b>(i, j)[2] = 255;
				}
				else {
					noise.at<Vec3b>(i, j)[0] = 0;
					noise.at<Vec3b>(i, j)[1] = 0;
					noise.at<Vec3b>(i, j)[2] = 0;
				}
			}
		}
	}
	return noise;
}

Mat RandNoise(Mat img)
{
	Mat res;
	img.copyTo(res);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			if (!(rand() % 256))
			{
				res.at<Vec3b>(i, j)[0] = 255;
				res.at<Vec3b>(i, j)[1] = 255;
				res.at<Vec3b>(i, j)[2] = 255;
			}
		}
	return res;
}

Mat ArifmeticMean(Mat img)
{
	Mat res;
	img.copyTo(res);
	int Radx = 4;
	int Rady = 4;
	for (int x = 0; x < img.cols; x++)
		for (int y = 0; y < img.rows; y++)
		{
			double resR = 1.0;
			double resG = 1.0;
			double resB = 1.0;
			for (int i = -Rady; i < Rady; i++)
				for (int j = -Radx; j < Radx; j++)
				{
					int X = std::min(x + j, img.cols - 1);
					X = std::max(X, 0);
					int Y = std::min(y + i, img.rows - 1);
					Y = std::max(Y, 0);
					Vec3b bgr = img.at<Vec3b>(Y, X);
					resR += bgr[2];
					resG += bgr[1];
					resB += bgr[0];
				}
			res.at<Vec3b>(y, x)[0] = std::min((int)(resB / (64)), 255);
			res.at<Vec3b>(y, x)[1] = std::min((int)(resG / (64)), 255);
			res.at<Vec3b>(y, x)[2] = std::min((int)(resR / (64)), 255);
		}
	return res;
}

Mat GeometricMean(Mat img)
{
	Mat res;
	img.copyTo(res);
	int Radx = 2;
	int Rady = 2;
	for (int x = 0; x < img.cols; x++)
		for (int y = 0; y < img.rows; y++)
		{
			double resR = 1;
			double resG = 1;
			double resB = 1;
			for (int i = -Rady; i < Rady; i++)
				for (int j = -Radx; j < Radx; j++)
				{
					int X = std::min(x + j, img.cols - 1);
					X = std::max(X, 0);
					int Y = std::min(y + i, img.rows - 1);
					Y = std::max(Y, 0);
					Vec3b bgr = img.at<Vec3b>(Y, X);
					if (bgr[2])
						resR *= bgr[2];
					if (bgr[1])
						resG *= bgr[1];
					if (bgr[0])
						resB *= bgr[0];
				}
			int a = pow(resB, 1.0 / (9 * Radx * Rady));
			res.at<Vec3b>(y, x)[0] = std::min(pow(resB, 1.0 / (16)), 255.0);
			res.at<Vec3b>(y, x)[1] = std::min(pow(resG, 1.0 / (16)), 255.0);
			res.at<Vec3b>(y, x)[2] = std::min(pow(resR, 1.0 / (16)), 255.0);
		}
	return res;
}

Mat HarmonicMean(Mat img)
{
	Mat res;
	img.copyTo(res);
	int Radx = 2;
	int Rady = 2;
	for (int x = 0; x < img.cols; x++)
		for (int y = 0; y < img.rows; y++)
		{
			double resR = 0.0;
			double resG = 0.0;
			double resB = 0.0;
			for (int i = -Rady; i < Rady; i++)
				for (int j = -Radx; j < Radx; j++)
				{
					int X = std::min(x + j, img.cols - 1);
					X = std::max(X, 0);
					int Y = std::min(y + i, img.rows - 1);
					Y = std::max(Y, 0);
					Vec3b bgr = img.at<Vec3b>(Y, X);
					resR += 1.0 / bgr[2];
					resG += 1.0 / bgr[1];
					resB += 1.0 / bgr[0];
				}
			res.at<Vec3b>(y, x)[0] = std::min((int)(16.0 / resB), 255);
			res.at<Vec3b>(y, x)[1] = std::min((int)(16.0 / resG), 255);
			res.at<Vec3b>(y, x)[2] = std::min((int)(16.0 / resR), 255);
		}
	return res;
}


int main(int argc, char** argv)
{
	Mat image;
	image = imread("D:\\My OpenCV Website\\i.jpg");
	//image = imread("vg.jpg", CV_LOAD_IMAGE_COLOR);
	if (!image.data)
	{
		cout << "Ошибка открытия изображения" << endl;
		return -1;
	}
	namedWindow("My image");
	imshow("My image", image);




	Mat rand_noise = RandNoise(image);
	Mat sp_noise = Salt_pepper(image);

	Mat medianBlur_sp = image.clone();
	medianBlur(sp_noise, medianBlur_sp, 3);
	Mat Midpoint_sp = MidpointFilter(sp_noise);
	Mat Arifmetic_sp = ArifmeticMean(sp_noise);
	Mat Geometric_sp = GeometricMean(sp_noise);
	Mat Harmonic_sp = HarmonicMean(sp_noise);

	Mat medianBlur_rand = image.clone();
	medianBlur(rand_noise, medianBlur_rand, 3);
	Mat Midpoint_rand = MidpointFilter(rand_noise);
	Mat Arifmetic_rand = ArifmeticMean(rand_noise);
	Mat Geometric_rand = GeometricMean(rand_noise);
	Mat Harmonic_rand = HarmonicMean(rand_noise);


	imshow("sp_noise", sp_noise);
	imshow("MidpointFilter_sp", Midpoint_sp);
	imshow("medianBlur_sp", medianBlur_sp);
	imshow("Arifmetic_sp", Arifmetic_sp);
	imshow("Geometric_sp", Geometric_sp);
	imshow("Harmonic_sp", Harmonic_sp);

	imshow("rand_noise", rand_noise);
	imshow("MidpointFilter_rand", Midpoint_rand);
	imshow("medianBlur_rand", medianBlur_rand);
	imshow("Arifmetic_rand", Arifmetic_rand);
	imshow("Geometric_rand", Geometric_rand);
	imshow("Harmonic_rand", Harmonic_rand);

	std::cout << "SSIM Midpoint Salt_pepper    " << SSIM(image, Midpoint_sp) << std::endl;
	std::cout << "SSIM medianBlur Salt_pepper  " << SSIM(image, medianBlur_sp) << std::endl;
	std::cout << "SSIM arifmeic Salt_pepper    " << SSIM(Arifmetic_sp, image) << std::endl;
	std::cout << "SSIM geometric Salt_pepper   " << SSIM(Geometric_sp, image) << std::endl;
	std::cout << "SSIM harmonic Salt_pepper    " << SSIM(Harmonic_sp, image) << std::endl << std::endl;


	std::cout << "SSIM Midpoint rand    " << SSIM(image, Midpoint_rand) << std::endl;
	std::cout << "SSIM medianBlur rand  " << SSIM(image, medianBlur_rand) << std::endl;
	std::cout << "SSIM arifmeic rand    " << SSIM(Arifmetic_rand, image) << std::endl;
	std::cout << "SSIM geometric rand   " << SSIM(Geometric_rand, image) << std::endl;
	std::cout << "SSIM harmonic rand    " << SSIM(Harmonic_rand, image) << std::endl;


	image.release();
	sp_noise.release();
	Midpoint_sp.release();
	medianBlur_sp.release();
	Arifmetic_sp.release();
	Geometric_sp.release();
	Harmonic_sp.release();
	rand_noise.release();
	Midpoint_rand.release();
	medianBlur_rand.release();
	Arifmetic_rand.release();
	Geometric_rand.release();
	Harmonic_rand.release();

	waitKey(0);
	return 0;
}