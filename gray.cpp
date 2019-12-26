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

float ColourMatExpectation(const Mat img)//мат ожидание= сумма значений/число значений
{
	float M = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b bgr = img.at<Vec3b>(i, j);
			M += (bgr[0] + bgr[1] + bgr[2]) / 3;
		}
	M = M / (img.rows * img.cols);
	return M;
}

float GrayMatExpectation(const Mat img)//мат ожидание= сумма значений/число значений
{
	float M = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			M += img.at<uchar>(i, j);
			//Vec
			//Vec3b bgr = img.at<Vec3b>(i, j);
			//M += (bgr[0] + bgr[1] + bgr[2]) / 3;
		}
	M = M / (img.rows * img.cols);
	return M;
}


float ColourDispersia(const Mat img)//(мат ожидание от (значение-мат ож)^2)/число значений
{
	float D = 0;
	float M = ColourMatExpectation(img);
	for (int i = 0; i < img.rows; i++)
	{ 
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b bgr = img.at<Vec3b>(i, j);
			D += pow((bgr[0] + bgr[1] + bgr[2]) / 3 - M, 2);
		}
	}
	D = D / (img.rows * img.cols);
	return D;
}

float GrayDispersia(const Mat img)//(мат ожидание от (значение-мат ож)^2)/число значений
{
	float D = 0;
	float M = GrayMatExpectation(img);
	for (int i = 0; i < img.rows; i++)
	{ 
		for (int j = 0; j < img.cols; j++)
		{
			D += pow(img.at<uchar>(i, j) - M, 2);
		}
	}
	D = D / (img.rows * img.cols);
	return D;
}

//средне квадрат отклонение = корень из дисперсиии
float MSE(const Mat img)//средне квадрат ошибка =  дисперсиия
{
	return GrayDispersia(img);
}

float MSE2(const Mat img1, const Mat img2)
{
	float D = 0;
	for (int i = 0; i < img1.rows; i++)
	{
		for (int j = 0; j < img1.cols; j++)
		{
			D += pow(img1.at<uchar>(i, j) - img2.at<uchar>(i, j), 2);
		}
	}
	D = D / (img1.rows * img1.cols);
	return D;
}

float PSNR(const Mat img1, const Mat img2)//10*lg(мах знач пикселя/MSE)   пиковое отношение сигнал/шум
{
	float maxpixel;
	for (int i = 0; i < img1.rows; i++)
	{
		for (int j = 0; j < img1.cols; j++)
		{
			maxpixel= max(img1.at<uchar>(i, j) , img2.at<uchar>(i, j));
		}
	}
	return (10 * log(maxpixel*maxpixel / MSE2(img1, img2)));
}

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

float GetCov(Mat img1, Mat img2)//ковариация
{
	if (img1.size != img2.size)
	{
		cout << "Different sizes" << endl;
		throw - 111.11;
	}
	float Cov = 0;
	for (int i = 0; i < img1.rows; i++)
		for (int j = 0; j < img1.cols; j++)
		{
			Vec3b bgr1 = img1.at<Vec3b>(i, j);
			Vec3b bgr2 = img1.at<Vec3b>(i, j);
			Cov += (bgr1[0] + bgr1[1] + bgr1[2])*(bgr2[0] + bgr2[1] + bgr2[2]) / 9;
		}
	Cov = Cov / (img1.rows * img1.cols) - GetIntensity(img1)*GetIntensity(img2);
	return Cov;
}

float GetContrast(Mat img)//констрастность SIMM
{
	float Contrast = 0;
	float Intens = GetIntensity(img);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b bgr = img.at<Vec3b>(i, j);
			Contrast += pow((bgr[0] + bgr[1] + bgr[2]) / 3 - Intens, 2);
		}
	Contrast = sqrt(Contrast / (img.rows * img.cols));
	return Contrast;
}

float SIMMMetric(Mat img1, Mat img2)
{
	int c1 = 0;
	int c2 = 0;
	return (2.f * GetIntensity(img1) * GetIntensity(img2) + c2)*(2 * GetCov(img1, img2) + c1) /
		((GetIntensity(img1) * GetIntensity(img1) + GetIntensity(img2) * GetIntensity(img2) + c2)
			* (GetContrast(img1) * GetContrast(img1) + GetContrast(img2) * GetContrast(img2) + c1));
}

float SSIM(const Mat img1, const Mat img2) {
    float l = 0, c = 0;
    float Intensity1 = GetIntensity(img1);
    float Intensity2 = GetIntensity(img2);
    float Contrast1 = GetContrast(img1);
    float Contrast2 = GetContrast(img2);
    int c1 = 0, c2 = 0;
    l = (2 * Intensity1 * Intensity2 + c1) / (Intensity1 * Intensity1 + Intensity2 * Intensity2 + c1);
    c = (2 * Contrast1 * Contrast2 + c2) / (Contrast1 * Contrast1 + Contrast2 * Contrast2 + c2);
    return l + c;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	Mat image;
    image = imread("DarkHouse.jpg", CV_LOAD_IMAGE_COLOR);
    //image = imread("smallsmh.jpg", CV_LOAD_IMAGE_COLOR);
	//image = imread("vg.jpg", CV_LOAD_IMAGE_COLOR);
	if (!image.data)
	{
		cout << "Ошибка открытия изображения" << endl;
		return -1;
	}
	namedWindow("My image");
	imshow("My image", image);

    //////////////////////////GRAY///////////////////////////////////////////
    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);
    imshow("GRAY", gray);

    Mat image1, image2, image3, image4, image5, image6, image7, image8;
    image1 = image.clone();
    image2 = image.clone();
    image3 = image.clone();
    image4 = image.clone();
    image5 = image.clone();
    image6 = image.clone();
    image7 = image.clone();
    image8 = image.clone();

	//average(R + G + B) / 3
	for (int i = 0; i < image.rows; i++)
	{ 
		for (int j = 0; j < image.cols; j++)
		{
			Vec3b bgr = image.at<Vec3b>(i, j);
			float gray = (bgr[0] + bgr[1] + bgr[2]) / 3;
			image1.at<Vec3b>(i, j)[0] = gray;
			image1.at<Vec3b>(i, j)[1] = gray;
			image1.at<Vec3b>(i, j)[2] = gray;
		}
	}


	imshow("Average", image1);
	cout << "SIMM for GRAY and average = " << SIMM(gray, image1) <<endl;
	cout << "MSE GRAY " << MSE(gray) << " - MSE Average " << MSE(image1) << " = " << MSE(gray) - MSE(image1) << endl;
	
	//lightness (max(R, G, B) + min(R, G, B)) / 2
	for (int i = 0; i < image.rows; i++) 
	{
		for (int j = 0; j < image.cols; j++)
		{
			Vec3b bgr = image.at<Vec3b>(i, j);
			float gray = (max(bgr[0],max(bgr[1], bgr[2])) + min(bgr[0], min(bgr[1], bgr[2]))) / 2;
			image2.at<Vec3b>(i, j)[0] = gray;
			image2.at<Vec3b>(i, j)[1] = gray;
			image2.at<Vec3b>(i, j)[2] = gray;
		}
	}
	imshow("LightnessGray", image2);
	cout << "MSE GRAY " << MSE(gray) << " - MSE LightnessGray " << MSE(image2) << " = " << MSE(gray) - MSE(image2) << endl;

	//luminosity (0.21 R + 0.72 G + 0.07 B)
	for (int i = 0; i < image.rows; i++) 
	{
		for (int j = 0; j < image.cols; j++)
		{
			Vec3b bgr = image.at<Vec3b>(i, j);
			float gray = bgr[0] * 0.07f + bgr[1] * 0.72f + bgr[2] * 0.21f;
			image3.at<Vec3b>(i, j)[0] = gray;
			image3.at<Vec3b>(i, j)[1] = gray;
			image3.at<Vec3b>(i, j)[2] = gray;
		}
	}
	imshow("Luminosity", image3);
	cout << "MSE GRAY " << MSE(gray) << " - MSE Luminosity " << MSE(image3) << " = " << MSE(gray) - MSE(image3) << endl;

	// Photoshop, GIMP (0.3 Red+ 0.59 Green + 0.11 Blue)
	for (int i = 0; i < image.rows; i++) 
	{
		for (int j = 0; j < image.cols; j++)
		{
			Vec3b bgr = image.at<Vec3b>(i, j);
			float gray = bgr[0] * 0.11f + bgr[1] * 0.59f + bgr[2] * 0.3f;
			image4.at<Vec3b>(i, j)[0] = gray;
			image4.at<Vec3b>(i, j)[1] = gray;
			image4.at<Vec3b>(i, j)[2] = gray;
		}
	}
	imshow("Photoshop, GIMP", image4);
	//cout << "MSE GRAY " << MSE(gray) << " - MSE Photoshop, GIMP " << MSE(image4) << " = " << MSE(gray) - MSE(image4) << endl;

	// ITU-R, BT.709 (0.2126 Red + 0.7152 Green + 0.0722 Blue)
	//Для учёта особенностей восприятия изображения (чувствительность к зелёному и синему цвету) в модели HDTV 
	for (int i = 0; i < image.rows; i++) 
	{
		for (int j = 0; j < image.cols; j++)
		{
			Vec3b bgr = image.at<Vec3b>(i, j);
			float gray = bgr[0] * 0.0722f + bgr[1] * 0.7152f + bgr[2] * 0.2126f;
			image5.at<Vec3b>(i, j)[0] = gray;
			image5.at<Vec3b>(i, j)[1] = gray;
			image5.at<Vec3b>(i, j)[2] = gray;
		}
	}
	imshow("ITU-R, BT.709", image5);
	cout << "MSE GRAY " << MSE(gray) << " - ITU-R, BT.709 " << MSE(image5) << " = " << MSE(gray) - MSE(image5) << endl;

	//Max(Red, Green, Blue)
	for (int i = 0; i < image.rows; i++) 
	{
		for (int j = 0; j < image.cols; j++)
		{
			Vec3b bgr = image.at<Vec3b>(i, j);
			float gray = (max(bgr[0], max(bgr[1], bgr[2])));
			image6.at<Vec3b>(i, j)[0] = gray;
			image6.at<Vec3b>(i, j)[1] = gray;
			image6.at<Vec3b>(i, j)[2] = gray;
		}
	}
	imshow("Max(Red, Green, Blue)", image6);
	cout << "MSE GRAY " << MSE(gray) << " - Max(Red, Green, Blue) " << MSE(image6) << " = " << MSE(gray) - MSE(image6) << endl;

	//Min(Red, Green, Blue)
	for (int i = 0; i < image.rows; i++) 
	{
		for (int j = 0; j < image.cols; j++)
		{
			Vec3b bgr = image.at<Vec3b>(i, j);
			float gray = (min(bgr[0], min(bgr[1], bgr[2])));
			image7.at<Vec3b>(i, j)[0] = gray;
			image7.at<Vec3b>(i, j)[1] = gray;
			image7.at<Vec3b>(i, j)[2] = gray;
		}
	}
	imshow("Min(Red, Green, Blue)", image7);
	cout << "MSE GRAY " << MSE(gray) << " - Min(Red, Green, Blue) " << MSE(image7) << " = " << MSE(gray) - MSE(image7) << endl;

	//0.2952 R + 0.5547 G + 0.148 B
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			Vec3b bgr = image.at<Vec3b>(i, j);
			float gray = bgr[0] * 0.148f + bgr[1] * 0.5547f + bgr[2] * 0.2952f;
			image8.at<Vec3b>(i, j)[0] = gray;
			image8.at<Vec3b>(i, j)[1] = gray;
			image8.at<Vec3b>(i, j)[2] = gray;
		}
	}
	imshow("0.2952 R + 0.5547 G + 0.148 B", image8);
	cout << "MSE GRAY " << MSE(gray) << " - 0.2952 R + 0.5547 G + 0.148 B " << MSE(image8) << " = " << MSE(gray) - MSE(image8) << endl;

    //////////////////////////GRAY///////////////////////////////////////////

	image.release();
	image1.release();
	image2.release();
	image3.release();
	image4.release();
	image5.release();
	image6.release();
	image7.release();
	
	waitKey(0);
	return 0;
}