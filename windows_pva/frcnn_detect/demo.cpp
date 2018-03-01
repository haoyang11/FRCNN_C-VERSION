#include <caffe\api\FRCNN\frcnn_api.hpp>  //Detect head file  
#include "caffe/util/benchmark.hpp"
#include "caffe/FRCNN/util/frcnn_vis.hpp"
#include "Register.h"
using namespace std;
using namespace cv;
using namespace caffe::Frcnn;

int main() {
	//std::string proto_file = "test.prototxt";
	//std::string model_file = "VGG16_faster_rcnn_final.caffemodel";
	//std::string default_config_file = "voc_config.json";
	//std::string model_file = "pvanet_frcnn_iter_100000.caffemodel";
	std::string image_dir = "./test/";
	std::vector<std::string> images = caffe::Frcnn::get_file_list(image_dir, ".jpg");
	//FRCNN_API::Detector detect("test_vgg16.prototxt", "VGG16_faster_rcnn_final.caffemodel", "voc_config.json", true, false);
	FRCNN_API::Detector detect("test_inference.prototxt", "pvanet_traffic_noconv4_5_34_lowre2_iter_300000_inference.caffemodel", "voc_config_pva.json", true, false);
	std::vector<caffe::Frcnn::BBox<float> > results;
	caffe::Timer time_;
	DLOG(INFO) << "Test Image Dir : " << image_dir << "  , have " << images.size() << " pictures!";
	for (size_t index = 0; index < images.size(); ++index) {
		DLOG(INFO) << std::endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl
			<< "Demo for " << images[index];
		cv::Mat image = cv::imread(image_dir + images[index]);
		time_.Start();
		detect.predict(image, results);
		LOG(INFO) << "Predict " << images[index] << " cost " << time_.MilliSeconds() << " ms.";
		LOG(INFO) << "There are " << results.size() << " objects in picture.";
		for (size_t obj = 0; obj < results.size(); obj++) {
			LOG(INFO) << results[obj].to_string();
			rectangle(image, cv::Point(results[obj][0], results[obj][1]), cv::Point(results[obj][2], results[obj][3]), Scalar(0, 0, 255));
		}
		cv::imshow(images[index], image);
		/*
		for (int label = 0; label < caffe::Frcnn::FrcnnParam::n_classes; label++) {
			std::vector<caffe::Frcnn::BBox<float> > cur_res;
			for (size_t idx = 0; idx < results.size(); idx++) {
				if (results[idx].id == label) {
					cur_res.push_back(results[idx]);
				}
			}
			if (cur_res.size() == 0) continue;
			//cv::Mat ori;
			//image.convertTo(ori, CV_32FC3);
			//caffe::Frcnn::vis_detections(ori, cur_res, caffe::Frcnn::LoadVocClass());
			//cv::Mat out;
			//ori.convertTo(out, CV_8UC3);
			//rectangle(images, cv::Point(boxes[i][0], boxes[i][1]), cv::Point(boxes[i][2], boxes[i][3]), Scalar(0, 0, 255));
			cv::imshow(images[index], image);
		}*/
	}
	waitKey(0);
	return 0;



	
	//Mat frame = imread("D:\\Machinelearning\\caffe-faster-rcnn\\caffe-frcnn-base\\examples\\FRCNN\\images\\004545.jpg"); 
	//	/* Initiaze the detector, the four parameters were:
	//	1. network file
	//	2. trained model file
	//	3. config file
	//	4. whether to open the GPU mode, default true
	//	5. whether to ignore print log, default true
	//	*/
	//FRCNN_API::Detector detect("test.prototxt", "VGG16_faster_rcnn_final.caffemodel", "voc_config.json", true, true);
	//vector<BBox<float> > boxes = detect.predict(frame);    // forward, detect results saved here 
	//for (int i = 0; i < boxes.size(); i++)   //draw rects  
	//	rectangle(frame, cv::Point(boxes[i][0], boxes[i][1]), cv::Point(boxes[i][2], boxes[i][3]), Scalar(0, 0, 255));
	//imshow("demo", frame);
	//waitKey(0);
	//return 0;
	
}