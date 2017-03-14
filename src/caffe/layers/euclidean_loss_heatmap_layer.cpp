#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/layers/loss_layer.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/euclidean_loss_heatmap_layer.hpp"


// Euclidean loss layer that computes loss on a [x] x [y] x [ch] set of heatmaps,
// and enables visualisation of inputs, GT, prediction and loss.


namespace caffe {

template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    //@cf bottom[0] store label, bottom[1] store view1, bottom[2] store view2, bottom[3] store view3
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[1]->channels(), bottom[3]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[1]->height(), bottom[2]->height());
    CHECK_EQ(bottom[1]->height(), bottom[3]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    CHECK_EQ(bottom[1]->width(), bottom[2]->width());
    CHECK_EQ(bottom[1]->width(), bottom[3]->width());
    temp1.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
    temp2.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
    temp3.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
    diff_temp.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
    diff_temp1.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
    for_diff.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
    after_img2_all.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
    after_img3_all.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
    map2.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
    map3.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
}


template<typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (this->layer_param_.loss_weight_size() == 0) {
        this->layer_param_.add_loss_weight(Dtype(1));
    }

}

template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    Dtype loss = 0;

    int visualise_channel = this->layer_param_.visualise_channel();
    bool visualise = this->layer_param_.visualise();
    //@cf choose which view to be optimal
    int whichview = this->layer_param_.whichview();

    const Dtype* gt_pred = bottom[0]->cpu_data(); // GT predictions
    const Dtype* bottom_pred1 = bottom[1]->cpu_data();    // predictions for all view1 images
    const Dtype* bottom_pred2 = bottom[2]->cpu_data();    // predictions for all view2 images
    const Dtype* bottom_pred3 = bottom[3]->cpu_data();    // predictions for all view3 images
    const int num_images = bottom[0]->num();
    const int label_height = bottom[0]->height();
    const int label_width = bottom[0]->width();
    const int num_channels = bottom[1]->channels();
    const int count = bottom[0]->count();
    
    DLOG(INFO) << "bottom size: " << bottom[1]->height() << " " << bottom[1]->width() << " " << bottom[1]->channels();

    const int label_channel_size = label_height * label_width;
    const int label_img_size = label_channel_size * num_channels;
    cv::Mat bottom_img, gt_img, diff_img;  // Initialise opencv images for visualisation

    if (visualise)
    {
        cv::namedWindow("bottom", CV_WINDOW_AUTOSIZE);
        //cv::namedWindow("bottom2", CV_WINDOW_AUTOSIZE);
       // cv::namedWindow("bottom3", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("gt", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("diff", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("overlay", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("visualisation_bottom", CV_WINDOW_AUTOSIZE);
        bottom_img = cv::Mat::zeros(label_height, label_width, CV_32FC1);
       // bottom_img2 = cv::Mat::zeros(label_height, label_width, CV_32FC1);
        //bottom_img3 = cv::Mat::zeros(label_height, label_width, CV_32FC1);
        gt_img = cv::Mat::zeros(label_height, label_width, CV_32FC1);
        diff_img = cv::Mat::zeros(label_height, label_width, CV_32FC1);
    }
    
    //@cf compute fundumental matrix between 1 2, 1 3, 2 3
    cv::Mat T12 = (cv::Mat_<float>(3,3) << 0., 3.461, -0.058, -3.461, 0., 1.585, 0.058, -1.585, 0.);
    cv::Mat R12 = (cv::Mat_<float>(3,3) << -0.7482, -0.0200, -0.6632, 0.0110, 0.9990, -0.0425, 0.6634, -0.0391, -0.7472);
    cv::Mat T13 = (cv::Mat_<float>(3,3) << 0., 3.256, -0.178, -3.256, 0., -1.711, 0.178, 1.711, 0.);
    cv::Mat R13 = (cv::Mat_<float>(3,3) << -0.3929, 0.0410, 0.9187, -0.0335, 0.9977, -0.0588, -0.9190, -0.0538, -0.3906);
    cv::Mat R23 = R12*R13.inv();
    cv::Mat T_12 = (cv::Mat_<float>(3,1) << 1.585, 0.058, 3.461);
    cv::Mat T_13 = (cv::Mat_<float>(3,1) << -1.711, 0.178, 3.256);
    cv::Mat T_23 = R23*T_13 - T_12;
    cv::Mat T23 = (cv::Mat_<float>(3,3) << 0, -T_23.at<float>(2,0), T_23.at<float>(1,0), T_23.at<float>(2,0), 0, -T_23.at<float>(0,0), -T_23.at<float>(1,0), T_23.at<float>(0,0), 0);
    cv::Mat K1 = (cv::Mat_<float>(3,3) << 1.0717681657051608e+03, 0., 9.8423353195061145e+02, 0, -1.0755322668775907e+03, 5.6619143090534288e+02, 0., 0., 1.);
    cv::Mat K2 = (cv::Mat_<float>(3,3) << 1.0629624701217140e+03, 0., 9.7640281203786014e+02, 0.,-1.0572278734281156e+03, 5.3624684921864434e+02, 0., 0., 1.);
    cv::Mat K3 = (cv::Mat_<float>(3,3) << 1.0790502283080896e+03, 0., 9.5632812899289843e+02, 0.,-1.0727470561590965e+03, 5.8624531136777273e+02, 0., 0., 1.);
    cv::Mat scale = (cv::Mat_<float>(3,3) << -1, 0, 1500, 0, 1, 0, 0, 0, 16.875);
    cv::Mat k1 = scale*K1*64/1080;
    cv::Mat k2 = scale*K2*64/1080;
    cv::Mat k3 = scale*K3*64/1080;

    cv::Mat F12 = (k2.inv()).t()*T12*R12*k1.inv();
    cv::Mat F13 = (k3.inv()).t()*T13*R13*k1.inv();
    cv::Mat F23 = (k2.inv()).t()*T23*R23*k3.inv();
    
  
    //std::cout<<F23.at<float>(0,0)<<" "<<F23.at<float>(0,1)<<" "<<F23.at<float>(0,2)<<" "<<F23.at<float>(1,0)<<" "<<F23.at<float>(1,1)<<" "<<F23.at<float>(1,2)<<" "<<F23.at<float>(2,0)<<" "<<F23.at<float>(2,1)<<" "<<F23.at<float>(2,2)<<" "<<std::endl;
    
    
    for(int i = 0; i < count; i++)
    {
    	map2.mutable_cpu_data()[i] = -1;
    	map3.mutable_cpu_data()[i] = -1;
    }
    
    // Loop over images
    for (int idx_img = 0; idx_img < num_images; idx_img++)
    {
        // Compute loss
        for (int idx_ch = 0; idx_ch < num_channels; idx_ch++)
        {
            cv::Mat after_img2 = cv::Mat::zeros(label_height,label_width,CV_32F);
            cv::Mat after_img3 = cv::Mat::zeros(label_height,label_width,CV_32F);
            for (int i = 0; i < label_width; i++)
            {
                for (int j = 0; j < label_height; j++)
                {
                    int image_idx = idx_img * label_img_size + idx_ch * label_channel_size + i * label_height + j;
                    //@cf epipolar transform
                    cv::Mat p = (cv::Mat_<float>(3,1) << i, j, 1);
                    cv::Mat l2 = cv::Mat::zeros(3,1,CV_32F);
                    cv::Mat l3 = cv::Mat::zeros(3,1,CV_32F);
                    if(whichview == 1)
                    {
                        l2 = F12.t()*p;
                        l3 = F13.t()*p;
                       // std::cout<<"v1"<<std::endl;
                    }
                    if(whichview == 2)
                    {
                    	l2 = F12*p;
                    	l3 = F23*p;
                    	//std::cout<<"v2"<<std::endl;
                    }
                    if(whichview == 3)
                    {
                    	l2 = F13*p;
                    	l3 = F23.t()*p;
                    	//std::cout<<"v3"<<std::endl;
                    }
                    for(int x = 1;x<=label_width;x++)
                    {
                        //std::cout<<x<<std::endl;
                        //std::cout<<l2.at<float>(0,0)<<" "<<l2.at<float>(1,0)<<" "<<l2.at<float>(2,0)<<std::endl;
                    	int y2 = int((-(l2.at<float>(0,0)/l2.at<float>(1,0))*x-(l2.at<float>(2,0)/l2.at<float>(1,0)))+0.5);
                    	int y3 = int((-(l3.at<float>(0,0)/l3.at<float>(1,0))*x-(l3.at<float>(2,0)/l3.at<float>(1,0)))+0.5);
                        //std::cout<<y2<<std::endl;
                        //std::cout<<y3<<std::endl;
                        
                    	if(y2<=label_height&&y2>=1)
                    	{
                             
                                //std::cout<<bottom_pred2[image_idx]<<std::endl;
                                //std::cout<<y2<<std::endl;
                                //std::cout<<after_img2.at<float>(y2-1,x-1)<<std::endl;
                    	    if(bottom_pred2[image_idx]>after_img2.at<float>(y2-1,x-1))
                    		{
                               
                    			after_img2.at<float>(y2-1,x-1) = bottom_pred2[image_idx];
                    			int map_idx2 = idx_img * label_img_size + idx_ch * label_channel_size + (x-1) * label_height + y2-1;
                    			map2.mutable_cpu_data()[map_idx2] = image_idx;
                                        
                    		}
                       
                    	}
                        if(y3<=label_height&&y3>=1)
                   	    {
                   		    if(bottom_pred3[image_idx]>after_img3.at<float>(y3-1,x-1))
                   		    {
                   			    after_img3.at<float>(y3-1,x-1) = bottom_pred3[image_idx];
                   			    int map_idx3 = idx_img * label_img_size + idx_ch * label_channel_size + (x-1) * label_height + y3-1;
                   			    map3.mutable_cpu_data()[map_idx3] = image_idx;
                   		    }
                   	    }	
                       
                    }
       

                }
            }
            //compute loss
            for (int i = 0; i < label_width; i++)
            {
            	for (int j = 0; j < label_height; j++)
                {
            		int image_idx = idx_img * label_img_size + idx_ch * label_channel_size + i * label_height + j;
            		
            		float diff = (float)bottom_pred1[image_idx]*after_img2.at<float>(j,i)*after_img3.at<float>(j,i) - (float)gt_pred[image_idx];
            		//@cf store diff for backward propagate
            		for_diff.mutable_cpu_data()[image_idx] = diff;
            		loss += diff * diff;
            		
            		//@cf store after_img2 and use top[3] store after_img3
            		after_img2_all.mutable_cpu_data()[image_idx] = after_img2.at<float>(j,i);
            		after_img3_all.mutable_cpu_data()[image_idx] = after_img3.at<float>(j,i);
            		
                    // Store visualisation for given channel
                    if (idx_ch == visualise_channel && visualise)
                    {
                        bottom_img.at<float>((int)j, (int)i) = (float) bottom_pred1[image_idx];
                        gt_img.at<float>((int)j, (int)i) = (float) gt_pred[image_idx];
                        diff_img.at<float>((int)j, (int)i) = (float) diff * diff;
                    }

                }
            }
        }
        // Plot visualisation
        if (visualise)
        {
//            DLOG(INFO) << "num_images=" << num_images << " idx_img=" << idx_img;
//            DLOG(INFO) << "sum bottom: " << cv::sum(bottom_img) << "  sum gt: " << cv::sum(gt_img);
            int visualisation_size = 256;
            cv::Size size(visualisation_size, visualisation_size);            
            std::vector<cv::Point> points;
            this->Visualise(loss, bottom_img, gt_img, diff_img, points, size);
            this->VisualiseBottom(bottom, idx_img, visualise_channel, points, size);
            cv::waitKey(0);     // Wait forever a key is pressed
        }
    }
    
    DLOG(INFO) << "total loss: " << loss;
    loss /= (num_images * num_channels * label_channel_size);
    DLOG(INFO) << "total normalised loss: " << loss;

    top[0]->mutable_cpu_data()[0] = loss;

}



// Visualise GT heatmap, predicted heatmap, input image and max in heatmap
// bottom: predicted heatmaps
// gt: ground truth gaussian heatmaps
// diff: per-pixel loss
// overlay: prediction with GT location & max of prediction
// visualisation_bottom: additional visualisation layer (defined as the last 'bottom' in the loss prototxt def)
template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Visualise(float loss, cv::Mat bottom_img, cv::Mat gt_img, cv::Mat diff_img, std::vector<cv::Point>& points, cv::Size size)
{
    DLOG(INFO) << loss;

    // Definitions
    double minVal, maxVal;
    cv::Point minLocGT, maxLocGT;
    cv::Point minLocBottom, maxLocBottom;
    cv::Point minLocThird, maxLocThird;
    cv::Mat overlay_img_orig, overlay_img;

    // Convert prediction (bottom) into 3 channels, call 'overlay'
    overlay_img_orig = bottom_img.clone() - 1;
    cv::Mat in[] = {overlay_img_orig, overlay_img_orig, overlay_img_orig};
    cv::merge(in, 3, overlay_img);

    // Resize all images to fixed size
    PrepVis(bottom_img, size);
    cv::resize(bottom_img, bottom_img, size);
    PrepVis(gt_img, size);
    cv::resize(gt_img, gt_img, size);
    PrepVis(diff_img, size);
    cv::resize(diff_img, diff_img, size);
    PrepVis(overlay_img, size);
    cv::resize(overlay_img, overlay_img, size);

    // Get and plot GT position & prediction position in new visualisation-resized space
    cv::minMaxLoc(gt_img, &minVal, &maxVal, &minLocGT, &maxLocGT);
    DLOG(INFO) << "gt min: " << minVal << "  max: " << maxVal;
    cv::minMaxLoc(bottom_img, &minVal, &maxVal, &minLocBottom, &maxLocBottom);
    DLOG(INFO) << "bottom min: " << minVal << "  max: " << maxVal;
    cv::circle(overlay_img, maxLocGT, 5, cv::Scalar(0, 255, 0), -1);
    cv::circle(overlay_img, maxLocBottom, 3, cv::Scalar(0, 0, 255), -1);

    // Show visualisation images
    cv::imshow("bottom", bottom_img - 1);
    cv::imshow("gt", gt_img - 1);
    cv::imshow("diff", diff_img);
    cv::imshow("overlay", overlay_img - 1);

    // Store max locations
    points.push_back(maxLocGT);
    points.push_back(maxLocBottom);
}

// Plot another visualisation image overlaid with ground truth & prediction locations
// (particularly useful e.g. if you set this to the original input image)
template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::VisualiseBottom(const vector<Blob<Dtype>*>& bottom, int idx_img, int visualise_channel, std::vector<cv::Point>& points, cv::Size size)
{
    // Determine which layer to visualise
    Blob<Dtype>* visualisation_bottom = bottom[4];
    DLOG(INFO) << "visualisation_bottom: " << visualisation_bottom->channels() << " " << visualisation_bottom->height() << " " << visualisation_bottom->width();

    // Format as RGB / gray
    bool isRGB = visualisation_bottom->channels() == 3;
    cv::Mat visualisation_bottom_img;
    if (isRGB)
        visualisation_bottom_img = cv::Mat::zeros(visualisation_bottom->height(), visualisation_bottom->width(), CV_32FC3);
    else
        visualisation_bottom_img = cv::Mat::zeros(visualisation_bottom->height(), visualisation_bottom->width(), CV_32FC1);

    // Convert frame from Caffe representation to OpenCV image
    for (int idx_ch = 0; idx_ch < visualisation_bottom->channels(); idx_ch++)
    {
        for (int i = 0; i < visualisation_bottom->height(); i++)
        {
            for (int j = 0; j < visualisation_bottom->width(); j++)
            {
                int image_idx = idx_img * visualisation_bottom->width() * visualisation_bottom->height() * visualisation_bottom->channels() + idx_ch * visualisation_bottom->width() * visualisation_bottom->height() + i * visualisation_bottom->height() + j;
                if (isRGB && idx_ch < 3) {
                    visualisation_bottom_img.at<cv::Vec3f>((int)j, (int)i)[idx_ch] = 4 * (float) visualisation_bottom->cpu_data()[image_idx] / 255;
                } else if (idx_ch == visualise_channel)
                {
                    visualisation_bottom_img.at<float>((int)j, (int)i) = (float) visualisation_bottom->cpu_data()[image_idx];
                }
            }
        }
    }
    PrepVis(visualisation_bottom_img, size);

    // Convert colouring if RGB
    if (isRGB)
        cv::cvtColor(visualisation_bottom_img, visualisation_bottom_img, CV_RGB2BGR);

    // Plot max of GT & prediction
    cv::Point maxLocGT = points[0];
    cv::Point maxLocBottom = points[1];    
    cv::circle(visualisation_bottom_img, maxLocGT, 5, cv::Scalar(0, 255, 0), -1);
    cv::circle(visualisation_bottom_img, maxLocBottom, 3, cv::Scalar(0, 0, 255), -1);

    // Show visualisation
    cv::imshow("visualisation_bottom", visualisation_bottom_img - 1);
}



// Convert from Caffe representation to OpenCV img
template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::PrepVis(cv::Mat img, cv::Size size)
{
    cv::transpose(img, img);
    cv::flip(img, img, 1);
}


template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    const int count = bottom[1]->count();
    //const int channels = bottom[1]->channels();
    //const int num_images = bottom[1]->num();
    //const int height = bottom[1]->height();
    //const int width = bottom[1]->width();
    //const int channel_size = heigt * width;
    //const int img_size = channel_size * channels;
    
    //@cf bp for view1
    if(propagate_down[1])
    {
    	caffe_mul(count, for_diff.cpu_data(), after_img2_all.cpu_data(), temp1.mutable_cpu_data());
    	caffe_mul(count, temp1.cpu_data(), after_img3_all.cpu_data(), bottom[1]->mutable_cpu_diff());
    	//std::cout<<"1"<<std::endl;
    	//memcpy(bottom[1]->mutable_cpu_diff(), diff1->cpu_data(), sizeof(Dtype) * count);
    }
    
    //@cf bp for view2
    if(propagate_down[2])
    {
    	caffe_mul(count, for_diff.cpu_data(), bottom[1]->cpu_data(), temp2.mutable_cpu_data());
    	caffe_mul(count, temp2.cpu_data(), after_img3_all.cpu_data(), diff_temp.mutable_cpu_data());
    	for(int idx = 0; idx < count; idx++)
    	{
    		int transform_idx = map2.cpu_data()[idx];
    		if(transform_idx != -1)
    			bottom[2]->mutable_cpu_diff()[transform_idx] = diff_temp.cpu_data()[idx];
    	}
    	//memcpy(bottom[2]->mutable_cpu_diff(), diff2->cpu_data(), sizeof(Dtype) * count);
    	//std::cout<<"2"<<std::endl;
    }
    
    //@cf bp for view3
    if(propagate_down[3])
    {
    	caffe_mul(count, for_diff.cpu_data(), bottom[1]->cpu_data(), temp3.mutable_cpu_data());
    	caffe_mul(count, temp3.cpu_data(), after_img2_all.cpu_data(), diff_temp1.mutable_cpu_data());
    	for(int idx = 0; idx < count; idx++)
    	{
    		int transform_idx = map3.cpu_data()[idx];
    		if(transform_idx != -1)
    			bottom[3]->mutable_cpu_diff()[transform_idx] = diff_temp1.cpu_data()[idx];
    	}
    	//memcpy(bottom[3]->mutable_cpu_diff(), diff3->cpu_data(), sizeof(Dtype) * count);
    	//std::cout<<"3"<<std::endl;
    }

 /******************************for original backward_cpu code *************************************/
    //caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_.mutable_cpu_data());

    // strictly speaking, should be normalising by (2 * channels) due to 1/2 multiplier in front of the loss
    //Dtype loss = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data()) / Dtype(channels);

    // copy the gradient
    //memcpy(bottom[0]->mutable_cpu_diff(), diff_.cpu_data(), sizeof(Dtype) * count);
    //memcpy(bottom[1]->mutable_cpu_diff(), diff_.cpu_data(), sizeof(Dtype) * count);
 /**************************************************************************************************/
}


template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}

template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}



#ifdef CPU_ONLY
STUB_GPU(EuclideanLossHeatmapLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossHeatmapLayer);
REGISTER_LAYER_CLASS(EuclideanLossHeatmap);


}  // namespace caffe
