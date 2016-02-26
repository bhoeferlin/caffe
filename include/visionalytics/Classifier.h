#ifndef CLASSIFIER_HEADER__
#define CLASSIFIER_HEADER__

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe/caffe.hpp>

/*
#pragma comment(lib, "opencv_core300.lib")
#pragma comment(lib, "opencv_highgui300.lib")
#pragma comment(lib, "opencv_imgproc300.lib")
#pragma comment(lib, "opencv_imgcodecs300.lib")
*/

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "nppi.lib")
#pragma comment(lib, "cufft.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "libopenblas.lib")
#pragma comment(lib, "libhdf5.lib")
#pragma comment(lib, "libhdf5_cpp.lib")
#pragma comment(lib, "libhdf5_hl.lib")
#pragma comment(lib, "libhdf5_hl_cpp.lib")
#pragma comment(lib, "libhdf5_tools.lib")
#pragma comment(lib, "libglog_static.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "gflags.lib")

#ifdef _DEBUG
#   pragma comment(lib, "libprotobufd.lib")
#   pragma comment(lib, "libprotocd.lib")
#else
#   pragma comment(lib, "libprotobuf.lib")
#   pragma comment(lib, "libprotoc.lib")
#endif


/* Pair (label, confidence) representing a prediction. */
typedef std::pair< int, float > Prediction;

class Classifier 
{
public:
    Classifier( bool useGPU, 
                const std::string& model_file,
                const std::string& trained_file,
                const std::string& mean_file,
                const std::vector< int >& outputLabelAssignment );

    Classifier( bool useGPU, 
                const std::string& model_file,
                const std::string& trained_file,
                const cv::Scalar& meanPerChannel,
                const std::vector< int >& outputLabelAssignment );
        
    // Return the top N predictions.
    std::vector< Prediction > classify( const cv::Mat& img, int n = 5 );

    // Load the mean file in binaryproto format.
    static cv::Scalar GetMean( const std::string& mean_file, const unsigned int& num_channels );

private:

    void init( bool useGPU, 
               const std::string& model_file,
               const std::string& trained_file,
               const std::vector< int >& outputLabelAssignment );


    void createMeanImage( const cv::Scalar& channel_mean );
    
    std::vector< float > predict( const cv::Mat& img );

    /** Wrap the input layer of the network in separate cv::Mat objects
     * (one per channel). This way we save one memcpy operation and we
     * don't need to rely on cudaMemcpy2D. The last preprocessing
     * operation will write the separate channels directly to the input
     * layer. 
     */
    void wrapInputLayer( std::vector< cv::Mat >* input_channels );

    void preprocess( const cv::Mat& img,
                     std::vector<cv::Mat>* input_channels );

 private:

    std::shared_ptr< caffe::Net<float> > m_net;
    cv::Size m_input_geometry;
    int m_num_channels;
    cv::Mat m_mean; 

    std::vector< int > m_outputLabelAssignment;
};

#endif // CLASSIFIER_HEADER__