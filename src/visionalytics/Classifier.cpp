#include <visionalytics/classifier.h>

#include <boost/filesystem.hpp>
#include <fstream>

using namespace caffe;  
using namespace std;


bool Classifier::GoogleLoggingInitialized = Classifier::InitializeGoogleLogging();


bool Classifier::InitializeGoogleLogging()
{
    ::google::InitGoogleLogging("CaffeClassifier");
    return true;
}


Classifier::Classifier( bool useGPU, 
                        const string& model_file,
                        const string& trained_file,
                        const cv::Scalar& meanPerChannel,
                        const std::vector< int >& outputLabelAssignment ) 
{
    init( useGPU, model_file, trained_file, outputLabelAssignment );

    createMeanImage( meanPerChannel );
}



Classifier::Classifier( bool useGPU, 
                        const string& model_file,
                        const string& trained_file,
                        const string& mean_file,
                        const std::vector< int >& outputLabelAssignment ) 
{
    init( useGPU, model_file, trained_file, outputLabelAssignment );

    // Load the binaryproto mean file.
    createMeanImage( GetMean( mean_file, m_num_channels ) );
}



void Classifier::init( bool useGPU, 
            const std::string& model_file,
            const std::string& trained_file,
            const std::vector< int >& outputLabelAssignment )
{
    if( useGPU )
    {
        Caffe::set_mode(Caffe::GPU);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    // Load the network. 
    m_net.reset( new Net<float>( model_file, TEST ) );
    m_net->CopyTrainedLayersFrom( trained_file );

    CHECK_EQ( m_net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ( m_net->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = m_net->input_blobs()[0];
    m_num_channels = input_layer->channels();
    CHECK( m_num_channels == 3 || m_num_channels == 1 ) << "Input layer should have 1 or 3 channels.";
    m_input_geometry = cv::Size( input_layer->width(), input_layer->height() );
    
    // store output label assignments
    m_outputLabelAssignment = outputLabelAssignment;
    Blob<float>* output_layer = m_net->output_blobs()[0];
    CHECK_EQ( m_outputLabelAssignment.size(), output_layer->channels() ) << "Number of labels is different from the output layer dimension.";
}



static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) 
{
    return lhs.first > rhs.first;
}



// Return the indices of the top N values of vector v.
static std::vector<int> Argmax(const std::vector<float>& v, int N) 
{
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
    {
        pairs.push_back( std::make_pair( v[i], (int) i ) );
    }
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
    {
        result.push_back(pairs[i].second);
    }
    return result;
}



std::vector<Prediction> Classifier::classify(const cv::Mat& img, int n) 
{
    std::vector<float> output = predict(img);

    n = std::min<int>( m_outputLabelAssignment.size(), n );
    std::vector<int> maxN = Argmax(output, n);
    std::vector<Prediction> predictions;
    for (int i = 0; i < n; ++i) 
    {
        int idx = maxN[i];
        predictions.push_back( std::make_pair( m_outputLabelAssignment[idx], output[idx] ) );
    }

    return predictions;
}


 
cv::Scalar Classifier::GetMean( const string& mean_file, const unsigned int& num_channels ) 
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie( mean_file.c_str(), &blob_proto );

    // Convert from BlobProto to Blob<float> 
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels) << "Number of channels of mean file doesn't match input layer.";

    // The format of the mean file is planar 32-bit float BGR or grayscale. 
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for( unsigned int i = 0; i < num_channels; ++i ) 
    {
        // Extract an individual channel. 
        cv::Mat channel( mean_blob.height(), mean_blob.width(), CV_32FC1, data );
        channels.push_back( channel );
        data += mean_blob.height() * mean_blob.width();
    }

    // Merge the separate channels into a single image.
    cv::Mat mean;
    cv::merge(channels, mean);

    // Compute the global mean pixel value and create a mean image
    // filled with this value. 
    return cv::mean( mean );
}


  
void Classifier::createMeanImage( const cv::Scalar& channel_mean )
{
     m_mean = cv::Mat( m_input_geometry, /*mean.type()*/ CV_32FC3, channel_mean );
}



std::vector< float > Classifier::predict( const cv::Mat& img ) 
{
    Blob<float>* input_layer = m_net->input_blobs()[0];
    input_layer->Reshape( 1, m_num_channels,
                          m_input_geometry.height, m_input_geometry.width );

    // Forward dimension change to all layers.
    m_net->Reshape();

    std::vector<cv::Mat> input_channels;
    wrapInputLayer( &input_channels );

    preprocess( img, &input_channels );

    m_net->ForwardPrefilled();

    // Copy the output layer to a std::vector
    Blob<float>* output_layer = m_net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>( begin, end );
}



void Classifier::wrapInputLayer( std::vector< cv::Mat >* input_channels) 
{
    Blob<float>* input_layer = m_net->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) 
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}


void Classifier::preprocess( const cv::Mat& img,
                             std::vector<cv::Mat>* input_channels) 
{
    // Convert the input image to the input image format of the network. 
    cv::Mat sample;
    if (img.channels() == 3 && m_num_channels == 1)
        cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && m_num_channels == 1)
       cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && m_num_channels == 3)
       cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && m_num_channels == 3)
        cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if ( sample.size() != m_input_geometry )
    {
        cv::resize(sample, sample_resized, m_input_geometry);
    }
    else
    {
        sample_resized = sample;
    }

    cv::Mat sample_float;
    if( sample_resized.depth() != CV_32F )
    {
        if (m_num_channels == 3)
            sample_resized.convertTo(sample_float, CV_32FC3);
        else
            sample_resized.convertTo(sample_float, CV_32FC1);
    }
    else
    {
        sample_float = sample_resized;
    }

    cv::Mat sample_normalized;
    cv::subtract( sample_float, m_mean, sample_normalized );

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK( reinterpret_cast<float*>(input_channels->at(0).data)
        == m_net->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

