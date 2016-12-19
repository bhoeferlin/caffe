#include <visionalytics/classifier.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <fstream>

using namespace caffe;  
using namespace std;


    const std::string Classifier::RETRAIN_SNAPSHOTPATH_PLACEHOLDER = "%retrainSnapshotPath%";
    const std::string Classifier::RETRAIN_MODELPATH_PLACEHOLDER = "%retrainModelPath%";
    const std::string Classifier::RETRAIN_TRAINDATAPATH_PLACEHOLDER = "%trainAnnotationPath%"; 
    const std::string Classifier::RETRAIN_VALIDDATAPATH_PLACEHOLDER = "%validAnnotationPath%";

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

    n = std::min<int>( static_cast<int>( m_outputLabelAssignment.size() ), n );
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



    
void Classifier::retrain( const cv::Ptr< cv::ml::TrainData >& trainingAndValidationData, 
                          const cv::Size2i& imgDimensions,
                          unsigned int numberOfTrainIterations, 
                          std::function<void(float)> progressUpdate,
                          std::function<void(unsigned int, float, float, const std::string&)> snapshotUpdate,
                          const std::string& solver_file,
                          const std::string& model_file,
                          const std::string& trained_file,
                          const std::string& folderToStoreData )
{
    const bool useGPUForTrain = false;

    cv::Mat trainSamples = trainingAndValidationData->getTrainSamples();
    cv::Mat validSamples = trainingAndValidationData->getTestSamples();

    cv::Mat trainLabels = trainingAndValidationData->getTrainResponses();
    cv::Mat validLabels = trainingAndValidationData->getTestResponses();


    // Create inverse label assignment
    cv::Mat classLabels = trainingAndValidationData->getClassLabels();
    double minV, maxV;
    cv::minMaxLoc(classLabels, &minV, &maxV);
    int maxLabelID = static_cast<int>( maxV );
    for( int i = 0; i < m_outputLabelAssignment.size(); ++i )
    {
        maxLabelID = max( maxLabelID, m_outputLabelAssignment.at( i ) );
    }

    std::vector< int > inverseLabelAssignment;
    inverseLabelAssignment.resize( maxLabelID + 1, -1 );
    for( int i = 0; i < m_outputLabelAssignment.size(); ++i )
    {
        inverseLabelAssignment.at( m_outputLabelAssignment.at( i ) ) = i;
    }
    

    // Prepare the training and validation datasets
    std::string baseFolder = folderToStoreData;
    if( baseFolder.empty() )
    {
        baseFolder = "c:/tmp_train";
    }

    boost::system::error_code dirError;
    CHECK( boost::filesystem::create_directory( baseFolder + "/train_data/", dirError ) ) <<
        "Cannot create temporary train_data path due to error " << dirError;
    CHECK( boost::filesystem::create_directory( baseFolder + "/valid_data/", dirError ) ) <<
        "Cannot create temporary valid_data path due to error " << dirError;        

    for( int testValidCycle = 0; testValidCycle < 2; ++testValidCycle )
    {
        cv::Mat& samples = trainSamples;
        cv::Mat& responses = trainLabels;
        std::string annotationFilename;
        std::string dataPlaceholder;
        FILE* annotationsFile; 

        switch( testValidCycle )
        {
        case 0: // train
            samples = trainSamples;
            responses = trainLabels;
            annotationFilename = baseFolder + "/train_annotations.txt";
            dataPlaceholder = Classifier::RETRAIN_TRAINDATAPATH_PLACEHOLDER;
            break;
        case 1: // valid
            samples = validSamples;
            responses = validLabels;
            annotationFilename = baseFolder + "/valid_annotations.txt";
            dataPlaceholder = Classifier::RETRAIN_VALIDDATAPATH_PLACEHOLDER;
            break;
        }

        // Generate shuffled data indices
        std::vector<int> shuffledIndex;
        shuffledIndex.reserve( samples.rows );
        for( int i = 0; i < samples.rows; ++i )
        {
	        shuffledIndex.push_back( i );
        }
        random_shuffle( shuffledIndex.begin(), shuffledIndex.end() );


        // Write train and validation datasets
        annotationsFile = fopen( annotationFilename.c_str(), "w" );
        if( !annotationsFile )
        {
            assert( false );
            return;
        }

        for( int i = 0; i < samples.rows; ++i )
        {
            char tmpBuffer[256];
            cv::Mat sample = samples.row( shuffledIndex.at( i ) );
            cv::Mat response = responses.row( shuffledIndex.at( i ) );

            int classResponse;
            if( inverseLabelAssignment.size() > (int)response.at<float>(0,0) && 
                inverseLabelAssignment.at( (int)response.at<float>(0,0) ) < 0 )
            {
                assert( false );
                return;
            }
            else
            {
                classResponse = inverseLabelAssignment.at( (int)response.at<float>(0,0) );
            }


            std::string imgFilename;
            if( testValidCycle == 0 )
            {
                imgFilename = baseFolder + "/train_data/" + std::string( _itoa( i, tmpBuffer, 10 ) ) + ".png";
            }
            else
            {
                assert( testValidCycle == 1 );
                imgFilename = baseFolder + "/valid_data/" + std::string( _itoa( i, tmpBuffer, 10 ) ) + ".png";
            }

            // prepare train features
            assert( sample.cols == imgDimensions.width * imgDimensions.height * 3 );
		    cv::Mat reshapedSample = sample.reshape( 3, imgDimensions.height );    

            // store sample
            if( !cv::imwrite( imgFilename, reshapedSample ) )
            {
                assert( false );
                return;
            }

            fprintf( annotationsFile, "%s %d\n", imgFilename.c_str(), classResponse );
        }

        fclose( annotationsFile );

        // Replace annotation data placeholder in retrain model file
        ReplaceAllPlaceholdersInTextFile( model_file, dataPlaceholder, annotationFilename );
    }


    // Replace placeholdera in retrain solver file
    ReplaceAllPlaceholdersInTextFile( solver_file, Classifier::RETRAIN_MODELPATH_PLACEHOLDER, model_file );
    ReplaceAllPlaceholdersInTextFile( solver_file, Classifier::RETRAIN_SNAPSHOTPATH_PLACEHOLDER, folderToStoreData + "/snapshot" );


    // Init solver parameters
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie( solver_file, &solver_param );

    // Set snapshot props
    solver_param.set_snapshot_after_train( true );
    solver_param.set_snapshot_format( caffe::SolverParameter_SnapshotFormat_BINARYPROTO );

    // Display props
    solver_param.set_display( false );
    solver_param.set_debug_info( false );

    

    // Set GPU devices for training
    std::vector< int > gpus;
    if( useGPUForTrain )
    {
        std::string useGPUs = "all";

        // Set the mode and device to be set from the solver prototxt.
        if( solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ) 
        {
            if( solver_param.has_device_id() ) 
            {
                useGPUs = boost::lexical_cast<std::string>(solver_param.device_id());
            } 
            else 
            {  // Set default GPU if unspecified
                useGPUs = "0";
            }
        }

        // Determine which devices are available
        if (useGPUs == "all") 
        {
            int count = 0;
            CUDA_CHECK( cudaGetDeviceCount( &count ) );
            for( int i = 0; i < count; ++i ) 
            {
                gpus.push_back( i );
            }
        } 
        else if( useGPUs.size() ) 
        {
            vector<string> strings;
            boost::split( strings, useGPUs, boost::is_any_of(",") );
            for (int i = 0; i < strings.size(); ++i) 
            {
                gpus.push_back(boost::lexical_cast<int>(strings[i]));
            }
        } 

        if( gpus.size() == 0 ) 
        {
            LOG(INFO) << "Use CPU.";
            Caffe::set_mode( Caffe::CPU );
        } 
        else 
        {
            ostringstream s;
            for (int i = 0; i < gpus.size(); ++i) 
            {
                s << (i ? ", " : "") << gpus[i];
            }
            LOG(INFO) << "Using GPUs " << s.str();

            cudaDeviceProp device_prop;
            for( int i = 0; i < gpus.size(); ++i ) 
            {
                cudaGetDeviceProperties( &device_prop, gpus[i] );
                LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
            }

            solver_param.set_device_id( gpus[0] );
            Caffe::SetDevice( gpus[0] );
            Caffe::set_mode( Caffe::GPU );
            Caffe::set_solver_count( static_cast<int>( gpus.size() ) );
        }
    }


    // Create solver
    boost::shared_ptr<caffe::Solver<float> > solver( caffe::SolverRegistry<float>::CreateSolver( solver_param ) );
    solver->net()->CopyTrainedLayersFrom( trained_file );
    for( int j = 0; j < solver->test_nets().size(); ++j ) 
    {
        solver->test_nets()[j]->CopyTrainedLayersFrom( trained_file );
    }
  
    // Add callbacks
    SolverCallback<float> callbackClass( this, progressUpdate, snapshotUpdate );
    solver->add_callback( &callbackClass );
           

    /*
    caffe::SignalHandler signal_handler(
    GetRequestedAction(FLAGS_sigint_effect),
    GetRequestedAction(FLAGS_sighup_effect));
    solver->SetActionFunction(signal_handler.GetActionFunction());
    */


    if( gpus.size() > 1 ) 
    {
        caffe::P2PSync<float> sync( solver, NULL, solver->param() );
        sync.Run(gpus);
    } 
    else 
    {
        LOG(INFO) << "Starting Optimization";
        solver->Solve(); // TODO - get Feedback on snapshots and performance
    }

    LOG(INFO) << "Optimization Done.";

}



    
unsigned int Classifier::ReplaceAllPlaceholdersInTextFile( const std::string& filename, const std::string& from, const std::string& to )
{
    unsigned int replacedOccurrences = 0;
    FILE* replaceFile; 
    replaceFile = fopen( filename.c_str(), "rb+" );

    // check how many characters are there in the file
    fseek( replaceFile, 0, SEEK_END );
    long fsize = ftell( replaceFile );
    fseek( replaceFile, 0, SEEK_SET ); 

    if( fsize == 0 )
    {
        fclose( replaceFile );
        return replacedOccurrences;
    }

    // read all characters and append null termination
    char* fileContent = new char[ fsize + 1 ];
    fread( fileContent, fsize, 1, replaceFile );
    fileContent[ fsize ] = '\0';

    // convert to c++ string and replace all occurences of "from" with "to"
    std::string content = std::string( fileContent );

    size_t start_pos = 0;
    while( ( start_pos = content.find( from, start_pos ) ) != std::string::npos ) 
    {
        content.replace( start_pos, from.length(), to );
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
        ++replacedOccurrences;
    }

    // write back the modified content
    fseek( replaceFile, 0, SEEK_SET ); 
    fwrite( content.c_str() , sizeof(char), content.length(), replaceFile);

    fclose( replaceFile );

    return replacedOccurrences;
}
