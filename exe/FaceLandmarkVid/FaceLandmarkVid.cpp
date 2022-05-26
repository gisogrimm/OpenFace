///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////
// FaceTrackingVid.cpp : Defines the entry point for the console application for tracking faces in videos.

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
#include <Face_utils.h>
#include <FaceAnalyser.h>

#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>

#include <lo/lo.h>
#include <lo/lo_cpp.h>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
	abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

std::vector<std::string> get_arguments(int argc, char **argv)
{

	std::vector<std::string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(std::string(argv[i]));
	}
	return arguments;
}

int main(int argc, char **argv)
{

	std::vector<std::string> arguments = get_arguments(argc, argv);

	// no arguments: output usage
	if (arguments.size() == 1)
	{
		std::cout << "For command line arguments see:" << std::endl;
		std::cout << " https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments";
		return 0;
	}

	LandmarkDetector::FaceModelParameters det_parameters(arguments);

        FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
        FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

        // The modules that are being used for tracking
        LandmarkDetector::CLNF face_model(det_parameters.model_location);
        if (!face_model.loaded_successfully)
	{
		std::cout << "ERROR: Could not load the landmark detector" << std::endl;
		return 1;
	}

	if (!face_model.eye_model)
	{
		std::cout << "WARNING: no eye model found" << std::endl;
	}

	// Open a sequence
	Utilities::SequenceCapture sequence_reader;

	// A utility for visualizing the results (show just the tracks)
	Utilities::Visualizer visualizer(true, false, false, false);

	// Tracking FPS for visualization
	Utilities::FpsTracker fps_tracker;
	fps_tracker.AddFrame();

	int sequence_number = 0;

        lo::Address to_tascar("localhost", "9877");
        lo::Address to_blender("localhost", "9999");

	while (true) // this is not a for loop as we might also be reading from a webcam
	{

		// The sequence reader chooses what to open based on command line arguments provided
		if (!sequence_reader.Open(arguments))
			break;

		INFO_STREAM("Device or file opened");

		cv::Mat rgb_image = sequence_reader.GetNextFrame();

		INFO_STREAM("Starting tracking");
		while (!rgb_image.empty()) // this is not a for loop as we might also be reading from a webcam
		{

			// Reading the images
			cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame();

			// The actual facial landmark detection / tracking
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(rgb_image, face_model, det_parameters, grayscale_image);

			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);

			// If tracking succeeded and we have an eye model, estimate gaze
			if (detection_success && face_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(face_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
				GazeAnalysis::EstimateGaze(face_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);

                                auto gazeAngle = GazeAnalysis::GetGazeAngle(
                                    gazeDirection0, gazeDirection1);
                                to_tascar.send("/gaze", "ff", gazeAngle(0),
                                       gazeAngle(1));
                        }

			// Work out the pose of the head from the tracked model
			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

                        to_tascar.send("/headpos", "fff", pose_estimate[0], pose_estimate[1],
                                       pose_estimate[2]);
                        to_tascar.send("/headrot", "fff", pose_estimate[3], pose_estimate[4],
                                       pose_estimate[5]);

                        if(detection_success) {

                          auto landmarks =
                              face_model.GetShape(sequence_reader.fx, sequence_reader.fy,
                                                  sequence_reader.cx, sequence_reader.cy);
                          for(size_t k = 0; k < landmarks.cols; ++k)
                            to_tascar.send("/lm" + std::to_string(k), "fff",
                                           landmarks(0, k) - pose_estimate[0],
                                           landmarks(1, k) - pose_estimate[1],
                                           landmarks(2, k) - pose_estimate[2]);

                          double mouthopen =
                              std::max(0.0, sqrt(pow(landmarks(0, 57) - landmarks(0, 51), 2.0) +
                                                 pow(landmarks(1, 57) - landmarks(1, 51), 2.0) +
                                                 pow(landmarks(2, 57) - landmarks(2, 51), 2.0)) -
                                                8.0) /
                              10.0;

                          double mouthwidth =
                              1.0 -
                              std::min(
                                  1.0,
                                  std::max(0.0,
                                           sqrt(pow(landmarks(0, 54) - landmarks(0, 48), 2.0) +
                                                pow(landmarks(1, 54) - landmarks(1, 48), 2.0) +
                                                pow(landmarks(2, 54) - landmarks(2, 48), 2.0)) -
                                               30.0) /
                                      30.0);

                          to_blender.send("/maartje", "sfff", "/lipsync", (float)mouthwidth, 0.0f,
                                          (float)mouthopen);
                          to_blender.send("/maartje", "sfff", "/headGaze", pose_estimate[3],
                                          pose_estimate[4], pose_estimate[5]);
                          to_tascar.send("/mouthopen", "f", mouthopen);
                          to_tascar.send("/mouthwidth", "f", mouthwidth);

                          face_analyser.AddNextFrame(rgb_image, face_model.detected_landmarks,
                                                     face_model.detection_success,
                                                     sequence_reader.time_stamp,
                                                     sequence_reader.IsWebcam());

                          //face_analyser.PredictStaticAUsAndComputeFeatures(
                          //   rgb_image, face_model.detected_landmarks);

                          auto aus_intensity = face_analyser.GetCurrentAUsReg();
                          auto aus_presence = face_analyser.GetCurrentAUsClass();
                          for(size_t k = 0; k < aus_intensity.size(); ++k)
                            to_tascar.send("/au" + std::to_string(k) + "i", "f", aus_intensity[k]);
                          for(size_t k = 0; k < aus_presence.size(); ++k)
                            to_tascar.send("/au" + std::to_string(k) + "p", "f", aus_presence[k]);
                        }

                        // Keeping track of FPS
			fps_tracker.AddFrame();

			// Displaying the tracking visualizations
			visualizer.SetImage(rgb_image, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
			visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, face_model.GetVisibilities());
			visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
			visualizer.SetObservationGaze(gazeDirection0, gazeDirection1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_model.detection_certainty);
			visualizer.SetFps(fps_tracker.GetFPS());
			// detect key presses (due to pecularities of OpenCV, you can get it when displaying images)
			char character_press = visualizer.ShowObservation();

			// restart the tracker
			if (character_press == 'r')
			{
				face_model.Reset();
			}
			// quit the application
			else if (character_press == 'q')
			{
				return(0);
			}

			// Grabbing the next frame in the sequence
			rgb_image = sequence_reader.GetNextFrame();

		}

		// Reset the model, for the next video
		face_model.Reset();
		sequence_reader.Close();

		sequence_number++;

	}
	return 0;
}


/*
 * Local Variables:
 * compile-command: "make -C ../../build"
 * End:
 */
