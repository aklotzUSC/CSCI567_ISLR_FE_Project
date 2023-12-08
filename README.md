# CSCI567_ISLR_FE_Project

Start by running 'pip install -r requirements.txt' to install requirements.

### Keypoint Extraction

As keypoint extraction is heavy on computing and ram, and openpose's instructions and files are outdated making it hard to get working,
the processed keypoints have been provided in the form of the WLASL200_mediapipe.pkl and WLASL200_openpose.pkl files. If you'd like to generate
these files yourself, below are instructions.

The complete preprocessed dataset was provided by the author of the WLASL dataset via drive at https://drive.google.com/file/d/11eFE_quM2_2-h3H_zTTjq0i0D6pkx62Z/view.
In order to download the videos and preprocess them yourself you can follow the instructions at WLASL's github https://github.com/dxli94/WLASL, though note
this was attempted in this project and foregone due to many of the original sources of the videos not being available.

To generate the mediapipe keypoints, utilize mediapie_keypoints.py. Before running, change the vid_dir variable to the directory which the videos were installed to.

Generating the openpose keypoints is a lot more difficult as openpose does not have a library available via pip. The instructions for installing openpose with python can be found here: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md
Below is a list of issues ran into when attempting to install openpose on Windows and the fixes found, it is not guarenteed these are all the issues you will run into if you attempt to install yourself as the instructions are out of date.
* Python versions newer than 3.10 do not appear to work.
* The host for the dependencies attempted to be installed during cmake-GUI is down. From https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1602 someone provided google drive links to download the 3rdparty folder (https://drive.google.com/file/d/1WvftDLLEwAxeO2A-n12g5IFtfLbMY9mG/edit) and the model folder (https://drive.google.com/file/d/1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh/edit). Manually put these in your openpose directory before configuring.
* The instructions don't say to, but it seems necessary to check the PYBIND11_INSTALL option in cmake-GUI after configuring with the BUILD_PYTHON option and configure again.
* If the toolchain selected by default in VS is wrong, select all the projects in your solution in VS and right click properties and change the toolset to an installed toolset.
* If you get an issue with the ssize_t definition when building in VS, follow the instructions of the top comment at https://github.com/davisking/dlib/issues/2463 to change numpy.h. You can direcly locate numpy.h by double clicking the error in VS. You may need to clean your solution before building again.
* openpose_keypoints.py utilizes a different method of letting the program find the necessary DLLs than the given examples by openpose. The given examples add the path of the directory containing the DLLs whereas openpose_keypoints.py utilizes os.add_dll_directory as I found only the latter to work. If you get an error, you maybe can try referring to the given openpose examples and try their method of adding DLL directories.

If you have openpose installed correctly, change the location of dir in openpose_keypoints.py to where openpose was downloaded and vid_dir to the location of your video directory. Note that the program assumes you utilized openpose/build as the directory you generate into when using cmake-GUI during openpose installation as given as default. If you changed the name of your build location, you will need to adjust lines 20-24 to account for the difference in directory names.

Both openpose_keypoints.py and mediapie_keypoints.py process videos for the top 200 glosses. If you'd like to process for a different number, change the MAX_GLOSS_COUNT variable. The complete WLASL dataset has 2000 glosses.