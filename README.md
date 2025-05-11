Prerrequisites:
 - Python 3.10 or higher.
 - Install required libreries with: `pip install requirements.txt`.
 - Install git if not installed: `https://git-scm.com/downloads`.
 - Clone XMem: `git clone XMem`.
 - Create account at CVAT: `https://app.cvat.ai/`.
 - Create a swift project if not created, with the names and colors defined in `labelmap.txt`.

Steps:
Extract frames from video: 
 - `video-toimg data/video.mp4`, or `video-toimg --per PER data/video.mp4` to only extract 1 frame for every PER frames, if the video is e.g. 30 fps `--per 6`, this turns it to 30/6=5 fps. This can save space if the videos are long.
Labelling:
 - Automatic labelling not functional yet.
 - Semi-automatic labeling using XMem and CVAT:
   1. Add a task for the video to label, and upload at least 1 frame for it, possibly several, one for every e.g. 10-20 minutes of video.
   2. Open the task and use the magic wand on the right, select 'Interact' and enable 'Start with bounding box'.
   3. Draw bounding box around object (swift or nest) and after that use right click to add missing areas from the auto-generated mask, or left click to remove them. Do this for all obhects in the frame. And all updated frames.
   4. Export the frames going to 'Menu', 'Export job dataset'. Select 'Segmentation mask 1.1' as the putput format. A name can be given now to the download file. Click 'Ok' and go to 'Requests'. Once the request finishes it can be downloaded from the 3 dots on its right. Repeat this for all the videos that need to be analyzed.
   5. Extract files from the downloaded zip to a folder specific to each video. You should have a 'SegmentationClass' and a 'SegmentationObject' folder.
   6. Create a folder `data/video/masks/SwiftClasses/` and a folder `data/video/masks/SwiftObject`, in each of them we need an `Annotations` and an `JPEGImages` folders, the first one with the mask image(s), and the second with the original images. There can be several videos in extra subfolders, but both folders should have the same folder hierarchy in them. Finally, create an output folder `data/video/output/SwiftClasses/` and `data/video/output/SwiftObjects/`
   6. Run `python3 XMem/eval.py --output data/video/masks/output/SwiftClasses/ --dataset G --generic_path data/video/masks/SwiftClasses6`, and `python3 XMem/eval.py --output data/video/masks/output/SwiftObjects/ --dataset G --generic_path data/video/masks/SwiftObjects`.
   7. This will create all required label masks in `data/video/output`.
Mask processing:
 - Modify configuration file `config.ini`. Specially set the fps on the video (accounting for the PER paramter in video-toimg) and the frame_step parameter, which indicates every how many frames a frame is processed. You can set also the camera parameters.
 - Run `python3 mask_processing.py --config config.ini data/video/ data/video/masks/output/ HuggingFaceVideoClassification/model/ data/video/mask-process-output/`. You can use --help to see all the parameters that can be set. These override the config file.
 --TODO: Add parameters for camera specs.
 - The output will give the detections from the Activity event detector, triggered when the different objects in the video are moving quickly enough, and the Closeness event detector, triggered when the objects get close enough to each other. 
 - A subfolder will be created for each video, and a txt file will list all the events in the video, with information about them, including a classification of the event.
    - This classification is not too accurate yet, more training data will be required to improve it.
 - Each event will have a subfolder with the cropped video for the event, and its frames in image format sepparately. The folder name will indicate the start and end time of the event, and the classification of the event. 