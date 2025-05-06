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
 - Semi-automatic label using XMem and CVAT:
   1. Add a task for the video to label, and upload at least 1 frame for it, possibly several, one for every e.g. 10-20 minutes of video.
   2. Open the task and use the magic wand on the right, select 'Interact' and enable 'Start with bounding box'.
   3. Draw bounding box around object (swift or nest) and after that use right click to add missing areas from the auto-generated mask, or left click to remove them. Do this for all obhects in the frame. And all updated frames.
   4. Export the frames going to 'Menu', 'Export job dataset'. Select 'Segmentation mask 1.1' as the putput format. A name can be given now to the download file. Click 'Ok' and go to 'Requests'. Once the request finishes it can be downloaded from the 3 dots on its right. Repeat this for all the videos that need to be analyzed.
   5. Extract files from the downloaded zip to a folder specific to each video. You should have a 'SegmentationClass' and a 'SegmentationObject' folder.
   6. Move these around....
   TODOOOOOOOOOO
   6. Run `python3 XMem/eval.py --output data/Video1/output/SwiftObjects/ --dataset G --generic_path data/Video1/SegmentationClass`, and `python3 XMem/eval.py --output data/Video1/output/SwiftObjects/ --dataset G --generic_path data/Video1/SegmentationClass`

`