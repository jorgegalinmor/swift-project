import os
import shutil
import subprocess
import argparse
from pathlib import Path
import stat


def process_video(video_path, output_path):
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    # Copy video to output path
    copied_video_path = output_path / video_path.name

    if copied_video_path.exists():
        os.chmod(copied_video_path, stat.S_IWUSR | stat.S_IRUSR)  # Give write and read permissions
        subprocess.run(["rm", "-f", str(copied_video_path)], check=True)

    try:
        shutil.copy2(video_path, copied_video_path)
    except Exception as e:
        print(f"Error during video-toimg execution: {e}")
        print(f"stdout: {e}")

    # Run video-toimg to extract frames
    try:
        subprocess.run(["video-toimg", str(copied_video_path)], shell=True, check=True)
    except Exception as e:
        print(f"Error during video-toimg execution: {e}")

    # Remove copied video
    copied_video_path.unlink()

    # The extracted images folder will have the same name as the video (without extension)
    video_name = video_path.stem
    images_folder = output_path / video_name

    if not images_folder.exists():
        raise RuntimeError(f"Expected images folder not found: {images_folder}")
    # Create dataset structure
    for category in ["Classes", "Objects"]:
        category_path = output_path / category
        annotations_folder = category_path / "Annotations" / video_name
        jpeg_images_link = category_path / "JPEGImages"

        annotations_folder.mkdir(parents=True, exist_ok=True)

        # Create or update symlink for JPEGImages
        if jpeg_images_link.exists() or jpeg_images_link.is_symlink():
            jpeg_images_link.unlink()
        jpeg_images_link.symlink_to(images_folder, target_is_directory=True)

    print(f"Processing complete. Images extracted and dataset structure created at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert a video to images and organize into dataset folders.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("output_path", type=str, help="Path to the output directory")

    args = parser.parse_args()

    try:
        process_video(args.video_path, args.output_path)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
