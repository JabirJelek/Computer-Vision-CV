import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def extract_frames(video_path, output_dir, fps=1, format="jpg", quality=2, prefix="frame"):
    """
    Extract frames from a video using FFmpeg

    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory where frames will be saved
        fps (int): Frames per second to extract (default: 1)
        format (str): Image format (jpg, png)
        quality (int): JPEG quality level (1-31 where 2 is highest)
        prefix (str): Prefix for output filenames
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate a timestamp for this extraction session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
    ]
    
    # Add quality parameter for JPEG format
    if format == "jpg":
        cmd.extend(["-q:v", str(quality)])
    
    # Add output file and logging options
    cmd.extend([
        f"{output_dir}/{prefix}_{timestamp}%04d.{format}",
        "-loglevel", "error",
        "-stats"
    ])

    try:
        print(f"Extracting frames from {video_path}...")
        # Let FFmpeg output directly to console for progress display
        result = subprocess.run(cmd, check=True)
        print("Frame extraction completed successfully!")

        # Count extracted frames
        frame_count = len(list(Path(output_dir).glob(f"{prefix}_{timestamp}*.{format}")))
        print(f"Extracted {frame_count} frames to {output_dir}")

    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output directory (default: frames_<filename>)")
    parser.add_argument("-f", "--fps", type=float, default=10, help="Frames per second to extract (default: 1)")
    parser.add_argument("-fmt", "--format", default="png", choices=["jpg", "png"], help="Output image format (default: jpg)")
    parser.add_argument("-q", "--quality", type=int, default=2, help="JPEG quality (1-31, 2=best, default: 2)")
    parser.add_argument("-p", "--prefix", default="frame", help="Filename prefix (default: frame)")

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.isfile(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        return

    # Generate output directory with filename if not provided
    if args.output is None:
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"frames_{filename}"

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        format=args.format,
        quality=args.quality,
        prefix=args.prefix
    )

if __name__ == "__main__":
    main()