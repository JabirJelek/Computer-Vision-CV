from ultralytics import YOLO

# Load model
model = YOLO(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\v2-apd.torchscript")

# Use built-in tracking
results = model.track(source=r"D:\RaihanFarid\Dokumen\Object Detection\1.1_APD\output_video.mp4", 
                     #tracker="bytetrack.yaml",  # or "botsort.yaml"
                     conf=0.5,
                     show=True,
                     stream=True)
                     

# # Or for single image with tracking persistence
# results = model.track(source="video.mp4", 
#                      persist=True,  # Maintain tracking between frames
#                      conf=0.5)