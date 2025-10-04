### This is documentation about ptCamera-Complex.py

If you want to change / edit / use the code in future, here are some documentation to help you make that changes. Please make a copy of the file, and do further from that. Dont change the main file. Make your own name based on your own project version. Thanks~

#### Comprehensive Documentation: YOLOv8 Real-Time Object Detection System
Overview
This system provides a modular, configurable real-time object detection pipeline using YOLOv8, OpenCV, and PyGame. It's designed for webcam-based object detection with audio alerts, visual feedback, and comprehensive counting functionality.

<ol>1. Core Functionality & System Architecture</ol>
- Purpose: A modular, real-time object detection system built with YOLOv8.
- Key Components:

    - AppConfig: Centralizes all settings (model, camera, detection, alerts) for easy management.

    - ObjectDetector: The main orchestrator that runs the detection pipeline.

    - AlertManager: Handles audio alerts with class-specific cooldowns using non-blocking threads.

    - DetectionVisualizer: Manages all on-screen graphics (boxes, labels, info panels).

    - Smart Counting: Tracks both real-time detections per frame and cumulative counts over a configurable time window.

<ol>2. Configuration & Setup</ol>
- Camera Source: Flexible support for both local webcams (via index 0, 1, 2...) and network streams (RTSP).

- Model: Uses a custom-trained YOLOv8 model in .torchscript format.

- Basic Settings: Easily configure resolution, confidence threshold, and model path.

- Class-Specific Settings:

    - class_color_map: Define unique colors for bounding boxes per class.

    - class_cooldowns: Set different alert cool-down timers (in seconds) for each class.

<ol>3. Key Usage Examples</ol>
- Basic Use: Instantiate with default settings and run.

```python
detector = ObjectDetector(AppConfig())
detector.run()
```

- Custom Use: Tailor the configuration for specific needs.

``` python
config = AppConfig(
    model_path=Path("custom_model.torchscript"),
    class_color_map={0: (255, 255, 0)}, # Yellow for class 0
    class_cooldowns={0: 15} # 15-sec cooldown for class 0
)
detector = ObjectDetector(config)
```

<ol>4. Extensibility & Customization Points</ol>
- Adding New Classes: Update the model, then add entries to class_color_map and class_cooldowns.

- Custom Alert Logic: Override should_trigger_alert() to implement:

    - Count-based alerts (e.g., alert after N detections).

    - Position-based alerts (e.g., only in certain areas).

    - Combination rules (e.g., alert when two classes appear together).

- Additional Visualizations: Extend DetectionVisualizer to draw heatmaps, historical tracking paths, or custom info panels.

- RTSP Display: Uncomment the resize line to adjust the window size for RTSP streams.

<ol>5. Performance & Troubleshooting</ol>
- Performance:

    - FPS vs. Resolution: Higher resolutions (e.g., 1280x720) reduce frame rate. 640x480 is optimal for real-time use.

    - Memory: Model is loaded once at start; frame processing is memory-efficient.

- Troubleshooting:

    - Webcam not found: Try different camera_index values (0, 1, 2).

    - Low FPS: Reduce resolution or lower the confidence_threshold.

    - Model error: Verify the model file path and format.

    - Debug: Add print statements to monitor frame processing time and detection counts.

<ol>6. Future Expansion Capabilities</ol>
- Immediate Additions:

    - Multiple camera support.

    - Record video clips upon detection.

    - Stream output via RTSP/HTTP.

- Advanced Features:

    - Object tracking across frames.

    - Cross-line detection (trigger alert when an object crosses a virtual line).

    - REST API for remote control and monitoring.

    - Cloud integration for alerts and metrics.
