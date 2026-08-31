package org.thermocell.vision.detection;

import org.opencv.core.Mat;

/** Plain container for one frame's detection output. Port of heat_detector.py::DetectionResult. */
public class DetectionResult {
    public final Mat frame; // composited BGR frame with overlays baked in (caller owns / must release)
    public final boolean hotDetected;
    public final boolean placementRecommended;
    public final double confidence; // 0-100
    public final boolean surfaceFound;

    public DetectionResult(Mat frame, boolean hotDetected, boolean placementRecommended,
                            double confidence, boolean surfaceFound) {
        this.frame = frame;
        this.hotDetected = hotDetected;
        this.placementRecommended = placementRecommended;
        this.confidence = confidence;
        this.surfaceFound = surfaceFound;
    }
}
