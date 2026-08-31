package org.thermocell.vision.detection;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Java port of vision/temporal_smoother.py::MaskSmoother.
 *
 * Exponentially-weighted moving average over a binary mask. Keeping a
 * float accumulator (rather than blending the previous binary mask
 * directly) prevents the boundary from "chattering" pixel-by-pixel
 * between frames.
 */
public class MaskSmoother {

    private final double alpha; // weight given to the new frame; lower = smoother/slower
    private Mat acc; // CV_32F accumulator, values in [0, 255]

    public MaskSmoother(double alpha) {
        this.alpha = alpha;
    }

    public void reset() {
        if (acc != null) {
            acc.release();
            acc = null;
        }
    }

    /** binaryMask: CV_8U, values 0 or 255. Returns a new CV_8U mask (caller owns it). */
    public Mat update(Mat binaryMask) {
        Mat maskF = new Mat();
        binaryMask.convertTo(maskF, CvType.CV_32F);

        Size sz = binaryMask.size();
        if (acc == null || !acc.size().equals(sz)) {
            if (acc != null) acc.release();
            acc = maskF; // take ownership, no copy needed
        } else {
            Mat blended = new Mat();
            Core.addWeighted(maskF, alpha, acc, 1.0 - alpha, 0.0, blended);
            maskF.release();
            acc.release();
            acc = blended;
        }

        Mat smoothed = new Mat();
        Imgproc.threshold(acc, smoothed, 127, 255, Imgproc.THRESH_BINARY);
        smoothed.convertTo(smoothed, CvType.CV_8U);
        return smoothed;
    }
}
