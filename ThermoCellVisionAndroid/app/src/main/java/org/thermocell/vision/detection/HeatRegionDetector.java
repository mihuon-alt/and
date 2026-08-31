package org.thermocell.vision.detection;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.List;

/**
 * The computer-vision core of ThermoCell Vision — Java/OpenCV port of
 * vision/heat_detector.py.
 *
 * HONESTY NOTE (kept from the original): this class never measures real
 * temperature. A phone RGB camera has no thermal sensor. Everything here
 * is a *visual* heuristic: it looks for regions that *look* hot (warm
 * hue, high saturation/brightness -> reds, oranges, glowing yellows)
 * using classic color-based computer vision. The public-facing feature
 * name is deliberately "Visual Heat Estimation", never "temperature
 * measurement".
 *
 * The 12-stage pipeline mirrors heat_detector.py::process() exactly.
 */
public class HeatRegionDetector {

    // Colors are BGR (OpenCV convention) since we composite in BGR space.
    private static final Scalar COLOR_HOT = new Scalar(60, 60, 255);
    private static final Scalar COLOR_PLACEMENT = new Scalar(90, 230, 110);
    private static final Scalar COLOR_SURFACE_EDGE = new Scalar(230, 230, 230);

    private final int processWidth;

    private double sensitivity = 50;      // 0-100, UI-controlled
    private double overlayOpacity = 0.45; // 0-1, UI-controlled

    private final MaskSmoother hotMaskSmoother = new MaskSmoother(0.4);
    private final MaskSmoother placementMaskSmoother = new MaskSmoother(0.3);
    private final ValueSmoother confidenceSmoother = new ValueSmoother(0.25);

    private final Mat kOpen = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
    private final Mat kClose = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7, 7));
    private final Mat kHotDilate = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(9, 9));

    private Mat lastHotMaskFull;
    private Mat lastPlacementMaskFull;
    private MatOfPoint lastSurfaceContourFull;

    public HeatRegionDetector() {
        this(320);
    }

    public HeatRegionDetector(int processWidth) {
        this.processWidth = processWidth;
    }

    // ------------------------------------------------------------------
    // Public controls (wired to the sliders / buttons in the UI)
    // ------------------------------------------------------------------
    public void setSensitivity(double value) {
        sensitivity = clamp(value, 0, 100);
    }

    public void setOverlayOpacity(double value) {
        overlayOpacity = clamp(value, 0.0, 1.0);
    }

    /** Called by the "Reset detection" button - clears temporal state so old smoothed shapes don't linger. */
    public void reset() {
        hotMaskSmoother.reset();
        placementMaskSmoother.reset();
        confidenceSmoother.reset();
    }

    // ------------------------------------------------------------------
    // Main entry point
    // ------------------------------------------------------------------
    /** frameBgr: full-resolution analysis frame, CV_8UC3 (BGR). Caller owns frameBgr. */
    public DetectionResult process(Mat frameBgr) {
        int w = frameBgr.cols();
        int h = frameBgr.rows();

        // --- 1. Resize down for the expensive per-pixel math -----------
        double scale = processWidth / (double) w;
        int procW = processWidth;
        int procH = Math.max(1, (int) Math.round(h * scale));
        Mat small = new Mat();
        Imgproc.resize(frameBgr, small, new Size(procW, procH), 0, 0, Imgproc.INTER_AREA);

        // --- 2 & 3. HSV conversion + thresholding -----------------------
        Mat hotMaskSmall = thresholdHotRegions(small);

        // --- 4. Morphological opening then closing ----------------------
        Imgproc.morphologyEx(hotMaskSmall, hotMaskSmall, Imgproc.MORPH_OPEN, kOpen);
        Imgproc.morphologyEx(hotMaskSmall, hotMaskSmall, Imgproc.MORPH_CLOSE, kClose);

        // --- 5. Connected-component filtering (drop noise specks) -------
        double minHotArea = 0.0015 * procW * procH;
        hotMaskSmall = filterSmallComponents(hotMaskSmall, minHotArea);

        // --- 6. Contour detection on the cleaned heat mask ---------------
        List<MatOfPoint> hotContoursSmall = new ArrayList<>();
        Imgproc.findContours(hotMaskSmall.clone(), hotContoursSmall, new Mat(),
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        int hotContourCount = 0;
        for (MatOfPoint c : hotContoursSmall) {
            if (Imgproc.contourArea(c) >= minHotArea) hotContourCount++;
        }
        boolean hotDetected = hotContourCount > 0;

        // --- 7. Surface detection: largest contour + polygon approx -----
        SurfaceResult surfaceResult = detectSurface(small);
        Mat surfaceMaskSmall = surfaceResult.mask;
        boolean surfaceFound = surfaceResult.found;

        // --- 8. Erode/inset the surface mask (edge-safe region) ---------
        int insetPx = Math.max(3, (int) Math.round(Math.min(procW, procH) * 0.05));
        Mat kInset = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(insetPx, insetPx));
        Mat surfaceErodedSmall = new Mat();
        Imgproc.erode(surfaceMaskSmall, surfaceErodedSmall, kInset);

        // --- 9. Remove hot regions (dilated for a safety margin) --------
        Mat hotDilatedSmall = new Mat();
        Imgproc.dilate(hotMaskSmall, hotDilatedSmall, kHotDilate);
        Mat hotDilatedInv = new Mat();
        Core.bitwise_not(hotDilatedSmall, hotDilatedInv);
        Mat usableSmall = new Mat();
        Core.bitwise_and(surfaceErodedSmall, hotDilatedInv, usableSmall);

        // --- 10. Largest remaining usable region -> placement zone ------
        double minPlaceArea = 0.01 * procW * procH;
        ComponentResult placementResult = largestComponentMask(usableSmall, minPlaceArea);
        Mat placementMaskSmall = placementResult.mask; // may be null
        double placementArea = placementResult.area;

        double surfaceArea = Math.max(Core.countNonZero(surfaceMaskSmall), 1);
        double confidenceRaw;
        if (placementMaskSmall != null) {
            confidenceRaw = 100.0 * (placementArea / surfaceArea);
            confidenceRaw = clamp(confidenceRaw, 5.0, 98.0);
        } else {
            confidenceRaw = 0.0;
        }

        // --- 11. Temporal smoothing --------------------------------------
        Mat hotMaskSmoothed = hotMaskSmoother.update(hotMaskSmall);
        Mat placementMaskSmoothed;
        if (placementMaskSmall != null) {
            placementMaskSmoothed = placementMaskSmoother.update(placementMaskSmall);
        } else {
            Mat blank = Mat.zeros(procH, procW, CvType.CV_8U);
            placementMaskSmoothed = placementMaskSmoother.update(blank);
            blank.release();
        }
        double confidence = confidenceSmoother.update(confidenceRaw);
        boolean placementRecommended = confidenceRaw > 0
                && Core.countNonZero(placementMaskSmoothed) > minPlaceArea * 0.5;

        // --- Scale masks back up to display resolution -------------------
        Mat hotMaskFull = new Mat();
        Imgproc.resize(hotMaskSmoothed, hotMaskFull, new Size(w, h), 0, 0, Imgproc.INTER_NEAREST);
        Mat placementMaskFull = new Mat();
        Imgproc.resize(placementMaskSmoothed, placementMaskFull, new Size(w, h), 0, 0, Imgproc.INTER_NEAREST);

        MatOfPoint surfaceContourFull = null;
        if (surfaceFound) {
            double invScale = 1.0 / scale;
            MatOfPoint contourSmall = maskToContour(surfaceMaskSmall);
            surfaceContourFull = scaleContour(contourSmall, invScale);
        }

        // --- 12. Composite translucent overlays onto the ORIGINAL frame -
        Mat composited = compositeOverlays(frameBgr, hotMaskFull, surfaceContourFull, placementMaskFull);

        // Cache for recomposite() on frame-skipped calls.
        if (lastHotMaskFull != null) lastHotMaskFull.release();
        if (lastPlacementMaskFull != null) lastPlacementMaskFull.release();
        lastHotMaskFull = hotMaskFull;
        lastPlacementMaskFull = placementMaskFull;
        lastSurfaceContourFull = surfaceContourFull;

        // release intermediates
        small.release();
        hotMaskSmall.release();
        surfaceMaskSmall.release();
        surfaceErodedSmall.release();
        hotDilatedSmall.release();
        hotDilatedInv.release();
        usableSmall.release();
        hotMaskSmoothed.release();
        placementMaskSmoothed.release();

        return new DetectionResult(composited, hotDetected, placementRecommended, confidence, surfaceFound);
    }

    /**
     * Cheap path for frame-skipped calls: re-blends the *last* computed masks
     * onto a freshly captured background frame, without repeating any
     * thresholding/morphology/contour work.
     */
    public Mat recomposite(Mat frameBgr) {
        if (lastHotMaskFull == null && lastPlacementMaskFull == null) {
            return frameBgr.clone();
        }
        int w = frameBgr.cols();
        int h = frameBgr.rows();
        Mat hot = lastHotMaskFull;
        Mat placement = lastPlacementMaskFull;
        if (hot != null && (hot.cols() != w || hot.rows() != h)) {
            Mat resized = new Mat();
            Imgproc.resize(hot, resized, new Size(w, h), 0, 0, Imgproc.INTER_NEAREST);
            hot = resized;
        }
        if (placement != null && (placement.cols() != w || placement.rows() != h)) {
            Mat resized = new Mat();
            Imgproc.resize(placement, resized, new Size(w, h), 0, 0, Imgproc.INTER_NEAREST);
            placement = resized;
        }
        return compositeOverlays(frameBgr, hot, lastSurfaceContourFull, placement);
    }

    // ------------------------------------------------------------------
    // Stage implementations
    // ------------------------------------------------------------------

    /**
     * HSV-style thresholding for visually "hot-looking" color: warm hues
     * (red -> orange -> yellow) with reasonably high saturation and
     * brightness, plus a "glowing" near-white/yellow band. Sensitivity
     * widens/narrows the thresholds. This is explicitly NOT a temperature
     * measurement.
     */
    private Mat thresholdHotRegions(Mat bgrSmall) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrSmall, hsv, Imgproc.COLOR_BGR2HSV);

        double sens = sensitivity / 100.0; // 0..1

        int satMin = (int) Math.round(150 - sens * 95);
        int valMin = (int) Math.round(150 - sens * 75);
        satMin = (int) clamp(satMin, 40, 200);
        valMin = (int) clamp(valMin, 60, 200);

        int hueUpper1 = (int) clamp(Math.round(22 + sens * 20), 20, 45);
        Mat mask1 = new Mat();
        Core.inRange(hsv, new Scalar(0, satMin, valMin), new Scalar(hueUpper1, 255, 255), mask1);

        int hueLower2 = (int) clamp(Math.round(168 - sens * 8), 155, 179);
        Mat mask2 = new Mat();
        Core.inRange(hsv, new Scalar(hueLower2, satMin, valMin), new Scalar(179, 255, 255), mask2);

        int glowValMin = (int) clamp(Math.round(235 - sens * 40), 190, 250);
        Mat mask3 = new Mat();
        Core.inRange(hsv, new Scalar(0, 0, glowValMin), new Scalar(45, 255, 255), mask3);

        Mat combined = new Mat();
        Core.bitwise_or(mask1, mask2, combined);
        Core.bitwise_or(combined, mask3, combined);

        hsv.release();
        mask1.release();
        mask2.release();
        mask3.release();
        return combined;
    }

    private static class SurfaceResult {
        Mat mask;
        boolean found;
    }

    /**
     * Finds the main usable surface/object using edge detection -> largest
     * relevant contour -> polygon approximation. Falls back to "whole
     * frame minus a safety margin" when nothing confident is found.
     */
    private SurfaceResult detectSurface(Mat bgrSmall) {
        int w = bgrSmall.cols();
        int h = bgrSmall.rows();
        Mat gray = new Mat();
        Imgproc.cvtColor(bgrSmall, gray, Imgproc.COLOR_BGR2GRAY);
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new Size(5, 5), 0);
        Mat edges = new Mat();
        Imgproc.Canny(blurred, edges, 40, 120);
        Imgproc.dilate(edges, edges, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3)), new Point(-1, -1), 2);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(edges, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double minSurfaceArea = 0.15 * w * h;
        MatOfPoint candidate = null;
        double candidateArea = 0;
        for (MatOfPoint c : contours) {
            double area = Imgproc.contourArea(c);
            if (area > candidateArea) {
                candidate = c;
                candidateArea = area;
            }
        }

        Mat mask = Mat.zeros(h, w, CvType.CV_8U);
        SurfaceResult result = new SurfaceResult();

        if (candidate != null && candidateArea >= minSurfaceArea) {
            MatOfPoint2f candidate2f = new MatOfPoint2f(candidate.toArray());
            double perimeter = Imgproc.arcLength(candidate2f, true);
            MatOfPoint2f approx2f = new MatOfPoint2f();
            Imgproc.approxPolyDP(candidate2f, approx2f, 0.02 * perimeter, true);
            MatOfPoint approx = new MatOfPoint(approx2f.toArray());
            List<MatOfPoint> fillList = new ArrayList<>();
            fillList.add(approx);
            Imgproc.fillPoly(mask, fillList, new Scalar(255));
            result.mask = mask;
            result.found = true;
            candidate2f.release();
            approx2f.release();
        } else {
            // Fallback: treat the framed view (minus a margin) as the surface.
            int marginX = (int) (w * 0.04);
            int marginY = (int) (h * 0.04);
            Imgproc.rectangle(mask, new Point(marginX, marginY), new Point(w - marginX, h - marginY), new Scalar(255), -1);
            result.mask = mask;
            result.found = false;
        }

        gray.release();
        blurred.release();
        edges.release();
        return result;
    }

    /**
     * Connected-component filtering: drops any blob smaller than minArea
     * so isolated noise pixels can't create hundreds of tiny detections.
     */
    private static Mat filterSmallComponents(Mat binaryMask, double minArea) {
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int numLabels = Imgproc.connectedComponentsWithStats(binaryMask, labels, stats, centroids, 8);

        Mat cleaned = Mat.zeros(binaryMask.size(), binaryMask.type());
        for (int label = 1; label < numLabels; label++) {
            double area = stats.get(label, Imgproc.CC_STAT_AREA)[0];
            if (area >= minArea) {
                Mat labelMask = new Mat();
                Core.compare(labels, new Scalar(label), labelMask, Core.CMP_EQ);
                labelMask.convertTo(labelMask, CvType.CV_8U);
                Core.bitwise_or(cleaned, labelMask, cleaned);
                labelMask.release();
            }
        }
        labels.release();
        stats.release();
        centroids.release();
        return cleaned;
    }

    private static class ComponentResult {
        Mat mask; // may be null
        double area;
    }

    /**
     * Returns the mask of just the largest connected component (>= minArea),
     * or a null mask if none qualifies. Using connected-component labels
     * (rather than RETR_EXTERNAL contours) means holes inside the region -
     * e.g. an excluded hot-region "island" - are preserved correctly.
     */
    private static ComponentResult largestComponentMask(Mat binaryMask, double minArea) {
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int numLabels = Imgproc.connectedComponentsWithStats(binaryMask, labels, stats, centroids, 8);

        int bestLabel = -1;
        double bestArea = 0;
        for (int label = 1; label < numLabels; label++) {
            double area = stats.get(label, Imgproc.CC_STAT_AREA)[0];
            if (area >= minArea && area > bestArea) {
                bestLabel = label;
                bestArea = area;
            }
        }

        ComponentResult result = new ComponentResult();
        if (bestLabel == -1) {
            labels.release();
            stats.release();
            centroids.release();
            result.mask = null;
            result.area = 0;
            return result;
        }

        Mat mask = new Mat();
        Core.compare(labels, new Scalar(bestLabel), mask, Core.CMP_EQ);
        mask.convertTo(mask, CvType.CV_8U);
        labels.release();
        stats.release();
        centroids.release();

        result.mask = mask;
        result.area = bestArea;
        return result;
    }

    private static MatOfPoint maskToContour(Mat binaryMask) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(binaryMask.clone(), contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        if (contours.isEmpty()) return null;
        MatOfPoint best = null;
        double bestArea = -1;
        for (MatOfPoint c : contours) {
            double area = Imgproc.contourArea(c);
            if (area > bestArea) {
                bestArea = area;
                best = c;
            }
        }
        return best;
    }

    private static MatOfPoint scaleContour(MatOfPoint contour, double factor) {
        if (contour == null) return null;
        Point[] pts = contour.toArray();
        Point[] scaled = new Point[pts.length];
        for (int i = 0; i < pts.length; i++) {
            scaled[i] = new Point(pts[i].x * factor, pts[i].y * factor);
        }
        MatOfPoint out = new MatOfPoint();
        out.fromArray(scaled);
        return out;
    }

    // ------------------------------------------------------------------
    // Overlay compositing (kept semi-transparent - original image always
    // stays visible underneath, per spec).
    // ------------------------------------------------------------------
    private Mat compositeOverlays(Mat baseBgr, Mat hotMaskFull, MatOfPoint surfaceContour, Mat placementMaskFull) {
        Mat overlay = baseBgr.clone();

        boolean hotPresent = hotMaskFull != null && Core.countNonZero(hotMaskFull) > 0;
        if (hotPresent) {
            overlay.setTo(COLOR_HOT, hotMaskFull);
        }

        boolean placementPresent = placementMaskFull != null && Core.countNonZero(placementMaskFull) > 0;
        if (placementPresent) {
            overlay.setTo(COLOR_PLACEMENT, placementMaskFull);
        }

        Mat blended = new Mat();
        Core.addWeighted(overlay, overlayOpacity, baseBgr, 1 - overlayOpacity, 0, blended);
        overlay.release();

        if (surfaceContour != null) {
            List<MatOfPoint> contours = new ArrayList<>();
            contours.add(surfaceContour);
            Imgproc.polylines(blended, contours, true, COLOR_SURFACE_EDGE, 2, Imgproc.LINE_AA);
        }

        if (placementPresent) {
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(placementMaskFull.clone(), contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
            if (!contours.isEmpty()) {
                Imgproc.polylines(blended, contours, true, COLOR_PLACEMENT, 2, Imgproc.LINE_AA);
            }
            hierarchy.release();
        }

        return blended;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
