package org.thermocell.vision;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.drawable.DrawableCompat;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.thermocell.vision.detection.DetectionResult;
import org.thermocell.vision.detection.GroqVisionClassifier;
import org.thermocell.vision.detection.HeatRegionDetector;

import java.nio.ByteBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * ThermoCell Vision - entry point.
 *
 * Java/CameraX/OpenCV port of main.py + ui/camera_view.py. On Android it:
 *   - Requests the CAMERA runtime permission and shows the permission
 *     screen if denied, instead of crashing or showing a blank preview.
 *   - Starts CameraX ImageAnalysis once permission is confirmed and runs
 *     every frame through HeatRegionDetector, displaying the already
 *     composited (overlay-baked-in) frame - same behaviour as the
 *     original camera4kivy analyze_pixels_callback design.
 */
public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CAMERA_PERMISSION = 100;
    // Only analyze every Nth frame at full cost; recomposite() cheaply
    // re-blends cached masks on the frames in between, mirroring the
    // frame-skip design noted in heat_detector.py's recomposite().
    private static final int ANALYZE_EVERY_N_FRAMES = 2;
    // Groq vision calls are slow/expensive relative to a camera frame -
    // run the classification on a timer, independent of the per-frame
    // OpenCV pipeline, rather than on every analyzed frame.
    private static final long GROQ_CLASSIFY_INTERVAL_MS = 3000L;

    static {
        OpenCVLoader.initLocal();
    }

    private final HeatRegionDetector detector = new HeatRegionDetector();
    private GroqVisionClassifier groqClassifier;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private Bitmap latestFrameForGroq;
    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;
    private Camera camera;

    private boolean paused = false;
    private boolean torchOn = false;
    private int lensFacing = CameraSelector.LENS_FACING_BACK;
    private int frameCounter = 0;

    // Views
    private android.view.View screenCamera;
    private android.view.View screenPermission;
    private ImageView cameraOutput;
    private ImageView liveDot;
    private TextView statusText;
    private TextView hotStatusLabel;
    private TextView placementStatusLabel;
    private TextView confidenceLabel;
    private TextView sensitivityValueLabel;
    private TextView opacityValueLabel;
    private Button btnPauseResume;
    private Button btnTorch;
    private TextView aiVerdictLabel;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        screenCamera = findViewById(R.id.screenCamera);
        screenPermission = findViewById(R.id.screenPermission);

        cameraOutput = findViewById(R.id.cameraOutput);
        liveDot = findViewById(R.id.liveDot);
        statusText = findViewById(R.id.statusText);
        hotStatusLabel = findViewById(R.id.hotStatusLabel);
        placementStatusLabel = findViewById(R.id.placementStatusLabel);
        confidenceLabel = findViewById(R.id.confidenceLabel);
        sensitivityValueLabel = findViewById(R.id.sensitivityValueLabel);
        opacityValueLabel = findViewById(R.id.opacityValueLabel);
        aiVerdictLabel = screenCamera.findViewById(R.id.aiVerdictLabel);

        groqClassifier = new GroqVisionClassifier(BuildConfig.GROQ_API_KEY);
        if (!groqClassifier.isConfigured()) {
            aiVerdictLabel.setText("NO API KEY SET");
        }

        setupLegend();
        setupControls();
        startGroqClassificationLoop();

        cameraExecutor = Executors.newSingleThreadExecutor();

        if (hasCameraPermission()) {
            showCameraScreen();
            startCamera();
        } else {
            showPermissionScreen();
            requestCameraPermission();
        }
    }

    // ------------------------------------------------------------------
    // Permission handling
    // ------------------------------------------------------------------
    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (hasCameraPermission()) {
                showCameraScreen();
                startCamera();
            } else {
                showPermissionScreen();
            }
        }
    }

    /** Bound to the 'TRY AGAIN' button on the permission screen. */
    private void retryCameraPermission() {
        if (hasCameraPermission()) {
            showCameraScreen();
            startCamera();
        } else {
            requestCameraPermission();
        }
    }

    private void showCameraScreen() {
        screenCamera.setVisibility(android.view.View.VISIBLE);
        screenPermission.setVisibility(android.view.View.GONE);
    }

    private void showPermissionScreen() {
        screenCamera.setVisibility(android.view.View.GONE);
        screenPermission.setVisibility(android.view.View.VISIBLE);
    }

    // ------------------------------------------------------------------
    // UI wiring (legend + control rail + sliders) - ports the .kv bindings
    // ------------------------------------------------------------------
    private void setupLegend() {
        bindLegendChip(R.id.legendHot, 0xFFFF5259, getString(R.string.legend_hot));
        bindLegendChip(R.id.legendRecommended, 0xFF6BEB8C, getString(R.string.legend_recommended));
        bindLegendChip(R.id.legendSurface, 0xFFD9E0E6, getString(R.string.legend_surface));
    }

    private void bindLegendChip(int includeId, int color, String label) {
        android.view.View row = findViewById(includeId);
        ImageView dot = row.findViewById(R.id.chipDot);
        TextView text = row.findViewById(R.id.chipLabel);
        DrawableCompat.setTint(DrawableCompat.wrap(dot.getDrawable()).mutate(), color);
        text.setText(label);
    }

    private void setupControls() {
        android.view.View permScreen = findViewById(R.id.screenPermission);
        permScreen.findViewById(R.id.btnTryAgain).setOnClickListener(v -> retryCameraPermission());

        android.view.View camScreen = findViewById(R.id.screenCamera);
        camScreen.findViewById(R.id.btnScan).setOnClickListener(v -> requestScan());

        btnPauseResume = camScreen.findViewById(R.id.btnPauseResume);
        btnPauseResume.setOnClickListener(v -> togglePause());

        camScreen.findViewById(R.id.btnReset).setOnClickListener(v -> resetDetection());
        camScreen.findViewById(R.id.btnFlip).setOnClickListener(v -> switchCamera());

        btnTorch = camScreen.findViewById(R.id.btnTorch);
        btnTorch.setOnClickListener(v -> toggleTorch());

        SeekBar sensitivitySlider = camScreen.findViewById(R.id.sensitivitySlider);
        sensitivitySlider.setProgress(50);
        sensitivitySlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                setSensitivity(progress);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Original range is 10-90; SeekBar progress is 0-80, offset by +10.
        SeekBar opacitySlider = camScreen.findViewById(R.id.opacitySlider);
        opacitySlider.setProgress(35);
        opacitySlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                setOverlayOpacity(progress + 10);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
    }

    private void setSensitivity(int percent) {
        detector.setSensitivity(percent);
        sensitivityValueLabel.setText(percent + "%");
    }

    private void setOverlayOpacity(int percent) {
        detector.setOverlayOpacity(percent / 100.0);
        opacityValueLabel.setText(percent + "%");
    }

    private void requestScan() {
        // A manual "kick" - forces the next frame to run full detection
        // even if it would otherwise have been a skipped/recomposited frame.
        frameCounter = 0;
    }

    private void togglePause() {
        paused = !paused;
        btnPauseResume.setText(paused ? R.string.btn_resume : R.string.btn_pause);
        runOnUiThread(() -> statusText.setText(paused ? "PAUSED" : "LIVE"));
    }

    private void resetDetection() {
        detector.reset();
    }

    private void switchCamera() {
        lensFacing = (lensFacing == CameraSelector.LENS_FACING_BACK)
                ? CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK;
        torchOn = false;
        startCamera();
    }

    private void toggleTorch() {
        if (camera == null || camera.getCameraInfo().hasFlashUnit() != true) return;
        torchOn = !torchOn;
        camera.getCameraControl().enableTorch(torchOn);
        btnTorch.setBackgroundColor(torchOn ? getColor(R.color.torch_on_bg) : 0);
    }

    // ------------------------------------------------------------------
    // CameraX setup
    // ------------------------------------------------------------------
    private void startCamera() {
        com.google.common.util.concurrent.ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                bindAnalysisUseCase();
            } catch (Exception e) {
                runOnUiThread(() -> statusText.setText("CAMERA ERROR"));
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindAnalysisUseCase() {
        if (cameraProvider == null) return;
        cameraProvider.unbindAll();

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

        CameraSelector selector = new CameraSelector.Builder().requireLensFacing(lensFacing).build();
        camera = cameraProvider.bindToLifecycle(this, selector, imageAnalysis);

        runOnUiThread(() -> statusText.setText("LIVE"));
    }

    private void analyzeFrame(@NonNull ImageProxy imageProxy) {
        try {
            if (paused) return;

            Mat bgr = imageProxyToBgrMat(imageProxy);
            if (bgr == null) return;

            boolean fullAnalysis = (frameCounter % ANALYZE_EVERY_N_FRAMES) == 0;
            frameCounter++;

            Mat composited;
            DetectionResult result = null;
            if (fullAnalysis) {
                result = detector.process(bgr);
                composited = result.frame;
            } else {
                composited = detector.recomposite(bgr);
            }

            Bitmap bitmap = matToBitmap(composited);
            composited.release();
            bgr.release();

            final DetectionResult finalResult = result;
            runOnUiThread(() -> {
                cameraOutput.setImageBitmap(bitmap);
                if (finalResult != null) updateStatusLabels(finalResult);
            });
            latestFrameForGroq = bitmap; // used by the periodic Groq classification loop
        } finally {
            imageProxy.close();
        }
    }

    // ------------------------------------------------------------------
    // Groq vision classification - periodic, independent of the per-frame
    // OpenCV pipeline. HeatRegionDetector keeps drawing the live
    // outline/placement overlay every frame regardless of whether Groq is
    // configured, slow, or offline; Groq just adds an "AI verdict" on top.
    // ------------------------------------------------------------------
    private final Runnable groqClassifyTick = new Runnable() {
        @Override
        public void run() {
            if (!paused && latestFrameForGroq != null && groqClassifier.isConfigured()) {
                groqClassifier.classifyAsync(latestFrameForGroq, new GroqVisionClassifier.Callback() {
                    @Override
                    public void onResult(boolean hot, int confidence, String label) {
                        runOnUiThread(() -> {
                            String labelPart = (label != null && !label.isEmpty()) ? (label.toUpperCase() + " — ") : "";
                            aiVerdictLabel.setText(labelPart + (hot ? "HOT" : "NOT HOT") + " (" + confidence + "%)");
                            aiVerdictLabel.setTextColor(hot ? 0xFFFF6668 : 0xFF73F394);
                        });
                    }

                    @Override
                    public void onError(String message) {
                        runOnUiThread(() -> {
                            aiVerdictLabel.setText("AI UNAVAILABLE");
                            aiVerdictLabel.setTextColor(0xFF99A3AD);
                        });
                    }
                });
            }
            mainHandler.postDelayed(this, GROQ_CLASSIFY_INTERVAL_MS);
        }
    };

    private void startGroqClassificationLoop() {
        mainHandler.removeCallbacks(groqClassifyTick);
        mainHandler.postDelayed(groqClassifyTick, GROQ_CLASSIFY_INTERVAL_MS);
    }

    private void updateStatusLabels(DetectionResult result) {
        hotStatusLabel.setText(result.hotDetected ? R.string.status_hot_detected : R.string.status_no_hot);
        hotStatusLabel.setTextColor(result.hotDetected ? 0xFFFF6668 : 0xFF99A3AD);

        placementStatusLabel.setText(result.placementRecommended ? R.string.status_placement_yes : R.string.status_placement_no);
        placementStatusLabel.setTextColor(result.placementRecommended ? 0xFF73F394 : 0xFF99A3AD);

        int confidencePercent = (int) Math.round(result.confidence);
        confidenceLabel.setText(String.format("CONFIDENCE: %02d%%", confidencePercent));
    }

    // ------------------------------------------------------------------
    // Frame conversion helpers (YUV_420_888 ImageProxy -> OpenCV BGR Mat)
    // ------------------------------------------------------------------
    private Mat imageProxyToBgrMat(ImageProxy imageProxy) {
        if (imageProxy.getFormat() != ImageFormat.YUV_420_888) return null;

        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();

        Mat yuvMat = new Mat(height + height / 2, width, CvType.CV_8UC1);
        yuvMat.put(0, 0, nv21);

        Mat bgr = new Mat();
        Imgproc.cvtColor(yuvMat, bgr, Imgproc.COLOR_YUV2BGR_NV21, 3);
        yuvMat.release();

        int rotation = imageProxy.getImageInfo().getRotationDegrees();
        if (rotation != 0) {
            Mat rotated = new Mat();
            if (rotation == 90) Core_rotate(bgr, rotated, 0);
            else if (rotation == 180) Core_rotate(bgr, rotated, 1);
            else if (rotation == 270) Core_rotate(bgr, rotated, 2);
            else rotated = bgr;
            if (rotated != bgr) bgr.release();
            return rotated;
        }
        return bgr;
    }

    private static void Core_rotate(Mat src, Mat dst, int rotateCode) {
        // 0 = ROTATE_90_CLOCKWISE, 1 = ROTATE_180, 2 = ROTATE_90_COUNTERCLOCKWISE
        org.opencv.core.Core.rotate(src, dst, rotateCode);
    }

    private Bitmap matToBitmap(Mat bgr) {
        Mat rgba = new Mat();
        Imgproc.cvtColor(bgr, rgba, Imgproc.COLOR_BGR2RGBA);
        Bitmap bitmap = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgba, bitmap);
        rgba.release();
        return bitmap;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mainHandler.removeCallbacks(groqClassifyTick);
        try {
            if (cameraProvider != null) cameraProvider.unbindAll();
        } catch (Exception ignored) {}
        cameraExecutor.shutdown();
    }
}
