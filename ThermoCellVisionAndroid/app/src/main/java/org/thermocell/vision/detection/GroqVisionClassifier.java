package org.thermocell.vision.detection;

import android.graphics.Bitmap;
import android.util.Base64;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * Talks to Groq's OpenAI-compatible vision-model endpoint to classify
 * whether the object currently in frame looks hot, with a confidence
 * score. This is deliberately a SEPARATE, periodic signal on top of the
 * frame-by-frame OpenCV pipeline in HeatRegionDetector - vision-LLM calls
 * are far too slow/expensive to run every frame, so this runs every few
 * seconds in the background while HeatRegionDetector keeps drawing the
 * live outline/placement overlay every frame.
 *
 * IMPORTANT: like heat_detector.py, this is a visual/heuristic judgement
 * from a model looking at a photo - not a real temperature reading.
 *
 * Model name: Groq's vision-capable model lineup changes over time.
 * "llama-3.2-11b-vision-preview" is used as a placeholder default -
 * check https://console.groq.com/docs/models for the current
 * vision-capable model id before shipping and update MODEL below.
 */
public class GroqVisionClassifier {

    private static final String TAG = "GroqVisionClassifier";
    private static final String ENDPOINT = "https://api.groq.com/openai/v1/chat/completions";
    private static final String MODEL = "llama-3.2-11b-vision-preview"; // verify current model id

    private static final String PROMPT =
            "Look at this photo. Decide whether the main object/surface visually " +
            "looks HOT (glowing, flame, molten, visibly heated metal, steam, etc.) " +
            "purely from what you can see - you have no thermal sensor, so base this " +
            "only on visual cues. Respond with ONLY a compact JSON object, no prose, " +
            "no markdown fences, in exactly this shape: " +
            "{\"hot\": true|false, \"confidence\": 0-100, \"label\": \"short object name\"}";

    private final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(8, TimeUnit.SECONDS)
            .readTimeout(15, TimeUnit.SECONDS)
            .build();

    private final String apiKey;
    private final AtomicBoolean requestInFlight = new AtomicBoolean(false);

    public interface Callback {
        void onResult(boolean hot, int confidence, String label);
        void onError(String message);
    }

    public GroqVisionClassifier(String apiKey) {
        this.apiKey = apiKey;
    }

    public boolean isConfigured() {
        return apiKey != null && !apiKey.trim().isEmpty();
    }

    /** Returns false (and does not call back) if a request is already in flight. */
    public boolean classifyAsync(Bitmap frame, Callback callback) {
        if (!isConfigured()) {
            callback.onError("Groq API key not set (see README: local.properties -> groq.apiKey)");
            return false;
        }
        if (!requestInFlight.compareAndSet(false, true)) {
            return false; // still waiting on a previous call - skip this tick
        }

        new Thread(() -> {
            try {
                String base64Jpeg = bitmapToBase64Jpeg(frame, 70);
                JSONObject result = callGroq(base64Jpeg);
                boolean hot = result.optBoolean("hot", false);
                int confidence = result.optInt("confidence", 0);
                String label = result.optString("label", "");
                callback.onResult(hot, confidence, label);
            } catch (Exception e) {
                Log.w(TAG, "Groq classification failed", e);
                callback.onError(e.getMessage() != null ? e.getMessage() : "request failed");
            } finally {
                requestInFlight.set(false);
            }
        }, "groq-vision-classify").start();

        return true;
    }

    private static String bitmapToBase64Jpeg(Bitmap bitmap, int quality) {
        // Downscale before sending - keeps the request small/fast; we only
        // need a coarse visual judgement, not full resolution.
        int targetW = 512;
        int targetH = Math.max(1, Math.round(bitmap.getHeight() * (targetW / (float) bitmap.getWidth())));
        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        scaled.compress(Bitmap.CompressFormat.JPEG, quality, baos);
        if (scaled != bitmap) scaled.recycle();
        return Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP);
    }

    private JSONObject callGroq(String base64Jpeg) throws IOException, org.json.JSONException {
        JSONObject imageUrl = new JSONObject();
        imageUrl.put("url", "data:image/jpeg;base64," + base64Jpeg);

        JSONObject imageContent = new JSONObject();
        imageContent.put("type", "image_url");
        imageContent.put("image_url", imageUrl);

        JSONObject textContent = new JSONObject();
        textContent.put("type", "text");
        textContent.put("text", PROMPT);

        JSONArray contentArray = new JSONArray();
        contentArray.put(textContent);
        contentArray.put(imageContent);

        JSONObject userMessage = new JSONObject();
        userMessage.put("role", "user");
        userMessage.put("content", contentArray);

        JSONArray messages = new JSONArray();
        messages.put(userMessage);

        JSONObject body = new JSONObject();
        body.put("model", MODEL);
        body.put("messages", messages);
        body.put("temperature", 0.0);
        body.put("max_tokens", 100);

        RequestBody requestBody = RequestBody.create(body.toString(), MediaType.get("application/json"));
        Request request = new Request.Builder()
                .url(ENDPOINT)
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("Content-Type", "application/json")
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new IOException("Groq API HTTP " + response.code() + ": " + responseBody);
            }
            JSONObject json = new JSONObject(responseBody);
            String content = json.getJSONArray("choices")
                    .getJSONObject(0)
                    .getJSONObject("message")
                    .getString("content")
                    .trim();

            // Defensively strip markdown fences if the model adds them
            // despite the "no markdown fences" instruction.
            if (content.startsWith("```")) {
                content = content.replaceAll("^```(json)?", "").replaceAll("```$", "").trim();
            }
            return new JSONObject(content);
        }
    }
}
