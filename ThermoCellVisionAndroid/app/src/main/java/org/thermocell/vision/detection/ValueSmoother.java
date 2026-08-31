package org.thermocell.vision.detection;

/**
 * Java port of vision/temporal_smoother.py::ValueSmoother.
 * Scalar exponential moving average, used for the confidence percent.
 */
public class ValueSmoother {

    private final double alpha;
    private Double value = null;

    public ValueSmoother(double alpha) {
        this.alpha = alpha;
    }

    public void reset() {
        value = null;
    }

    public double update(double newValue) {
        if (value == null) {
            value = newValue;
        } else {
            value = (alpha * newValue) + (1.0 - alpha) * value;
        }
        return value;
    }
}
