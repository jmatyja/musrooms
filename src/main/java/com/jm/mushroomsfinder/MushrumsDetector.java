package com.jm.mushroomsfinder;


import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.RectF;
import android.os.Trace;
import android.util.TimingLogger;

import com.jm.mushroomsfinder.env.ImageUtils;
import com.jm.mushroomsfinder.env.Logger;
import com.jm.mushroomsfinder.env.SplitTimer;


import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;


/** An object detector that uses TF and a YOLO model to detect objects. */
public class MushrumsDetector implements Classifier {



    private static final Logger LOGGER = new Logger();
    private MushrumsDetector mushrumsDetector = null;
    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 5;

    private static final int NUM_CLASSES = 1;

    private static final int NUM_BOXES_PER_BLOCK = 5;

    // TODO(andrewharp): allow loading anchors and classes
    // from files.
    private static final double[] ANCHORS = {
            1.08, 1.19,
            3.42, 4.41,
            6.63, 11.38,
            9.42, 5.11,
            16.62, 10.52
    };

    private static final String[] LABELS = {
            "boletus"
    };

    // Config values.
    private String inputName;
    private int inputSize;

    // Pre-allocated buffers.
    private int[] intValues;
    private float[] floatValues;
    private String[] outputNames;

    private int blockSize;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /** Initializes a native TensorFlow session for classifying images. */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final int inputSize,
            final String inputName,
            final String outputName,
            final int blockSize) {
        MushrumsDetector d = new MushrumsDetector();
        d.inputName = inputName;
        d.inputSize = inputSize;

        // Pre-allocate buffers.
        d.outputNames = outputName.split(",");
        d.intValues = new int[inputSize * inputSize];
        d.floatValues = new float[inputSize * inputSize * 3];
        d.blockSize = blockSize;

        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        return d;
    }

    private MushrumsDetector() {}



    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        LOGGER.i("START");
        if(null == bitmap)
          return null;

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        byte[] byteValues = new byte[inputSize * inputSize * 3];
        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }
        inferenceInterface.feed(inputName, byteValues, 1,  inputSize,  inputSize, 3);
        TimingLogger timings = new TimingLogger("m", "methodA");
        timings.addSplit("run");
        inferenceInterface.run(outputNames, false);
        timings.addSplit("stop");
        timings.dumpToLog();
        float[] outputLocations = new float[MAX_RESULTS * 400];
        float[] outputScores = new float[MAX_RESULTS*100];
        inferenceInterface.fetch(outputNames[0], outputLocations);
        inferenceInterface.fetch(outputNames[1], outputScores);
        // Find the best detections.
        final PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition lhs, final Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < outputScores.length; ++i) {
            final RectF detection =
                    new RectF(
                            outputLocations[4 * i + 1] * inputSize,
                            outputLocations[4 * i] * inputSize,
                            outputLocations[4 * i + 3] * inputSize,
                            outputLocations[4 * i + 2] * inputSize);
            pq.add(
                    new Recognition("" + i, "boletus", outputScores[i], detection));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }


    private int[] getIntPixels(Bitmap bitmap) {
        int[] intValues = new int[bitmap.getWidth() * bitmap.getHeight()];
        int[] intResults = new int[bitmap.getWidth() * bitmap.getHeight()*3];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            intResults[i * 3 + 2] = Color.red(val);
            intResults[i * 3 + 1] = Color.green(val);
            intResults[i * 3] = Color.blue(val);
        }
        return intResults;
    }

    private byte[] toByteArray(int value) {
        return new byte[] {
                (byte)(value >> 24),
                (byte)(value >> 16),
                (byte)(value >> 8),
                (byte)value};
    }

    @Override
    public void enableStatLogging(final boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
