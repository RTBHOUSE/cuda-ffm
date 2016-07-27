package com.rtbhouse.model.natives;

import static com.rtbhouse.model.natives.FactorizationMachineNativeOps.FACTOR_SIZE;
import static org.junit.Assert.assertEquals;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import org.junit.Test;

public class FactorizationMachinesNativeOpsTest {

    @Test
    public void testSmallModel() {
        int numFeatures = 1;
        int numFields = 2;
        int size = FACTOR_SIZE * numFeatures * numFields;

        FloatBuffer weights = allocateDirectFloatBuffer(size);
        weights.put(new float[]{ 1, 0.5f, 0.25f, 0.125f, 2, 4, 8, 16 });

        IntBuffer feats = allocateDirectIntBufferOf(0, 0);
        assertEquals(0.98201379003790850, FactorizationMachineNativeOps.ffmPredict(weights, feats), 1e-07); // sigmoid(0.5 * 8)
    }

    @Test
    public void testLargerModelOddNumberOfFields() {
        int numFeatures = 9;
        int numFields = 7;
        int size = FACTOR_SIZE * numFeatures * numFields;

        FloatBuffer weights = allocateDirectFloatBuffer(size);

        for (int i = 0; i < size; i++) {
            weights.put((float) i / size - 0.5f);
        }

        IntBuffer feats1 = allocateDirectIntBufferOf(0, 3, 4, 5, 6, 1, 2);
        IntBuffer feats2 = allocateDirectIntBufferOf(4, 3, 3, 5, 7, 1, 6);
        IntBuffer feats3 = allocateDirectIntBufferOf(1, 2, 0, 5, 8, 3, 4);

        assertEquals(0.5192726966, FactorizationMachineNativeOps.ffmPredict(weights, feats1), 1e-07);
        assertEquals(0.4872870062, FactorizationMachineNativeOps.ffmPredict(weights, feats2), 1e-07);
        assertEquals(0.5098327584, FactorizationMachineNativeOps.ffmPredict(weights, feats3), 1e-07);
    }

    @Test
    public void testLargerModelEvenNumberOfFields() {
        int numFeatures = 9;
        int numFields = 8;
        int size = FACTOR_SIZE * numFeatures * numFields;

        FloatBuffer weights = allocateDirectFloatBuffer(size);

        for (int i = 0; i < size; i++) {
            weights.put((float) i / size - 0.5f);
        }

        IntBuffer feats1 = allocateDirectIntBufferOf(0, 3, 4, 5, 6, 1, 2, 8);
        IntBuffer feats2 = allocateDirectIntBufferOf(4, 3, 3, 5, 7, 1, 6, 8);
        IntBuffer feats3 = allocateDirectIntBufferOf(1, 2, 0, 5, 8, 3, 4, 8);

        assertEquals(0.4966875725733104, FactorizationMachineNativeOps.ffmPredict(weights, feats1), 1e-07);
        assertEquals(0.5126955971881075, FactorizationMachineNativeOps.ffmPredict(weights, feats2), 1e-07);
        assertEquals(0.4998221691818809, FactorizationMachineNativeOps.ffmPredict(weights, feats3), 1e-07);
    }

    private static IntBuffer allocateDirectIntBufferOf(int... src) {
        return ByteBuffer
                .allocateDirect(src.length * Integer.BYTES)
                .order(ByteOrder.nativeOrder())
                .asIntBuffer()
                .put(src);
    }

    private static FloatBuffer allocateDirectFloatBuffer(int size) {
        return ByteBuffer
                .allocateDirect(size * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
    }
}
