package com.and1ss;

import Jama.Matrix;

import java.util.Random;

public class NeuralNetwork {
    public int[] neuronsInLayer;

    public Matrix[] weights;
    public Matrix[] weightsErrors;
    public Matrix[] biases;
    public Matrix[] biasesErrors;
    public Matrix[] outputs;

    public int layers;

    public NeuralNetwork(int[] neuronsInLayer) {
        this.neuronsInLayer = new int[neuronsInLayer.length];
        for(int i = 0; i < neuronsInLayer.length; i++) {
            this.neuronsInLayer[i] = neuronsInLayer[i];
        }

        layers = neuronsInLayer.length;

        weights = new Matrix[layers - 1];
        weightsErrors = new Matrix[layers - 1];
        biases = new Matrix[layers - 1];
        biasesErrors = new Matrix[layers - 1];
        outputs = new Matrix[layers];

        for(int i = 0; i < layers - 1; i++) {
            weights[i] = randomInitialise(neuronsInLayer[i + 1], neuronsInLayer[i]);
            weightsErrors[i] = new Matrix(neuronsInLayer[i + 1], neuronsInLayer[i]);
            biases[i] = randomInitialise(neuronsInLayer[i + 1], 1);
            biasesErrors[i] = new Matrix(neuronsInLayer[i + 1], 1);
        }
    }

    public void print(Matrix m) {
        for(int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                System.out.print(m.get(i, j) + " ");
            }
            System.out.println();
        }
    }

    public void forwardPropagation(double[] inputs) {
        Matrix inputMatrix = arrayToColumnMatrix(inputs);
        outputs[0] = inputMatrix;
        for(int i = 0; i < layers - 1; i++) {
            inputMatrix = sigmoid(weights[i].times(inputMatrix).plus(biases[i]));
            outputs[i + 1] = inputMatrix;
        }
    }

    public void backPropagation(Matrix desiredOutputs) {
        Matrix errors = outputs[layers - 1].minus(desiredOutputs);

        for(int i = layers - 1; i > 0; i--) {
            errors.arrayTimesEquals(sigmoidDerivative(outputs[i])); //inputs error
            weightsErrors[i - 1].plusEquals(errors.times(outputs[i - 1].transpose())); // weights error
            biasesErrors[i - 1].plusEquals(errors); // biases error
            errors = weights[i - 1].transpose().times(errors); // outputs error
        }
    }

    private Matrix sigmoidDerivative(Matrix m) {
        return m.minus(m.arrayTimes(m));
    }

    public void learn(double[][][] learningData, int epoches, double learningRate, int miniBatchSize) {
        Random random = new Random();
        for(int i = 0; i < epoches; i++) {
            double cost = 0;
            for(int j = 0; j < miniBatchSize; j++) {
                int index = (random.nextInt() & Integer.MAX_VALUE) % learningData.length;

                forwardPropagation(learningData[index][0]);
                backPropagation(arrayToColumnMatrix(learningData[index][1]));

                cost += cost(arrayToColumnMatrix(learningData[index][1]));
            }
            System.out.println("epoch #" + i + " error: " + cost);

            for(int j = 0; j < layers - 1; j++) {
                weights[j].minusEquals(weightsErrors[j].times(learningRate / miniBatchSize));
                biases[j].minusEquals(biasesErrors[j].times(learningRate / miniBatchSize));

                weightsErrors[j] = new Matrix(weightsErrors[j].getRowDimension(),
                        weightsErrors[j].getColumnDimension());
                biasesErrors[j] = new Matrix(biasesErrors[j].getRowDimension(), 1);
            }
        }
    }

    private Matrix arrayToColumnMatrix(double[] array) {
        Matrix m = new Matrix(array.length, 1);

        for(int i = 0; i < array.length; i++) {
            m.set(i, 0, array[i]);
        }

        return m;
    }

    private Matrix sigmoid(Matrix m) {
        Matrix temp = new Matrix(m.getRowDimension(), 1);
        for(int i = 0; i < m.getRowDimension(); i++) {
            double value = 1.0 / (1.0 + Math.exp(-m.get(i, 0)));
            temp.set(i, 0, value);
        }
        return temp;
    }

    private double cost(Matrix desired) {
        double result = 0;
        for(int i = 0; i < desired.getRowDimension(); i++) {
            double temp = outputs[layers - 1].get(i, 0)
                    - desired.get(i, 0);
            result += temp * temp;
        }
        result /= desired.getRowDimension();
        return result;
    }

    private Matrix randomInitialise(int m, int n) {
        Random rnd = new Random();
        Matrix temp = new Matrix(m, n);
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                temp.set(i, j, (rnd.nextGaussian()));
            }
        }
        return temp;
    }
}
