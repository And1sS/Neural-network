package com.and1ss;

import Jama.Matrix;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
    private static final String TOPOLOGY_FLAG = "NETWORK_TOPOLOGY";
    private static final String WEIGHTS_FLAG = "WEIGHTS";
    private static final String BIASES_FLAG = "BIASES";
    private static final String START_OF_MATRIX_FLAG = "START_OF_MATRIX";
    private static final String END_OF_MATRIX_FLAG = "END_OF_MATRIX";
    private static final String MATRIX_HEIGHT_FLAG = "HEIGHT_OF_MATRIX";
    private static final String MATRIX_WIDTH_FLAG = "WIDTH_OF_MATRIX";

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
        for(int i = 0; i < layers - 2; i++) {
            inputMatrix = sigmoid(weights[i].times(inputMatrix).plus(biases[i]));
            outputs[i + 1] = inputMatrix;
        }
        inputMatrix = softmax(weights[layers - 2].times(inputMatrix)
                .plus(biases[layers - 2]));
        outputs[layers - 1] = inputMatrix;
    }

    public void backPropagation(Matrix desiredOutputs) {
        Matrix errors = outputs[layers - 1].minus(desiredOutputs);

        errors = softmaxDerivative(outputs[layers - 1]).times(errors); // last layer inputs error
        weightsErrors[layers - 2].plusEquals(errors.
                times(outputs[layers - 2].transpose())); // weights error
        biasesErrors[layers - 2].plusEquals(errors); // biases error
        errors = weights[layers - 2].transpose().times(errors); // outputs error

        for(int i = layers - 2; i > 0; i--) {
            errors.arrayTimesEquals(sigmoidDerivative(outputs[i])); //inputs error
            weightsErrors[i - 1].plusEquals(errors.times(outputs[i - 1].transpose())); // weights error
            biasesErrors[i - 1].plusEquals(errors); // biases error
            errors = weights[i - 1].transpose().times(errors); // outputs error
        }
    }

    public void learn(double[][][] learningData, int epoches, double learningRate, int miniBatchSize) {
        Random random = new Random();
        for(int i = 0; i < epoches; i++) {
            double cost = 0;

            // collecting errors for current mini batch
            for(int j = 0; j < miniBatchSize; j++) {
                int index = (random.nextInt() & Integer.MAX_VALUE) % learningData.length;

                forwardPropagation(learningData[index][0]);
                backPropagation(arrayToColumnMatrix(learningData[index][1]));

                cost += quadratic_cost(arrayToColumnMatrix(learningData[index][1]));
            }
            System.out.println("epoch #" + i + " error: " + cost);

            // updating weights and biases, resetting errors
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

    // sigmoid activation function
    private Matrix sigmoid(Matrix m) {
        Matrix temp = new Matrix(m.getRowDimension(), 1);
        for(int i = 0; i < m.getRowDimension(); i++) {
            double value = 1.0 / (1.0 + Math.exp(-m.get(i, 0)));
            temp.set(i, 0, value);
        }
        return temp;
    }

    private Matrix sigmoidDerivative(Matrix m) {
        return m.minus(m.arrayTimes(m));
    }

    private Matrix softmax(Matrix m)  {
        Matrix temp = new Matrix(m.getRowDimension(), 1);
        double sum = 0;
        for (int i = 0; i < m.getRowDimension(); i++) {
            sum += Math.exp(m.get(i, 0));
        }
        for(int i = 0; i < m.getRowDimension(); i++) {
            temp.set(i, 0, Math.exp(m.get(i, 0)) / sum);
        }
        return temp;
    }

    //softmax derivative for outputs
    private Matrix softmaxDerivative(Matrix m) {
        Matrix temp = new Matrix(m.getRowDimension(), m.getRowDimension());
        for(int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getRowDimension(); j++) {
                temp.set(i, j, i == j ? m.get(i, 0) * (1.0 - m.get(i, 0))
                        : -m.get(i, 0) * m.get(j, 0));
            }
        }
        return temp;
    }

    // quadratic cost function
    private double quadratic_cost(Matrix desired) {
        double result = 0;
        for(int i = 0; i < desired.getRowDimension(); i++) {
            double temp = outputs[layers - 1].get(i, 0)
                    - desired.get(i, 0);
            result += temp * temp;
        }
        result /= desired.getRowDimension();
        return result;
    }

    //random initialised matrix with gaussian nubmers
    private Matrix randomInitialise(int m, int n) {
        Random rnd = new Random();
        Matrix temp = new Matrix(m, n);
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                temp.set(i, j, rnd.nextGaussian());
            }
        }
        return temp;
    }

    public void toFile(String fileName) {
        PrintWriter writer = null;
        try {
                writer = new PrintWriter(fileName, "UTF-8");
                writer.println(TOPOLOGY_FLAG);
                for(int i : neuronsInLayer) {
                    writer.print(String.valueOf(i) + " ");
                }
                writer.println();
                writer.println(WEIGHTS_FLAG);
                for(Matrix m : weights) {
                    matrixToFile(m, writer);
                }
                writer.println(BIASES_FLAG);
                for(Matrix m : biases) {
                    matrixToFile(m, writer);
                }
        } catch (IOException e) {
            System.out.println("writing to file error");
        } finally {
            writer.close();
        }
    }

    private void matrixToFile(Matrix m, PrintWriter writer) {
        writer.println(START_OF_MATRIX_FLAG);
        writer.println(MATRIX_HEIGHT_FLAG);
        writer.println(m.getRowDimension());
        writer.println(MATRIX_WIDTH_FLAG);
        writer.println(m.getColumnDimension());
        for(int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                writer.print(m.get(i, j) +
                        (j == m.getColumnDimension() - 1 ? "" : " "));
            }
            writer.println();
        }
        writer.println(END_OF_MATRIX_FLAG);
    }

    public static NeuralNetwork fromFile(String fileName) {
        NeuralNetwork net = null;

        int[] topology;
        try {
            String line = null;
            FileReader fileReader = new FileReader(fileName);

            BufferedReader bufferedReader =
                    new BufferedReader(fileReader);

            if(!bufferedReader.readLine().equals(TOPOLOGY_FLAG)) {
                throw new Exception();
            }
            line = bufferedReader.readLine();
            List<String> elements =
                    new ArrayList<String>(Arrays.asList(line.split(" ")));
            topology = new int[elements.size()];
            for (int i = 0; i < elements.size(); i++) {
                topology[i] = Integer.valueOf(elements.get(i));
            }
            net = new NeuralNetwork(topology);

            if(!bufferedReader.readLine().equals(WEIGHTS_FLAG)) {
                throw new Exception();
            }
            for(int i = 0; i < topology.length - 1; i++) {
                net.weights[i] = readMatrix(bufferedReader);
            }
            if(!bufferedReader.readLine().equals(BIASES_FLAG)) {
                throw new Exception();
            }
            for(int i = 0; i < topology.length - 1; i++) {
                net.biases[i] = readMatrix(bufferedReader);
            };
            bufferedReader.close();
        } catch(FileNotFoundException ex) {
            System.out.println("Unable to open file '" +
                            fileName + "'");
            return null;
        } catch(IOException ex) {
            System.out.println("Error reading file '"
                            + fileName + "'");
            return null;
        } catch (Exception e) {
            System.out.println("File '"
                    + fileName + "' is corrupted");
            return null;
        }
        return net;
    }

    private static Matrix readMatrix(BufferedReader reader) throws
            IOException, Exception{
        Matrix temp;

        if(!reader.readLine().equals(START_OF_MATRIX_FLAG))
            throw new Exception();
        if(!reader.readLine().equals(MATRIX_HEIGHT_FLAG))
            throw new Exception();
        int height = Integer.valueOf(reader.readLine());
        if(!reader.readLine().equals(MATRIX_WIDTH_FLAG))
            throw new Exception();
        int width = Integer.valueOf(reader.readLine());
        temp = new Matrix(height, width);
        List<String> rowElements;
        for(int i = 0; i < temp.getRowDimension(); i++) {
            rowElements = new ArrayList<String>(
                    Arrays.asList(reader.readLine()
                            .split(" ")));
            for (int j = 0; j < temp.getColumnDimension(); j++) {
                temp.set(i, j, Double.valueOf(rowElements.get(j)));
            }
        }
        if(!reader.readLine().equals(END_OF_MATRIX_FLAG)) {
            throw new Exception();
        }

        return temp;
    }
}
