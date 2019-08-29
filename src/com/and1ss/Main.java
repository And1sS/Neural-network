package com.and1ss;

import Jama.Matrix;

import java.util.Scanner;

public class Main {

    // data for XOR logic gate
    private static final double[][][] learningData = {
            { {0, 0}, {0, 1} },
            { {0, 1}, {1, 0} },
            { {1, 0}, {1, 0} },
            { {1, 1}, {0, 1} }
    };

    private static NeuralNetwork net;
    public static void main(String[] args) {
       //net = new NeuralNetwork(new int[] {2, 10, 10, 2});

        //net.learn(learningData, 10000, 1, 1000);
        //net.toFile("test.net");
        net = NeuralNetwork.fromFile("test.net");
        Scanner scanner = new Scanner(System.in);

        while (true) {
            int a = scanner.nextInt();
            int b = scanner.nextInt();

            if(a == -2) {
                for(Matrix m : net.weights) {
                    net.print(m);
                    System.out.println();
                }
                for(Matrix m : net.biases) {
                    net.print(m);
                    System.out.println();
                }
            } else {
                net.forwardPropagation(new double[]{a, b});
                net.print(net.outputs[net.layers - 1]);
            }
        }
    }
}

