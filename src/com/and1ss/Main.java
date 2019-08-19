package com.and1ss;

import Jama.Matrix;
import com.sun.jdi.InvalidTypeException;

import java.util.Scanner;

public class Main {

    private static final double[][][] learningData = {
            { {0, 0}, {0} },
            { {0, 1}, {1} },
            { {1, 0}, {1} },
            { {1, 1}, {0} }
    };

    private static NeuralNetwork net;
    public static void main(String[] args) {
        net = new NeuralNetwork(new int[] {2, 3, 1});

        net.learn(learningData, 100000, 1, 10);

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

