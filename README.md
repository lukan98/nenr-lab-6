# ANFIS Neuro-Fuzzy Network (NENR Lab 6)

A neuro-fuzzy network in the style of ANFIS (Adaptive Neuro-Fuzzy Inference System), written in Java for the **NENR (Fuzzy, Evolutionary and Neurocomputing)** course at FER, University of Zagreb.

## Problem

The network approximates the two-dimensional function

```text
f(x, y) = ((x - 1)^2 + (y + 2)^2 - 5*x*y + 3) * cos(x/5)^2
```

on the domain `[-4, 4] × [-4, 4]`.

## Implementation

- 15 fuzzy rules with Gaussian-style membership functions
- Weighted-rule firing and weighted consequent outputs
- Batch backpropagation training of rule parameters
- Mean-squared-error evaluation on a grid of training points

## Requirements

- Java JDK 11 or newer
- Eclipse is optional; the repository includes Eclipse project metadata

## Build and run

From the project root:

```bash
mkdir -p bin
find src -name "*.java" -print0 | xargs -0 javac -d bin
java -cp bin hr.fer.zemris.neurofuzzy.Demo
```

## Project structure

```text
src/hr/fer/zemris/neurofuzzy/
├── Demo.java          # creates training data and trains the network
├── Network.java       # forward pass, MSE and backpropagation
├── Rule.java          # fuzzy rule parameters and firing strength
└── FunctionValue.java # target function and training sample
```
