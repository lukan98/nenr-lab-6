package hr.fer.zemris.neurofuzzy;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class Network {
	
	private Rule rules[];
	private List<FunctionValue> trainingData;
	
	private double weights[];
	private double weightedWeights[];
	
	public Network(int noOfRules, List<FunctionValue> trainingData) {
		this.rules = new Rule[noOfRules];
		this.trainingData = trainingData;
		
		for (int i=0; i<noOfRules; i++) {
			this.rules[i] = new Rule();
		}
		
		this.weights = new double[noOfRules];
		this.weightedWeights = new double[noOfRules];
	}
	
	public double forwardPropagate(double x, double y) {
		
		for (int i=0; i<this.rules.length; i++)
			weights[i] = rules[i].calculateW(x, y);
		
		double weightSum = getWeightSum();
		
		for (int i=0; i<this.rules.length; i++)
			weightedWeights[i] = weights[i]/weightSum;
		
		double f = 0;
		
		for (int i=0; i<this.rules.length; i++) {
			f += weightedWeights[i]*this.rules[i].calculateZ(x, y);
		}
		
		return f;
		
	}
	
	public double getMSE() {
		double sum = 0;
		
		for (FunctionValue fv : this.trainingData)
			sum += Math.pow(fv.getF() - forwardPropagate(fv.getX(), fv.getY()), 2);
		
		return sum/this.trainingData.size();
	}
	
	private double getWeightSum() {
		double sum = 0;
		
		for (double w : this.weights) {
			sum += w;
		}
		
		return sum;
	}
	
	private double getWeightZSum(double x, double y) {
		double sum = 0;
		
		for (int i=0; i<this.rules.length; i++) {
			sum += this.weights[i]*this.rules[i].calculateZ(x, y);
		}
		
		return sum;
	}

	public void stochasticBackpropagation(double learningRate, int noOfEpochs) {
		
		for (int epoch=0; epoch<noOfEpochs; epoch++) {
			for (FunctionValue fv : this.trainingData) {
				double x = fv.getX();
				double y = fv.getY();
				double fk = forwardPropagate(x, y);
				
				double diff = fv.getF() - fk;
				double weightSum = getWeightSum();
				double wZSum = getWeightZSum(x, y);
				
				if (weightSum == 0) throw new ArithmeticException("Cannot divide by zero; check your weights");
				
				for (int i=0; i<this.rules.length; i++) {
					double alphaI = rules[i].calculateAlpha(x);
					double betaI = rules[i].calculateBeta(y);
					
					rules[i].addDeltaP(diff*alphaI*betaI*x/weightSum);
					rules[i].addDeltaQ(diff*alphaI*betaI*y/weightSum);
					rules[i].addDeltaR(diff*alphaI*betaI/weightSum);
					
					double zI = rules[i].calculateZ(x, y);
					
					double aI = rules[i].getA();
					double bI = rules[i].getB();
					double cI = rules[i].getC();
					double dI = rules[i].getD();
					
					double W = weightSum - this.weights[i];
					double WZ = wZSum - this.weights[i]*zI;
					
					rules[i].addDeltaA(diff*betaI*(zI*W - WZ)*alphaI*(1-alphaI)*bI / Math.pow(weightSum, 2));
					rules[i].addDeltaB(diff*betaI*(zI*W - WZ)*(-alphaI)*(1-alphaI)*(x-aI) / Math.pow(weightSum, 2));
					rules[i].addDeltaC(diff*alphaI*(zI*W - WZ)*betaI*(1-betaI)*dI / Math.pow(weightSum, 2));
					rules[i].addDeltaD(diff*alphaI*(zI*W - WZ)*(-betaI)*(1-betaI)*(y-cI) / Math.pow(weightSum, 2));
				}
				
				for (int i=0; i<this.rules.length; i++)
					this.rules[i].updateRule(learningRate);
				
			}
			System.out.println((epoch+1)+" "+this.getMSE());
		}
	}
	
	public void batchBackpropagation(double learningRate, int noOfEpochs) {
		
		for (int epoch=0; epoch<noOfEpochs; epoch++) {
			for (FunctionValue fv : this.trainingData) {
				double x = fv.getX();
				double y = fv.getY();
				double fk = forwardPropagate(x, y);
				
				double diff = fv.getF() - fk;
				double weightSum = getWeightSum();
				double wZSum = getWeightZSum(x, y);
				
				if (weightSum == 0) throw new ArithmeticException("Cannot divide by zero; check your weights");
				
				for (int i=0; i<this.rules.length; i++) {
					double alphaI = rules[i].calculateAlpha(x);
					double betaI = rules[i].calculateBeta(y);
					
					rules[i].addDeltaP(diff*alphaI*betaI*x/weightSum);
					rules[i].addDeltaQ(diff*alphaI*betaI*y/weightSum);
					rules[i].addDeltaR(diff*alphaI*betaI/weightSum);
					
					double zI = rules[i].calculateZ(x, y);
					
					double aI = rules[i].getA();
					double bI = rules[i].getB();
					double cI = rules[i].getC();
					double dI = rules[i].getD();
					
					double W = weightSum - this.weights[i];
					double WZ = wZSum - this.weights[i]*zI;
					
					rules[i].addDeltaA(diff*betaI*(zI*W - WZ)*alphaI*(1-alphaI)*bI / Math.pow(weightSum, 2));
					rules[i].addDeltaB(diff*betaI*(zI*W - WZ)*(-alphaI)*(1-alphaI)*(x-aI) / Math.pow(weightSum, 2));
					rules[i].addDeltaC(diff*alphaI*(zI*W - WZ)*betaI*(1-betaI)*dI / Math.pow(weightSum, 2));
					rules[i].addDeltaD(diff*alphaI*(zI*W - WZ)*(-betaI)*(1-betaI)*(y-cI) / Math.pow(weightSum, 2));
				}
			}
			
			for (int i=0; i<this.rules.length; i++)
				this.rules[i].updateRule(learningRate);
			
			System.out.println((epoch+1)+" "+this.getMSE());
		}
	}
	
	public void printOutputs() {
		for (FunctionValue fv : this.trainingData) {
			System.out.print("Stvarna vrijednost: \t"+fv.getX()+", "+fv.getY()+" -> "+fv.getF()+"\n");
			System.out.print("Izlaz modela: \t\t"+fv.getX()+", "+fv.getY()+" -> "+this.forwardPropagate(fv.getX(), fv.getY())+"\n");
			System.out.println("---------------------------------------------------");
		}
	}
	
	public void printOutputsToFile() {
		try (BufferedWriter br = new BufferedWriter(new FileWriter("/Users/lukanamacinski/FER-workspace/NENR-workspace/lab6/modelOutputs.out"))) {
			
			StringBuilder sb = new StringBuilder();
			
			for(int i=0; i<trainingData.size(); i++) {
				FunctionValue fv = trainingData.get(i);
				sb.append(fv.getX()+"\t"+fv.getY()+"\t"+this.forwardPropagate(fv.getX(), fv.getY())+"\n");
				if ((i+1)%9==0)
					sb.append("\n");
			}
			
			br.write(sb.toString());
			
		} catch(IOException e) {
			System.out.println("An error occured while trying to write to file.");
			e.printStackTrace();
		}
	}
	
	public void printDifferencesToFile() {
		try (BufferedWriter br = new BufferedWriter(new FileWriter("/Users/lukanamacinski/FER-workspace/NENR-workspace/lab6/differences.out"))) {
			
			StringBuilder sb = new StringBuilder();
			
			for(int i=0; i<trainingData.size(); i++) {
				FunctionValue fv = trainingData.get(i);
				sb.append(fv.getX()+"\t"+fv.getY()+"\t"+(this.forwardPropagate(fv.getX(), fv.getY())-fv.getF())+"\n");
				if ((i+1)%9==0)
					sb.append("\n");
			}
			
			br.write(sb.toString());
			
		} catch(IOException e) {
			System.out.println("An error occured while trying to write to file.");
			e.printStackTrace();
		}
	}

	public Rule[] getRules() {
		return rules;
	}
	
}
