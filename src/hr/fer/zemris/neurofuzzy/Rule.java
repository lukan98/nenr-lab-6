package hr.fer.zemris.neurofuzzy;

import java.util.Random;

public class Rule {
	
	private double a, b, c, d, p, q, r;
	
	private double deltaA, deltaB, deltaC, deltaD, deltaP, deltaQ, deltaR;

	public Rule() {
		Random rand = new Random();
		
		this.a = rand.nextDouble();
		this.b = rand.nextDouble();
		this.c = rand.nextDouble();
		this.d = rand.nextDouble();
		this.p = rand.nextDouble();
		this.q = rand.nextDouble();
		this.r = rand.nextDouble();
		
		this.deltaA = this.deltaB = this.deltaC = this.deltaD = this.deltaP = this.deltaQ = this.deltaR = 0;
	}
	
	public double calculateAlpha(double x) {
		return 1/(1+Math.exp(b*(x-a)));
	}
	
	public double calculateBeta(double y) {
		return 1/(1+Math.exp(d*(y-c)));
	}
	
	public double calculateZ(double x, double y) {
		return p*x + q*y + r;
	}
	
	public double calculateW(double x, double y) {
		return calculateAlpha(x)*calculateBeta(y);
	}

	public void addDeltaA(double deltaA) {
		this.deltaA += deltaA;
	}

	public void addDeltaB(double deltaB) {
		this.deltaB += deltaB;
	}

	public void addDeltaC(double deltaC) {
		this.deltaC += deltaC;
	}

	public void addDeltaD(double deltaD) {
		this.deltaD += deltaD;
	}

	public void addDeltaP(double deltaP) {
		this.deltaP += deltaP;
	}

	public void addDeltaQ(double deltaQ) {
		this.deltaQ += deltaQ;
	}

	public void addDeltaR(double deltaR) {
		this.deltaR += deltaR;
	}
	
	public double getA() {
		return a;
	}

	public double getB() {
		return b;
	}

	public double getC() {
		return c;
	}

	public double getD() {
		return d;
	}
	
	public void updateRule(double learningRate) {
		this.a += learningRate*this.deltaA;
		this.b += learningRate*this.deltaB;
		this.c += learningRate*this.deltaC;
		this.d += learningRate*this.deltaD;
		this.p += learningRate*this.deltaP;
		this.q += learningRate*this.deltaQ;
		
		this.deltaA = this.deltaB = this.deltaC = this.deltaD = this.deltaP = this.deltaQ = this.deltaR = 0;
	}

	@Override
	public String toString() {
		return "Rule a=" + a + ", b=" + b + ", c=" + c + ", d=" + d + ", p=" + p + ", q=" + q + ", r=" + r;
	}

}
