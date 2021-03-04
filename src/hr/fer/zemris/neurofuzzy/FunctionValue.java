package hr.fer.zemris.neurofuzzy;

public class FunctionValue {
	
	private double x, y, f;
	
	public FunctionValue(double x, double y) {
		this.f = evaluate(x, y);
		this.x = x;
		this.y = y;
	}
	
	private static double evaluate(double x, double y) {
		if (x > 4 || x < -4 || y > 4 || y < -4) throw new IllegalArgumentException("Funkcija je definirana u području [-4,4]x[-4,4]");
		
		return (Math.pow(x-1, 2) + Math.pow(y+2, 2) - 5*x*y + 3) * Math.pow(Math.cos(x/5), 2);
	}
	
	public double getX() {
		return this.x;
	}
	
	public double getY() {
		return this.y;
	}
	
	public double getF() {
		return this.f;
	}

	@Override
	public String toString() {
		return "FunctionValue x=" + x + ", y=" + y + ", f=" + f;
	}

}
