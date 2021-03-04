package hr.fer.zemris.neurofuzzy;

import java.util.ArrayList;
import java.util.List;

public class Demo {

	public static void main(String[] args) {
		List<FunctionValue> trainingData = generateTrainingData();
		
		Network anfis = new Network(15, trainingData);
		
		anfis.batchBackpropagation(0.0001, 10000);
		
		
	}
	
	public static List<FunctionValue> generateTrainingData() {
		ArrayList<FunctionValue> trainingData = new ArrayList<>();
		
		for (int i=-4; i<=4; i++)
			for (int j=-4; j<=4; j++)
				trainingData.add(new FunctionValue(i, j));
		
		return trainingData;
	}

}
