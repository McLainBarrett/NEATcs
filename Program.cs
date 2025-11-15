using static NEATcs;

public class Program {

	static void Main(string[] args) {
		List<float> results = new List<float>();
		List<int> generationsTaken = new List<int>();
		int successes = 0;
		for (int i = 0; i < 25; i++) {
			var myResults = Train(XOR, generationSize:150, cycles: 300);
			var top = myResults.Item1[0];
			generationsTaken.Add(myResults.Item2);
			XOR(top, true);
			results.Add(top.fitness);
			if (results[i] > 15)
				successes++;
			Console.WriteLine(String.Format("Game: {0} -- Result: {1} -- Successes|Failures|Percentage: {2}|{3}|{4}% Size: {5} nodes Generations Taken: {6}", 
				i, Math.Round(results[i], 2), successes, i - successes + 1, Math.Round(successes / (i + 1f) * 100), top.nodes.Count, generationsTaken.Average()));
		}
		Console.WriteLine(String.Format("Total Successes = {0}", successes));
	}

	static float XOR(NN nn, bool verbose = false) {
		float abs(float num) { return Math.Abs(num); }
		float rnd(float num, int digits) { return (float)Math.Round(num, digits); }

		nn.Clear();

		float a = nn.Activate(new List<float>() { 0, 0, 1 })[0];
		float b = nn.Activate(new List<float>() { 0, 1, 1 })[0];
		float c = nn.Activate(new List<float>() { 1, 0, 1 })[0];
		float d = nn.Activate(new List<float>() { 1, 1, 1 })[0];
		float fitness = (float)Math.Pow(4 - (abs(-a) + abs(1 - b) + abs(1 - c) + abs(-d)), 2);

		if (verbose) {
			Console.WriteLine(String.Format("(0,0)->{0} ({1})", rnd(a, 2), rnd(0 - a, 2)));
			Console.WriteLine(String.Format("(0,1)->{0} ({1})", rnd(b, 2), rnd(1 - b, 2)));
			Console.WriteLine(String.Format("(1,0)->{0} ({1})", rnd(c, 2), rnd(1 - c, 2)));
			Console.WriteLine(String.Format("(1,1)->{0} ({1})", rnd(d, 2), rnd(0 - d, 2)));
			Console.WriteLine("Total: " + rnd(fitness, 4));
		}

		return fitness;
	}

}