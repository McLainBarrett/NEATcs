using System.Collections.Generic;

namespace NEAT {
	internal class Program {
		static void Main(string[] args) {
			List<float> results = new List<float>();
			int successes = 0;
			for (int i = 0; i < 100; i++) {
				results.Add(XORproblem());
				if (results[i] > -0.5)
					successes++;
				 Console.WriteLine(i + " -- " + Math.Round(results[i], 2) + " -- " + Math.Round(successes/(i+1f)*100) + "%");
			}
			Console.WriteLine(String.Format("Total Successes = {0}", successes));
		}

		static int SelectWeighted(int max) {
			float rand = (float)new Random().NextDouble();
			float sum = 0;
			float total = 0;
			for (int i = 0; i < max; i++)
				total++;
			for (int i = 0; i < max; i++) {
				sum += i;
				if (rand < sum / total)
					return i;
			}
			return 0;
		}

		static float NNdistance(NN a, NN b) {

			float co1 = 1, co2 = 1, co3 = 1;
			int ai = a.connections.Count - 1;
			int bi = b.connections.Count - 1;

			int disjoint = 0;
			int excess = 0;
			float weightDiff = 0;
			int total = Math.Max(ai, bi);
			bool isNowDisjoint = false;

			//Iterate backwards through genes
			while (Math.Min(ai, bi) >= 0) {
				if (a.connections[ai].Innovation == b.connections[bi].Innovation) {
					weightDiff += Math.Abs(a.connections[ai].Weight - b.connections[bi].Weight);
					ai--; bi--;

				} else {
					if (a.connections[ai].Innovation > b.connections[bi].Innovation)
						ai--;
					else
						bi--;

					//Count excess first, then after first same case, count disjoint
					if (isNowDisjoint)
						disjoint++;
					else
						excess++;
				}
			}
			disjoint += Math.Abs(ai - bi);

			return disjoint * co1 / total + excess * co2 / total + weightDiff * co3;
		}

		static float XORproblem() {
			//Create population
			List<NN> population = new List<NN>();
			List<NN> species = new List<NN>();
			List<int> speciesCounts = new List<int>() { 1 };

			int generationSize = 100;
			int cycles = 1000;

			for (int i = 0; i < generationSize; i++) {
				population.Add(new NN(3, 1));
				species.Add(population[0].Copy());
				population[i].Mutate();
			}

			//Train population
			for (int k = 0; k < cycles; k++) {

				//--Speciate--//
				//Remove empty species, reset count
				for (int i = 0; i < species.Count; i++) {
					if (speciesCounts[i] == 0) {
						speciesCounts.RemoveAt(i);
						species.RemoveAt(i);
					} else {
						speciesCounts[i] = 0;
					}
				}

				//For each population, iterate through each species, if comparability is within threshhold, assign species number
				//If end is reached, create new species with it as example
				float speciesThreshhold = 1;
				for (int i = 0; i < population.Count; i++) {
					for (int j = 0; j < species.Count; j++) {
						if (NNdistance(population[i], species[j]) < speciesThreshhold) {
							population[i].species = j;
							speciesCounts[j]++;
							break;
						}
						if (j == species.Count-1) {//No species found
							species.Add(population[i].Copy());
							speciesCounts.Add(1);
						}
					}
				}

				//Update representatives
				//Each existing species is represented by a random genome inside the species from the previous generation.
				var rand = new Random();
				List<int> chosenReps = new List<int>();
				for (int i = 0; i < speciesCounts.Count; i++)
					chosenReps.Add(rand.Next(speciesCounts[i]));

				for (int i = 0; i < population.Count; i++) {
					int mySpecies = population[i].species;
					chosenReps[mySpecies]--;
					if (chosenReps[mySpecies] == 0)
						species[i] = population[i].Copy();
				}



				//--Assess--//
				List<float> fitnesses = new List<float>();
				var speciesPops = new List<List<NN>>(new List<NN>[species.Count]);
				List<float> speciesFitnesses = new List<float>(new float[species.Count]);
				float totalFitness = 0;
				for (int i = 0; i < population.Count; i++) {
					float fitness = XOR(population[i]) / speciesCounts[population[i].species];
					population[i].fitness = fitness;
					fitnesses.Add(fitness);
					speciesFitnesses[population[i].species] += fitness;
					speciesPops[population[i].species].Add(population[i]);
					totalFitness += fitness;
				}



				//--Crossover and Mutate--//
				for (int j = 0; j < speciesPops.Count; j++) {
					//Sort and cull population by fitness
					var speciesPopulation = speciesPops[j];
					speciesPops[j].Sort((x, y) => y.fitness.CompareTo(x.fitness));
					var survivors = new List<NN>();
					int targ = speciesPopulation.Count / 2;
					for (int i = 0; i < targ; i++) {
						int index = SelectWeighted(speciesPopulation.Count);
						survivors.Add(speciesPopulation[index]);
						speciesPopulation.RemoveAt(index);
					}
					
					//Crossover population, replace with offspring
					int newPopCount = (int)Math.Floor(generationSize * speciesFitnesses[j] / totalFitness);
					var crossPool = new List<NN>(speciesPopulation);
					var offspring = new List<NN>();

					while (offspring.Count < newPopCount) {
						if (speciesPopulation.Count == 1) {//If only one member...
							var clone = speciesPopulation[0].Copy();//Use mitosis
							clone.Mutate();
							offspring.Add(clone);
							continue;
						}

						//Otherwise...
						var a = crossPool[new Random().Next(crossPool.Count)];
						crossPool.Remove(a);
						var b = crossPool[new Random().Next(crossPool.Count)];
						crossPool.Remove(b);
						offspring.Add(NN.Crossover(a, b));
						if (crossPool.Count < 2)//If not enough...
							crossPool.AddRange(speciesPopulation);//Refill crossing population
					}

					speciesPops[j] = new List<NN>(offspring);
				}

				population.Clear();
				for (int i = 0; i < speciesPops.Count; i++)
					population.AddRange(speciesPops[i]);

				//Mutate all networks
				foreach (NN nn in population) {
					nn.Clear();
					nn.Mutate();
				}

				if (k % 100 == 0)
					Console.WriteLine(k + " -- " + Math.Round(fitnesses.Average(), 2) + " -- " + Math.Round(fitnesses.Max(), 2));
			}

			Console.WriteLine(XOR(population[0], true));
			Console.WriteLine(population[0].fitness);
			//Console.WriteLine(population[0].nodes.Count + " -- " + population[0].connections.Count);
			return population[0].fitness;
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

		class NN {

			public List<Node> nodes = new List<Node>();
			public List<Connection> connections = new List<Connection>();
			public (int Inputs, int Outputs) size;
			public float fitness;
			public int species;

			public bool allowRecursion = true;//!!Might* Never works when false!!

			public NN(int InputCount, int OutputCount) {
				size = (InputCount, OutputCount);
				//Add input and output nodes
				for (int i = 0; i < InputCount; i++)
					nodes.Add(new Node(nodes.Count, 0, NodeType.Input));
				for (int i = 0; i < OutputCount; i++)
					nodes.Add(new Node(nodes.Count, 0, NodeType.Output));

				//Add connections from each input to each output
				for (int i = 0; i < InputCount; i++) {
					for (int j = 0; j < OutputCount; j++) {
						Connection conn = new Connection(i, InputCount + j, 0, connections.Count);
						connections.Add(conn);
						nodes[i].Connections.Add(conn);
						nodes[InputCount + j].Connections.Add(conn);
					}
				}

				if (GetGIN(0) == 0)
					GetGIN(connections.Count);
			}
			public List<float> Activate(List<float> input) {
				List<float> output = new List<float>();

				//Push inputs into input nodes
				for (int i = 0; i < size.Inputs; i++) {
					Node node = nodes[i];
					if (node.Type == NodeType.Input)
						node.Value = input[i] + node.Bias;
				}
				//Compute hidden nodes
				for (int i = size.Inputs + size.Outputs; i < nodes.Count; i++)
					nodes[i].Activate(nodes);

				//Compute output nodes, grab results
				for (int i = size.Inputs; i < size.Inputs + size.Outputs; i++) {
					nodes[i].Activate(nodes);
					output.Add(nodes[i].Value);
				}
				return output;
			}

			private static bool chance(float probability) {
				return new Random().NextDouble() < probability;
			}
			public void Mutate() {
				//Mutate weights and biases
				for (int i = 0; i < nodes.Count; i++)
					if (chance(0.05f))
						nodes[i].Bias += (float)(new Random().NextDouble() * 2 - 1);
				for (int i = 0; i < connections.Count; i++)
					if (chance(0.05f))
						connections[i].Weight += (float)(new Random().NextDouble() * 2 - 1);

				//Structural mutations

				float addConn = 0.05f;
				float addNode = 0.03f;
				if (chance(addConn)) {//Add connection
					Node node = nodes[new Random().Next(nodes.Count)];//Select random node
					//Find a node that has no connections from this node
					var otherNodes = nodes.FindAll(x => x.Connections.FindAll(y => y.From == node.Index).Count == 0 && (allowRecursion || x != node));//(allowRecursion || (x.Connections.FindAll(y => y.To == node.Index).Count == 0 && x != node)
					if (otherNodes.Count > 0) {
						Node otherNode = otherNodes[new Random().Next(otherNodes.Count)];

						//Check for previous identical mutations
						var mutation = mutations.FindAll(x => x.isNewNode == false && x.A == node.Index && x.B == otherNode.Index);
						int gin = (mutation.Count == 0) ? GetGIN() : mutation[0].GIN;

						//Create connection
						Connection connection = new Connection(node.Index, otherNode.Index, (float)(new Random().NextDouble() * 2 - 1), gin);
						connections.Add(connection);
						mutations.Add((false, node.Index, otherNode.Index, gin));
					}
				}

				if (chance(addNode)) {//Add Node
					Connection oldConn = connections[new Random().Next(connections.Count)];
					oldConn.Enabled = false;//Disable old connection
					Node newNode = new Node(nodes.Count, 0, NodeType.Hidden);
					nodes.Add(newNode);

					//Check for previous identical mutations
					var mutation = mutations.FindAll(x => x.isNewNode == true && x.A == oldConn.From && x.B == oldConn.To);
					int gin = (mutation.Count == 0) ? GetGIN() : mutation[0].GIN; GetGIN();//increment GIN twice

					connections.Add(new Connection(oldConn.From, newNode.Index, 1, gin));//Add connection from->new
					connections.Add(new Connection(newNode.Index, oldConn.To, oldConn.Weight, gin + 1));//Add connection new->to

					mutations.Add((true, oldConn.From, oldConn.To, gin));
				}

				/*
				float addConn = 0.05f;
				float addNode = 0.03f;
				if (chance(addConn)) {//Add connection
					//Select random node
					Node node = nodes[new Random().Next(nodes.Count)];
					//Find a node that has no connections from this node
					var otherNodes = nodes.FindAll(x => x.Connections.FindAll(y => y.From == node.Index).Count == 0);
					if (otherNodes.Count > 0) {
						Node otherNode = otherNodes[new Random().Next(otherNodes.Count)];

						//Check for previous identical mutations
						var mutation = mutations.FindAll(x => x.isNewNode == false && x.A == node.Index && x.B == otherNode.Index);
						int gin = (mutation.Count == 0) ? GetGIN() : mutation[0].GIN;

						//Create connection
						Connection connection = new Connection(node.Index, otherNode.Index, (float)(new Random().NextDouble() * 2 - 1), gin);
						connections.Add(connection);
						mutations.Add((false, node.Index, otherNode.Index, gin));
					}

				} else if (chance(addNode/(1-addConn))) {//Add Node
					Connection oldConn = connections[new Random().Next(connections.Count)];
					oldConn.Enabled = false;//Disable old connection
					Node newNode = new Node(nodes.Count, 0, NodeType.Hidden);
					nodes.Add(newNode);

					//Check for previous identical mutations
					var mutation = mutations.FindAll(x => x.isNewNode == true && x.A == oldConn.From && x.B == oldConn.To);
					int gin = (mutation.Count == 0) ? GetGIN() : mutation[0].GIN; GetGIN();//increment GIN twice

					connections.Add(new Connection(oldConn.From, newNode.Index, 1, gin));//Add connection from->new
					connections.Add(new Connection(newNode.Index, oldConn.To, oldConn.Weight, gin + 1));//Add connection new->to

					mutations.Add((true, oldConn.From, oldConn.To, gin));
				}
				*/
			}
			/*public static NN Crossover(NN parentA, NN parentB, float fitnessA, float fitnessB) {
				if (fitnessA < fitnessB) {
					NN tempNN = parentA;
					parentA = parentB;
					parentB = tempNN;
					float tempFit = fitnessA;
					fitnessA = fitnessB;
					fitnessB = tempFit;
				}
				return Crossover(parentA, parentB);
			}*/
			public static NN Crossover(NN parentA, NN parentB) {
				//ParentA is *always* more fit than parentB
				if (parentA.fitness < parentB.fitness) {
					NN tempNN = parentA;
					parentA = parentB;
					parentB = tempNN;
				}

				NN offspring = new NN(0, 0);

				//Splice connections
				int j = 0;
				var Aconns = parentA.connections;
				var Bconns = parentB.connections;
				for (int i = 0; i < Aconns.Count; i++) {//For each gene of the fitter parent...
					Connection conn;
					for (; j < Bconns.Count - 1 && Aconns[i].Innovation > Bconns[j].Innovation; j++) { }//Get next B connection until it is the same or greater inov number
					if (Aconns[i].Innovation == Bconns[j].Innovation)//If genes are matching, choose randomly
						conn = chance(0.5f) ? Aconns[i] : Bconns[j];
					else
						conn = Aconns[i];//If only fitter parent has gene, pass to offspring
					offspring.connections.Add(conn.Copy());
				}

				//Splice nodes
				for (int i = 0; i < parentA.nodes.Count; i++) {//For each gene of the fitter parent...
					Node node;
					if (i < parentB.nodes.Count)//If genes are matching, choose randomly
						node = chance(0.5f) ? parentA.nodes[i] : parentB.nodes[i];
					else
						node = parentA.nodes[i];//If only fitter parent has gene, pass to offspring
					offspring.nodes.Add(node.Copy());
				}

				//Update node connections list
				for (int i = 0; i < offspring.connections.Count; i++) {
					var conn = offspring.connections[i];
					if (conn.From < offspring.nodes.Count && conn.To < offspring.nodes.Count)
						offspring.nodes[conn.To].Connections.Add(conn);
				}
				offspring.size = (parentA.size.Inputs, parentA.size.Outputs);

				//Console.WriteLine(String.Format("Parents vs child: {0} and {1} vs {2}", parentA.nodes.Count, parentB.nodes.Count, offspring.nodes.Count));

				return offspring;
			}
			public NN Copy() {
				NN copy = new NN(0, 0);
				copy.size = size;

				//Duplicate nodes and connections
				for (int i = 0; i < nodes.Count; i++)
					copy.nodes.Add(nodes[i].Copy());
				for (int i = 0; i < connections.Count; i++)
					copy.connections.Add(connections[i].Copy());

				//Reconnect node connection lists
				foreach (var conn in copy.connections)
					copy.nodes[conn.To].Connections.Add(conn);

				return copy;
			}

			public void Clear() {
				foreach (Node node in nodes) {
					node.State = 0;
					node.Value = 0;
				}
			}

			public class Node {
				public float State = 0;//Before activation function
				public float Value = 0;//After activation function
				public int Index;
				public float Bias;
				public NodeType Type;
				public List<Connection> Connections = new List<Connection>();
				public Node(int index, float bias, NodeType type) {
					Index = index;
					Bias = bias;
					Type = type;
				}
				public void Activate(List<Node> nodes) {
					float sum = Bias;
					foreach (Connection connection in Connections) {
						if (!connection.Enabled)
							continue;

						Node prevNode = nodes[connection.From];
						float weight = connection.Weight;
						if (connection.From != Index)
							sum += prevNode.Value * weight;
						else
							sum += State * weight;
					}
					State = sum;
					Value = 1 / (1 + (float)Math.Exp(-State));
				}
				public Node Copy() {
					return new Node(Index, Bias, Type);
				}
				public override string ToString() {
					return String.Format("[{0} {1}]", Index, Bias);
				}
			}
			public enum NodeType {
				Input,
				Hidden,
				Output
			}
			public class Connection {
				public int From;
				public int To;
				public float Weight;
				public int Innovation;
				public bool Enabled = true;

				public Connection(int from, int to, float weight, int innovation) {
					From = from;
					To = to;
					Weight = weight;
					Innovation = innovation;
				}
				public Connection Copy() {
					Connection copy = new Connection(From, To, Weight, Innovation);
					copy.Enabled = Enabled;
					return copy;
				}
				public override string ToString() {
					return String.Format("({0}->{1} ; {2} {3} {4})\n", From, To, Weight, Innovation, Enabled);
				}
			}

			public override string ToString() {
				string output = "";

				foreach (Node node in nodes) {
					output += node.ToString();
				}
				output += "\n";
				foreach (Connection conn in connections) {
					output += conn.ToString();
				}

				return output;
			}

			public static int GIN = 0;
			public static int GetGIN(int count = 1) {
				GIN += count;
				return GIN;
			}
			public static List<(bool isNewNode, int A, int B, int GIN)> mutations = new List<(bool isNewNode, int A, int B, int GIN)>();
		}
	}
}