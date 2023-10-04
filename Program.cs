﻿namespace NEAT {
	internal class Program {
		static void Main(string[] args) {
			List<float> results = new List<float>();
			List<int> generationsTaken = new List<int>();
			int successes = 0;
			for (int i = 0; i < 100; i++) {
				var myResults = NN.Train(XOR, cycles: 100, c4: 3);
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

		class NN {

			public List<Node> nodes = new List<Node>();
			public List<Connection> connections = new List<Connection>();
			public (int Inputs, int Outputs) size;

			public float fitness;
			public Species species;

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
				float mutWeight = 0.80f;
				float mutBias = 0.80f;
				if (chance(mutWeight)) {
					for (int i = 0; i < connections.Count; i++)
						connections[i].Weight += (float)(new Random().NextDouble() * 2 - 1);
				}
				if (chance(mutBias)) {
					for (int i = 0; i < nodes.Count; i++)
						nodes[i].Bias += (float)(new Random().NextDouble() * 2 - 1);
				}

				//Structural mutations
				float addConn = 0.05f;
				float addNode = 0.03f;
				if (chance(addConn)) {//Add connection
					Node node = nodes[new Random().Next(nodes.Count)];//Select random node
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
			}
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
					if (copy.nodes.Count > conn.To)
						copy.nodes[conn.To].Connections.Add(conn);

				return copy;
			}

			public static (List<NN>, int) Train(Func<NN, bool, float> func, int cycles = 1000, int generationSize = 100, float c1 = 1, float c2 = 1, float c3 = 0.4f, float c4 = 3) {
				//Hyperparameters:
				//c1: Excess distance; c2: Disjoint distance
				//c3: Weight distance; c4: Species distance cutoff
				float trainingCutoff = 15.5f;

				//Create population
				List<NN> population = new List<NN>();
				List<Species> species = new List<Species>();
				List<float> genusHistory = new List<float>();
				int generationsTaken = 0;

				for (int i = 0; i < generationSize; i++) {
					population.Add(new NN(3, 1));
					population[i].Mutate();
				}
				species.Add(new Species(population[0]));
				species[0].subPopulation = new List<NN>(population);

				//Train population
				for (int k = 0; k < cycles; k++) {

					//--Speciate--//
					//Remove empty species, reset count
					species.RemoveAll(x => x.Count == 0);

					for (int i = 0; i < species.Count; i++) {
						species[i].subPopulation.Clear();
						species[i].speciesFitness = 0;
					}

					//For each population, iterate through each species, if comparability is within threshhold, assign species number
					//If end is reached, create new species with it as example
					float NNdistance(NN a, NN b) {

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

						return disjoint * c1 / total + excess * c2 / total + weightDiff * c3;
					}

					for (int i = 0; i < population.Count; i++) {
						for (int j = 0; j < species.Count; j++) {

							if (NNdistance(population[i], species[j].representative) < c4) {
								population[i].species = species[j];
								species[j].subPopulation.Add(population[i]);
								break;
							} else if (j == species.Count - 1) {//No species found
								Species speci = new Species(population[i]);
								population[i].species = speci;
								speci.subPopulation.Add(population[i]);
								species.Add(speci);
								break;
							}
						}
					}

					//Update representatives
					//Each existing species is represented by a random genome inside the species from the previous generation.
					var rand = new Random();
					List<int> chosenReps = new List<int>();
					for (int i = 0; i < species.Count; i++)
						chosenReps.Add(rand.Next(species[i].Count));

					for (int i = 0; i < population.Count; i++) {
						int mySpecies = species.IndexOf(population[i].species);
						chosenReps[mySpecies]--;
						if (chosenReps[mySpecies] == 0)
							species[mySpecies].representative = population[i].Copy();
					}



					//--Assess--//
					List<float> fitnesses = new List<float>();
					float totalFitness = 0;
					
					for (int i = 0; i < population.Count; i++) {
						Species speci = population[i].species;
						float fitness = func(population[i], false);
						fitnesses.Add(fitness);
						fitness /= speci.Count;
						population[i].fitness = fitness;
						speci.speciesFitness += fitness;
						totalFitness += fitness;
					}

					//Species Stagnation
					//If a species had not improved fitness for 15 generations
					//Cull them, remove fitnesses from total fitnesses
					species.Sort((x, y) => y.speciesFitness.CompareTo(x.speciesFitness));
					for (int i = 0; i < species.Count; i++) {
						species[i].history.Add(species[i].speciesFitness);
						if (i > 2 && species[i].history.Count > 15 && species[i].speciesFitness - species[i].history[14] < 0.5f) {
							totalFitness -= species[i].speciesFitness;
							species.RemoveAt(i);
							i--;
						}
					}

					//Genus Stagnation
					//If top hasn't improved for 20 generations
					//Cull all but top two species
					int genusStagGens = 100;
					genusHistory.Add(fitnesses.Max());
					if (genusHistory.Count > genusStagGens)
						genusHistory.RemoveAt(0);
					if (genusHistory.Count >= genusStagGens && genusHistory[genusHistory.Count-1] - genusHistory[0] < 0.05f) {
						Console.WriteLine("Genus Stagnation!" + k);
						for (int i = 2; i < species.Count; i++) {
							totalFitness -= species[i].speciesFitness;
							species.RemoveAt(i);
							i--;
						}
						genusHistory.Clear();
					}

					//Break off if goal met
					if (k == cycles - 1 || fitnesses.Max() >= trainingCutoff) {
						Console.WriteLine("\n--Generations Taken: " + k);
						generationsTaken = k;
						break;//Return unaltered results when finished
					}


					//--Crossover and Mutate--//
					int SelectWeighted(int max) {
						float r = (float)new Random().NextDouble();
						float i = (float)(1 + Math.Sqrt(1 + 4 * r * max * (max - 1))) / 2;
						return Math.Clamp(max - (int)Math.Floor(i) - 1, 0, max-1);
					}

					//Assign offspring targets for each species
					var popCounts = new int[species.Count];
					for (int i = 0; i < popCounts.Length; i++)
						popCounts[i] = (int)Math.Floor(generationSize * species[i].speciesFitness / totalFitness);
					int popError = generationSize - popCounts.Sum();
					for (int i = 0; i < Math.Min(popError, popCounts.Length); i++)
						popCounts[i]++;

					string myOut = "";
					for (int i = 0; i < popCounts.Length; i++)
						myOut += popCounts[i] + ", ";

					for (int j = 0; j < species.Count; j++) {
						//Sort and cull population by fitness
						Species speci = species[j];
						speci.subPopulation.Sort((x, y) => y.fitness.CompareTo(x.fitness));
						List<NN> survivors = new List<NN>();
						List<NN> offspring = new List<NN>();
						if (speci.Count > 4)
							offspring.Add(speci.subPopulation[0]);
						int targ = (int)Math.Ceiling(speci.Count / 2f);
						int formerCount = speci.Count;
						for (int i = 0; i < targ; i++) {
							int index = SelectWeighted(speci.Count);
							survivors.Add(speci.subPopulation[index]);
							speci.subPopulation.RemoveAt(index);
						}
						
						speci.subPopulation.Clear();

						//Crossover population, replace with offspring
						int newPopCount = popCounts[j];
						List<NN> crossPool = new List<NN>(survivors);

						while (offspring.Count < newPopCount) {
							if (offspring.Count < newPopCount * 0.25f || survivors.Count == 1) {//If only one member...
								var clone = survivors[new Random().Next(survivors.Count)].Copy();//Use mitosis
								clone.Mutate();
								offspring.Add(clone);
								continue;
							} else if (survivors.Count == 0) {
								Console.WriteLine("Populating failed, unborn children: " + newPopCount);
								break;
							}

							//Otherwise...
							var a = crossPool[new Random().Next(crossPool.Count)];
							crossPool.Remove(a);
							var b = crossPool[new Random().Next(crossPool.Count)];
							crossPool.Remove(b);
							offspring.Add(NN.Crossover(a, b));
							if (crossPool.Count < 2)//If not enough...
								crossPool.AddRange(survivors);//Refill crossing population
						}

						speci.subPopulation.AddRange(offspring);
					}

					population.Clear();
					for (int i = 0; i < species.Count; i++)
						population.AddRange(species[i].subPopulation);

					//Mutate all networks
					foreach (NN nn in population) {
						nn.Clear();
						nn.Mutate();
					}


					if (k % (cycles/10) == 0)
							Console.WriteLine(String.Format("k: {0,3}  AvrF: {1,3:N2}  MaxF: {2,3:N2}  Species: {3}", k, fitnesses.Average(), fitnesses.Max(), species.Where(x => x.Count != 0).Count()));
				}

				for (int i = 0; i < population.Count; i++)
					population[i].fitness = func(population[i], false);
				population.Sort((x, y) => y.fitness.CompareTo(x.fitness));

				return (population, generationsTaken);
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

						if (connection.From >= nodes.Count)
							continue;

						Node prevNode = nodes[connection.From];
						float weight = connection.Weight;
						if (connection.From != Index)
							sum += prevNode.Value * weight;
						else
							sum += State * weight;
					}
					State = sum;
					Value = 1 / (1 + (float)Math.Exp(-4.9f*State));
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
			public class Species {
				public NN representative;
				public List<NN> subPopulation = new List<NN>();
				public float speciesFitness;
				/*public float speciesFitness {
					get { return history[0]; }
					set {
						history.Insert(0, value);
						if (history.Count > 20)
							history.RemoveAt(20);
					}
				}*/
				public List<float> history = new List<float>();// { 0 };
				public int Count {
					get { return subPopulation.Count; }
				}

				public Species() { }
				public Species(NN Representative) {
					representative = Representative.Copy();
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