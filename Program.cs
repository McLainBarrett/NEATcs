//https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
//https://towardsdatascience.com/neuro-evolution-on-steroids-82bd14ddc2f6
//https://arxiv.org/pdf/1703.00548.pdf

namespace NEAT {
	internal class Program {
		static void Main(string[] args) {
			XORproblem();
			//NE ne = new NE(new int[] { 2, 2, 1});

		}

		static void XORproblem() {
			//Create population
			List<NE> population = new List<NE>();
			for (int i = 0; i < 20; i++) {
				population.Add(new NE(new int[] { 3, 3, 1}));
				population[i].Mutate();
			}

			//Train population
			for (int i = 0; i < 1200; i++) {
				List<float> fitnesses = new List<float>();
				//Assess
				for (int j = 0; j < population.Count; j++) {
					var fit = XOR(population[j]);
					population[j].fitness = fit;
					//Console.WriteLine(fit);
					fitnesses.Add(fit);
				}

				//Record average fitness
				System.Console.WriteLine(String.Format("{0}: {1} \t {2}", i+1, fitnesses.Average(), fitnesses.Max()));

				//--Crossover, Mutate

				//Sort population by fitness
				var temp = new List<NE>(population);
				population.Sort((x, y) => fitnesses[temp.IndexOf(x)].CompareTo(fitnesses[temp.IndexOf(y)]));
				population.Reverse();

				Console.WriteLine(XORv(population[0]));
				
				//Select using ranking
				var survivors = new List<NE>();
				for (int j = 0; j < population.Count / 2; j++) {
					survivors.Add(population[j]);
					var clone = population[j].Copy();
					clone.Mutate();
					survivors.Add(clone);
				}

				population.Clear();
				population.AddRange(survivors);

				//Mutate all networks
				foreach (NE nn in population)
					nn.Mutate();
			}
			System.Console.WriteLine(population[0]);
		}

		static float XOR(NE nn) {
			float fitness = 0;
			/*	00->0
				01->1
				10->1
				11->0  */

			fitness += (float)Math.Abs(0 - nn.Activate(new List<float>() {0, 0, 1})[0]);
			fitness += (float)Math.Abs(1 - nn.Activate(new List<float>() {0, 1, 1})[0]);
			fitness += (float)Math.Abs(1 - nn.Activate(new List<float>() {1, 0, 1})[0]);
			fitness += (float)Math.Abs(0 - nn.Activate(new List<float>() {1, 1, 1})[0]);
			return -fitness;
		}

		static float XORv(NE nn) {
			float fitness = 0;
			/*	00->0
				01->1
				10->1
				11->0  */

			fitness += (float)Math.Abs(0 - nn.Activate(new List<float>() { 0, 0, 1 })[0]);
			fitness += (float)Math.Abs(1 - nn.Activate(new List<float>() { 0, 1, 1 })[0]);
			fitness += (float)Math.Abs(1 - nn.Activate(new List<float>() { 1, 0, 1 })[0]);
			fitness += (float)Math.Abs(0 - nn.Activate(new List<float>() { 1, 1, 1 })[0]);

			Console.WriteLine("0, 0 -> " + nn.Activate(new List<float>() { 0, 0, 1 })[0]);
			Console.WriteLine("0, 1 -> " + nn.Activate(new List<float>() { 0, 1, 1 })[0]);
			Console.WriteLine("1, 0 -> " + nn.Activate(new List<float>() { 1, 0, 1 })[0]);
			Console.WriteLine("1, 1 -> " + nn.Activate(new List<float>() { 1, 1, 1 })[0]);

			return -fitness;
		}

		class NE {//Normal neuralevolutionary network
			public List<Node> nodes = new List<Node>();
			public List<Connection> connections = new List<Connection>();
			int[] size;
			public float fitness;
			public NE(int[] Size) {
				size = Size;
				//Add nodes
				for (int i = 0; i < size.Length; i++) {
					for (int j = 0; j < size[i]; j++) {
						NodeType nt = i == 0 ? NodeType.Input : (i == size.Length - 1 ? NodeType.Output : NodeType.Hidden);
						nodes.Add(new Node(nodes.Count, 0, nt));
					}
				}

				//Add connections
				int lastLayer = 0;
				for (int i = 1; i < size.Length; i++) {
					for (int j = 0; j < size[i]; j++) {
						Node node = nodes[lastLayer + size[i-1] + j];
						for (int k = lastLayer; k < lastLayer+size[i-1]; k++) {
							node.Connections.Add(new Connection(k, node.Index, 1, 0));
						}
					}
					lastLayer += size[i-1];
				}
			}

			public List<float> Activate(List<float> input) {
				List<float> output = new List<float>();

				for (int i = 0; i < nodes.Count; i++) {
					Node node = nodes[i];
					if (node.Type == NodeType.Input)
						node.Value = input[i] + node.Bias;
					else
						node.Activate(nodes);

					if (node.Type == NodeType.Output)
						output.Add(node.Value);
				}
				return output;
			}

			private static bool chance(float probability) {
				return new Random().NextDouble() < probability;
			}
			public void Mutate() {
				float prob = 0.05f;
				float range = 3;
				//Mutate weights and biases
				for (int i = 0; i < nodes.Count; i++)
					if (chance(prob))
						nodes[i].Bias += (float)(new Random().NextDouble() * range * 2 - range);
				for (int i = 0; i < connections.Count; i++)
					if (chance(prob))
						connections[i].Weight += (float)(new Random().NextDouble() * range * 2 - range);
			}
			/*
			public static NE Crossover(NE parentA, NE parentB) {
				//ParentA is *always* more fit than parentB
				if (parentA.fitness < parentB.fitness) {
					NE tempNN = parentA;
					parentA = parentB;
					parentB = tempNN;
					float tempFit = parentA.fitness;
					parentA.fitness = parentB.fitness;
					parentB.fitness = tempFit;
				}
				
				NE offspring = new NE(0, 0);

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
					offspring.nodes[conn.To].Connections.Add(conn);
				}
				offspring.size = (parentA.size.Inputs, parentA.size.Outputs);

				return offspring;
			}*/

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
					float range = 10;
					Value = range / (1 + (float)Math.Exp(-State)) - range/2;// 1 / (1 + (float)Math.Exp(-State));
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

			public NE Copy() {
				NE copy = new NE(size);

				for (int i = 0; i < nodes.Count; i++)
					copy.nodes[i].Bias = nodes[i].Bias;

				for (int i = 0; i < connections.Count; i++)
					copy.connections[i].Weight = connections[i].Weight;

				return copy;
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

		class NN {

			public List<Node> nodes = new List<Node>();
			public List<Connection> connections = new List<Connection>();
			public (int Inputs, int Outputs) size;
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
			public static NN Crossover(NN parentA, NN parentB, float fitnessA, float fitnessB) {
				if (fitnessA < fitnessB) {
					NN tempNN = parentA;
					parentA = parentB;
					parentB = tempNN;
					float tempFit = fitnessA;
					fitnessA = fitnessB;
					fitnessB = tempFit;
				}
				return Crossover(parentA, parentB);
			}
			public static NN Crossover(NN parentA, NN parentB) {
				//ParentA is *always* more fit than parentB
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
					offspring.nodes[conn.To].Connections.Add(conn);
				}
				offspring.size = (parentA.size.Inputs, parentA.size.Outputs);

				return offspring;
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