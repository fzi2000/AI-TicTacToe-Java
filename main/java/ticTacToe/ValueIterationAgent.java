package ticTacToe;


import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=10;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		this.policy = p; 
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 * This function updates values based on Bellman Ford Equation
	 *
	 */
	public void iterate()
	{		
//		V(s)=max_a sum_target=[T(s,a,target) * (R+gamma*V(target) )]	
		double sum = 0.0;
		List<Double> stateValues = new ArrayList<Double>();	
		
		//iterate k times
		for(int i = 0 ; i < k ; i++)	{
			//for each game state 
			for(Game g : valueFunction.keySet())	{
				if(g.isTerminal())	{
					valueFunction.put(g, 0.0);
					continue;
				}
				
				//Maximum q value
				double maxQValue = Double.NEGATIVE_INFINITY;
				
//				For each possible move in the game state
		            for (Move move : g.getPossibleMoves()) {
		                double qValue = calculateQValue(g, move);
		                List<TransitionProb> transitions = mdp.generateTransitions(g, move);
		        	    //For all transitions from current state g and Move m
		        	    for (TransitionProb transition : transitions) {
		        	        sum += transition.prob * (transition.outcome.localReward + (discount * valueFunction.get(transition.outcome.sPrime)));
		        	    }
		        	
//		        	    If qValue is greater than maxQValue, assign maxQValue as qValue
		                if(qValue>maxQValue) {
		                	maxQValue=qValue;
		                }
//		                maxQValue = Math.max(maxQValue, qValue);
		            }
//		            Update the valueFunction with maXQValue
		            valueFunction.put(g, maxQValue);
		        }
		        valueFunction.putAll(valueFunction);
		}
	}
	
	
		/* YOUR CODE HERE
		 */

	
//	HElper functionn that calculates Q-Value for a given state and move using Bellman Equation
	private double calculateQValue(Game state, Move move) {
	    double sum = 0.0;
	    List<TransitionProb> transitions = mdp.generateTransitions(state, move);
//		For all transitions possible, calculate sum using transition probabilities, discount factor & valueFunction
	    for (TransitionProb transition : transitions) {
	        sum += transition.prob * (transition.outcome.localReward + (discount * valueFunction.get(transition.outcome.sPrime)));
	    }

	    return sum;
	}
	
	
	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		/*
		 * YOUR CODE HERE
		 */

		Set<Game> states = valueFunction.keySet();
		Policy p = new Policy();

		
		for(Game g : states) 
//			If gameState is a terminal, continue ahead
		{
			if(g.isTerminal())		{
				continue;
				}
			List<Double> stateValues = new ArrayList<Double>();
//			For each possible moves in the game state
			for(Move m : g.getPossibleMoves())	{
				double sum = 0.0;
				List<TransitionProb> probs = mdp.generateTransitions(g, m);
//				For each transitions possible  in the current state
				for(int j = 0; j < probs.size(); j++)	{
					sum = sum + (probs.get(j).prob * (probs.get(j).outcome.localReward + (discount * valueFunction.get(probs.get(j).outcome.sPrime))));
					}
				stateValues.add(sum);
//				Select move with max reward
				if(sum == Collections.max(stateValues)) {
					p.policy.put(g, m);
				}
			}
				stateValues.clear();
		}
		return p;
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();

		

	
    }

   	
	
}
