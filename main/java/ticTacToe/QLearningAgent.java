package ticTacToe;

import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=50000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 * @throws IllegalMoveException 
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}

	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	

	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 * @throws IllegalMoveException 
	 */
	public QLearningAgent() 
	{
		this(new RandomAgent(), 0.1, 50000, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 * @throws IllegalMoveException 
	 */

	public void train()
	{
		/* 
		 * This method trains the Q-Learning agent. It iterates through the episodes
		 * and selects a move by epsilon greedy policy
		 * it then executes the selected move and updates the Q-value
		 * 
		 * YOUR CODE HERE
		 */
//		Variable to store current and new Q Value
		double currentQValue=0.0;
		double newQValue=0.0;
		
//		Iterate through the episodes 
		for(int i=0; i<numEpisodes; i++) {
			 env.reset();
			while(!this.env.isTerminal()) {
//				Get the current game state
				Game currentGameState = this.env.getCurrentGameState();
//				If the current state is a terminal state, skip it
				if(currentGameState.isTerminal()){
					continue;
				}
				
				Move move = epsilonFunction(currentGameState);
				try {
					
					Outcome outcome = this.env.executeMove(move);
					
//					Calculate new Q Value using Q learning
					currentQValue = this.qTable.getQValue(outcome.s, outcome.move);
					newQValue=updateQValue(currentQValue, outcome.localReward, outcome.sPrime);
//					Update the Q table with the Q value 
					this.qTable.addQValue(outcome.s, outcome.move, newQValue);
					
				} catch (IllegalMoveException e) {
					e.printStackTrace();
				}				
			}
			// Reset the environment after one iteration
		this.env.reset();
		}				
				
		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}

	
	/**
	 * Helper function to update the Q-Value
	 * Based on the current Q value, local reward and target state after the 
	 * opponent's move, it calculates and returns the updated Q value
	 * @param currentQValue
	 * @param localReward
	 * @param sPrime
	 * @return
	 */
	private double updateQValue(double currentQValue, double localReward, Game sPrime) {
	    double maxQValue = maximumQvalue(sPrime);
	    return (1 - this.alpha) * currentQValue + this.alpha * (localReward + this.discount * maxQValue);
	}
	
	
	/**
	 * Epsilon-greedy policy to select a move
	 * @param game
	 * @return
	 */
	private Move epsilonFunction(Game game) {
		Random random = new Random();
		double qvalue = 0.0;
		double maxValue = Integer.MIN_VALUE;
		List<Move> movesList = game.getPossibleMoves();	
		Move move =null;
		
//		Choose a random move
		if(random.nextDouble() < epsilon && !movesList.isEmpty()) {														
				return movesList.get(random.nextInt(movesList.size()));			
		}
//		Choose the move with the maximum Q value
		else {		
			for (Move move1 : movesList){
				qvalue = qTable.getQValue(game, move1);
				if (qvalue>=maxValue) {
					maxValue = qvalue;
					move = move1;
				}
			}
			return move; 
		}			
	}
	

	/**
	 * 
	 * function to find the maximum Q value for a given state
	 * @param gPrime
	 * @return
	 */
	private Double maximumQvalue(Game gPrime) {
		if(gPrime.isTerminal()){
			return 0.0;
		}
		double maxValue = Double.MIN_VALUE;			
		for (Move move : gPrime.getPossibleMoves()){			
			double qvalue = this.qTable.getQValue(gPrime, move);
//			If the qValue is greater than maxValue, set the maxValue as QValue
			if (qvalue>maxValue) {
				maxValue = qvalue;
			}
		}
		return maxValue;
	}
	
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * This method extracts the policy from the learned Q-values
	 * For each non-terminal state, it chooses the move with maximum Q-value and adds to the policy 
	 * Then it returns the extracted policy
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy() {
		/* 
		 * YOUR CODE HERE
		 */	
	    Policy policy = new Policy();
//	    Get all the states in the Q-table
	    Set<Game> states = qTable.keySet();

//	    If the state is a terminal state, skip it
	    for (Game state : states) {
	        if (state.isTerminal()) {
	            continue;
	        }

	        Move bestMove = null;
	        double maxQValue = Double.NEGATIVE_INFINITY;	        
//	        Find the move with maximum Q-value for the current state 
	        for (Move move : state.getPossibleMoves()) {
	            double qValue = qTable.getQValue(state, move);	            
	            if (qValue > maxQValue) {
	                maxQValue = qValue;
	                bestMove = move;
	            }
	        }
//	        Add the best move to the policy
	        if (bestMove != null) {
	            policy.policy.put(state, bestMove);
	        }
	    }
//		Return the policy
	    return policy;
	}

	
	/**
	 * Main function
	 * @param a
	 * @throws IllegalMoveException
	 */
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
				
	}
	
}

