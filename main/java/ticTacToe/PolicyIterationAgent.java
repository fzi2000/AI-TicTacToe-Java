package ticTacToe;


import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues = new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
				
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
//		this.policy = new Policy(); 
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 *  
	 *  Initialise random policy
	 */
	public void initRandomPolicy()
	{
		/*
		 * YOUR CODE HERE
		 */
		
//		Iterate through all game states in the policyValues map
		for(Game gameState: policyValues.keySet()) {
			Random random = new Random();	
			if (!gameState.isTerminal()) {   
//				Create a list of all possible moves for current game state. 
				List<Move> moves = gameState.getPossibleMoves();
				
				for (int i=0;i<moves.size();i++) {

//				Create randomMove from list of possible moves
				Move randomMove = moves.get(random.nextInt(moves.size())); 
//				Assign the random move to the current game state
				this.curPolicy.put(gameState, randomMove);  
				}				
			}
		}	
	}
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 * 
	 * 
	 * Initialise the random policy 
	 * Evaluate policy
	 * Improve policy
	 * Same as value iteration but we do not maximize  over actions
	 */
	protected void evaluatePolicy(double delta)
	{
		/* YOUR CODE HERE */
			double curStateValue=0; //current state's calculated value
			double prevStateValue; //previous state's  value
			double maxChange = Double.NEGATIVE_INFINITY;
			int stateCount =0;  // counter for how many times the while loop runs or number of times it evaluates the state value
			Move currentMove; //current state move
			
//			Iterate until maxChange is less than delta
			while(true) {
				//Iterates over all states in current policy			
				for (Game gameState: curPolicy.keySet()) {
					curStateValue = 0;//cur state value
					currentMove = curPolicy.get(gameState);
					//check for improvements 
					prevStateValue = policyValues.get(gameState);			
					
//					Calculate current state value using the formula
					for(TransitionProb moveProb: mdp.generateTransitions(gameState, currentMove)) {                                                 //perform the Q*(s,a) = sum{T(s,a,s')}
						curStateValue += (moveProb.prob * (moveProb.outcome.localReward + (discount * policyValues.get(moveProb.outcome.sPrime))));
					}	

						//update state value in policy values map
						policyValues.put(gameState, curStateValue);	
						 maxChange = Math.abs(prevStateValue - curStateValue);
					} 
				
				 if (maxChange <= delta) {
			            break;
			        }
				} 							
	}
		
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		/* YOUR CODE HERE */
		 boolean isPolicyImproved = false;

		//iterate over states in current policy
		    for (Game gameState : curPolicy.keySet()) {
		        double previousValue = policyValues.get(gameState);

		      //possible moves in cur state
		        for (Move possibleMove : gameState.getPossibleMoves()) {
		        	double calculatedValue = 0;					
					//for transition probability 
					for (TransitionProb mvProb : mdp.generateTransitions(gameState, possibleMove)){
						//calculate q value
						calculatedValue += (mvProb.prob * (mvProb.outcome.localReward + discount * policyValues.get(mvProb.outcome.sPrime)));
						
					}
					//improve policy
		            if (calculatedValue > previousValue) {
		                // Update policy and value if improvement is found
		                curPolicy.replace(gameState, possibleMove);
		                //update value of state
		                policyValues.replace(gameState, calculatedValue);
		                isPolicyImproved = true;
		            }
		        }
		    }
		    return isPolicyImproved;			   		
	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		/* YOUR CODE HERE */				
		this.evaluatePolicy(delta);
		
		while (this.improvePolicy()) { //loops until no changes
			this.evaluatePolicy(delta);
		}
		super.policy = new Policy(curPolicy);
	}



	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
