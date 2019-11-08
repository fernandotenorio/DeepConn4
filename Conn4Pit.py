from Conn4Game import Conn4Game
from Conn4Net import Conn4Net
from MCTS import MCTS
import numpy as np
from random import choice

class Conn4Pit(object):
	def __init__(self, net1, net2, n_sims, n_games, rows, cols):
		self.net1 = net1
		self.net2 = net2
		self.n_games = n_games
		self.n_sims = n_sims
		self.rows = rows
		self.cols = cols


	def run(self):
		mcts1 = MCTS(self.net1)
		mcts2 = MCTS(self.net2)

		score = self.play_net1_net2(int(self.n_games/2), mcts1, mcts2)
		score+= self.play_net2_net1(int(self.n_games/2), mcts1, mcts2)
		return 1.0 * score/self.n_games


	def play_net1_net2(self, ngames, mcts1, mcts2):		
		wins_net1 = 0
		draws = 0

		for _ in range(ngames):
			game = Conn4Game(self.rows, self.cols)
			end, winner, _ = False, None, -1

			while not end:
				player = self.net1 if game.currentPlayer == 1 else self.net2				
				mcts = mcts1 if player == self.net1 else mcts2
				p = mcts.get_action_prob(game, self.n_sims, temp=0)	
				a = np.argmax(p)				
				game.apply_action(a)			
				end, winner, _ = game.game_ended(1)

			if winner == 1:
				wins_net1+= 1			
			elif winner is None:
				draws+= 1

		return wins_net1 + draws * 0.5


	def play_net2_net1(self, ngames, mcts1, mcts2):		
		wins_net1 = 0
		draws = 0

		for _ in range(ngames):
			game = Conn4Game(self.rows, self.cols)
			end, winner, _ = False, None, -1

			while not end:
				player = self.net1 if game.currentPlayer == 2 else self.net2				
				mcts = mcts1 if player == self.net1 else mcts2
				p = mcts.get_action_prob(game, self.n_sims, temp=0)	
				a = np.argmax(p)		
				game.apply_action(a)			
				end, winner, _ = game.game_ended(1)

			if winner == 2:
				wins_net1+= 1			
			elif winner is None:
				draws+= 1

		return wins_net1 + draws * 0.5
		
		
if __name__ == '__main__':
	rows = 6
	cols = 7
	game = Conn4Game(rows, cols)
	input_dim = game.encoded_board_dim()
	output_dim = game.action_size()
	
	net1 = Conn4Net(input_dim, output_dim, load=True, fname='models/best_model6x7_98.hdf5')
	net2 = Conn4Net(input_dim, output_dim, load=True, fname='models/best_model6x7_80.hdf5')

	print(Conn4Pit(net1, net2, 50, 300, rows, cols).run())
		