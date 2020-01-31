from Conn4Game import Conn4Game
from Conn4Net import Conn4Net
from MCTS import MCTS
import numpy as np
from random import choice

class  HumanAI(object):
	def __init__(self, net_file, human_side, rows=6, cols=7):
		self.rows = rows
		self.cols = cols
		self.human_side = human_side
		self.net_file = net_file


	def main(self):
		end, winner, _ = False, None, -1
		game = Conn4Game(self.rows, self.cols)
		input_dim = game.encoded_board_dim()
		output_dim = game.action_size()
		net = Conn4Net(input_dim, output_dim, load=True, fname=self.net_file)
		mcts = MCTS(net)
		NSIMS = 500
		TEMP = 0

		while not end:						
			if self.human_side == game.currentPlayer:
				print(game)				
				print('Enter your move (column):')
				move = int(input())				
				game.apply_action(move)
			else:				
				p = mcts.get_action_prob(game, NSIMS, temp=TEMP)
				move = np.argmax(p)
				game.apply_action(move)
				print('AI just played on column {}'.format(move))				
			
			end, winner, _ = game.game_ended(1)


		print(game)
		if winner == 1 and self.human_side == 1:
			print('You win!')
		elif winner == 1 and self.human_side == 2:
			print('You lose!')
		elif winner == 2 and self.human_side == 2:
			print('You win!')
		elif winner == 2 and self.human_side == 1:
			print('You lose!')
		else:
			print('Draw!')



if __name__ == '__main__':
	ai = HumanAI('models_conv/best_model6x7_43.hdf5', human_side=2)
	ai.main()
