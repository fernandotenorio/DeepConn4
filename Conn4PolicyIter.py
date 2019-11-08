from Conn4Pit import Conn4Pit
from Conn4Game import Conn4Game
from Conn4Net import Conn4Net
from MCTS import MCTS
import numpy as np
from collections import deque
from random import sample

class PolicyIter(object):
	def __init__(self, nnet, n_iters, n_episodes, n_sims, n_games_pit, win_eps=0.55, batch_size=256, rows=6, cols=7):
		self.n_iters = n_iters
		self.n_episodes = n_episodes
		self.n_sims = n_sims
		self.n_games_pit = n_games_pit		
		self.win_eps = win_eps
		self.current_nn = nnet
		self.best_nn = self.current_nn.clone()		
		self.memory = []
		self.batch_size = batch_size
		self.rows = rows
		self.cols = cols
		self.START_VERSION = 90


	def policy_iter(self):
		for i in range(self.n_iters):			
			for e in range(self.n_episodes):				
				self.memory+= self.execute_episode()
			
			x = []
			p = []
			r = []

			for xi, pi, ri in self.memory:
				x.append(xi)
				p.append(pi)
				r.append(ri)
			
			x = np.array(x)			
			p = np.array(p)
			r = np.array(r).reshape(x.shape[0], 1)			

			# trains current nnet
			self.current_nn.model.fit(x, [p, r], batch_size=self.batch_size, epochs=1, verbose=1)			

			# reset memory
			self.memory = []

			# pit the two
			win_rate = Conn4Pit(self.current_nn, self.best_nn, n_sims=self.n_sims, n_games=self.n_games_pit, rows=self.rows, cols=self.cols).run()
		 
			if win_rate >= self.win_eps:				
				self.best_nn.model.set_weights(self.current_nn.model.get_weights())
				self.best_nn.save('models_conv/best_model6x7_{}.hdf5'.format(i + self.START_VERSION))
			

			print('{},{}'.format(i, win_rate))		


	def execute_episode(self):
		examples = []
		game = Conn4Game(self.rows, self.cols)
		dim = len(game.board)
		mcts = MCTS(self.best_nn)
		move = 0

		while True:
			temp = 1 #1 if move < 10 else 0.1						
			p_a = mcts.get_action_prob(game, self.n_sims, temp=temp)
			s_h = game.hash()
			s_x = game.encode_board()

			# filter nonsense moves
			#p_a = [p_a[i] if game.board[int(i/dim)][i%dim] == -1 else 0 for i in range(len(p_a))]
			#p_a = [pi/sum(p_a) for pi in p_a]
			# filter nonsense moves
			
			examples.append([s_x, p_a, None])					
			a = np.random.choice(range(len(p_a)), p=p_a)					
			game.apply_action(a)			
			end, winner, r = game.game_ended(game.currentPlayer)
			
			if end:				
				# does the first player won?
				if winner == 1:
					m = -1
				elif winner == 2:
					m = 1
				else:
					m = 0
				for ex in examples:
					ex[-1] = m * r
					m*= -1

				return examples
			move+= 1



if __name__ == '__main__':
	rows = 6
	cols = 7
	game = Conn4Game(rows, cols)
	input_dim = game.encoded_board_dim()
	output_dim = game.action_size()	
	net1 = Conn4Net(input_dim, output_dim, load=True, fname='models_conv/best_model6x7_89.hdf5')
	p = PolicyIter(nnet=net1, n_iters=100, n_episodes=50, n_sims=30, n_games_pit=30, batch_size=256, win_eps=0.55, rows=rows, cols=cols)
	p.policy_iter()	

	
