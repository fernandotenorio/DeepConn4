from copy import deepcopy
from random import choice, seed
import numpy as np

class Conn4Game(object):
	# variations 9×7, 10×7, 8×8
	def __init__(self, rows=6, cols=7):
		self.rows = rows
		self.cols = cols
		self.currentPlayer = 1

		# rows 0 is the topmost row, 1 first player, 2 sec player, 0 empty
		self.board = [[0]*cols for _ in range(rows)]

		# height of the piles, bottom to top
		self.height = [0 for _ in range(cols)]

		# (row, col)
		Conn4Game.board_directions = ((0, 1), (1, 0), (1, 1), (-1, 1))

		# last move (row, col)
		self.last_move = None


	def get_actions(self):
		return [col for col in range(self.cols) if self.board[0][col] == 0]

	# def get_actions(self):
	# 	cols = [col for col in range(self.cols) if self.board[0][col] == 0]
	# 	rows = [self.rows - self.height[col] - 1 for col in cols]
	# 	return [self.cols * r + c for (r, c) in zip(rows, cols)]

	
	def apply_action(self, col):
		assert self.board[0][col] == 0, 'Illegal move attempt'
		idx = self.rows - self.height[col] - 1
		self.height[col]+= 1
		self.board[idx][col] = self.currentPlayer
		self.last_move = (idx, col)
		self.currentPlayer = 2 if self.currentPlayer == 1 else 1

	# def apply_action(self, idx):		
	# 	row = int(idx/self.cols)
	# 	col = idx % self.cols
	# 	assert self.board[row][col] == 0, 'Illegal move attempt'
	# 	assert self.board[0][col] == 0, 'Illegal move attempt'		
	# 	self.height[col]+= 1
	# 	self.board[row][col] = self.currentPlayer
	# 	self.last_move = (row, col)
	# 	self.currentPlayer = 2 if self.currentPlayer == 1 else 1


	def action_size(self):
		return 7#self.cols * self.rows


	def game_ended(self, player):
		if self.last_move is None:
			return False, None, 0

		l_row = self.last_move[0]
		l_col = self.last_move[1]
		board = self.board
		opp = 1 if self.currentPlayer == 2 else 2

		reward = -1#1 if opp == player else -1

		# check vertical
		if l_row + 3 < self.rows:
			if board[l_row][l_col] == opp and board[l_row + 1][l_col] == opp and board[l_row + 2][l_col] == opp and board[l_row + 3][l_col] == opp:
				return True, opp, reward

		# check horizontal
		col_min = max(l_col - 3, 0)
		col_max =  min(l_col + 4, self.cols)

		for c in range(col_min, col_max):
			try:
				if board[l_row][c] == opp and board[l_row][c + 1] == opp and board[l_row][c + 2] == opp and board[l_row][c + 3] == opp:
					return True, opp, reward
			except:
				pass

		# diag
		row_min = max(l_row - 3, 0)
		row_max = min(self.rows, l_row + 4)

		for dy, dx in [(1, 1), (1, -1)]:			
			for r in range(row_min, row_max):
				for c in range(col_min, col_max):
					try:
						if r + dy < 0 or r + 2*dy < 0 or r + 3*dy < 0:
							continue
						if c + dx < 0 or c + 2*dx < 0 or c + 3*dx < 0:
							continue
						if board[r][c] == opp and board[r + dy][c + dx] == opp and board[r + 2*dy][c + 2*dx] == opp and board[r + 3*dy][c + 3*dx] == opp:
								return True, opp, reward
					except:
						pass

		# No winner so far and board is full
		if min(self.height) == self.rows:
			return True, None, 1e-6
		else:
			return False, None, 0



	def game_ended2(self):
		opp = 1 if self.currentPlayer == 2 else 2
		board = self.board

		for dy, dx in Conn4Game.board_directions:
			for r in range(self.rows):
				for c in range(self.cols):
					try:
						if r + dy < 0 or r + 2*dy < 0 or r + 3*dy < 0:
							continue
						if board[r][c] == opp and board[r + dy][c + dx] == opp and board[r + 2*dy][c + 2*dx] == opp and board[r + 3*dy][c + 3*dx] == opp:
							return True, opp, -1
					except:
						pass

		# No winner so far and board is full
		if min(self.height) == self.rows:
			return True, None, 0
		else:
			return False, None, 0


	def encoded_board_dim(self):
		return (self.rows, self.cols, 3)										


	def encode_board(self):
		p1 = [[0]*self.cols for _ in range(self.rows)]
		p2 = [[0]*self.cols for _ in range(self.rows)]
		player_flag = 1 if self.currentPlayer == 1 else 0
		player = [[player_flag]*self.cols for _ in range(self.rows)]
				
		for r in range(self.rows):
			for c in range(self.cols):				
				if self.board[r][c] == 1:
					p1[r][c] = 1
				elif self.board[r][c] == 2:
					p2[r][c] = 1
		
		return np.array([p1, p2, player]).reshape(self.rows, self.cols, 3)


	def hash(self):
		b = [v for row in self.board for v in row] + [self.currentPlayer]
		return tuple(b)


	def __str__(self):
		s = ''
		for r in range(self.rows):
			for c in range(self.cols):
				s+= str(self.board[r][c]) + ' '
			s+= '\n'
		return s


	def clone(self):
		game = Conn4Game(self.rows, self.cols)
		game.currentPlayer = self.currentPlayer
		game.board = deepcopy(self.board)
		game.height = deepcopy(self.height)
		game.last_move = deepcopy(self.last_move)
		return game


	@staticmethod
	def random_game(rows=6, cols=7):
		game = Conn4Game(rows, cols)
		gameover = False
		winner = None

		while not gameover:
			moves = game.get_actions()				
			move = choice(moves)		
			game.apply_action(move)			
			gameover, winner, _ = game.game_ended()
				
		#print(str(game) + 'Winner is {}\n'.format(winner))
		return winner


if __name__ == '__main__':
	wins = {1:0, 2:0, 'draw': 0}
	seed(4)
	for _ in range(5000):
		winner = Conn4Game.random_game(rows=6, cols=7)
		if winner is not None:
			wins[winner]+= 1
		else:
			wins['draw']+= 1
	print('Player 1 win rate: {:.2f}%'.format(100*wins[1]/sum(wins.values())))
	print('Player 2 win rate: {:.2f}%'.format(100*wins[2]/sum(wins.values())))
	print('Draw rate: {:.2f}%'.format(100*wins['draw']/sum(wins.values())))
	





