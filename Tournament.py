import glob
from Conn4Pit import Conn4Pit
from Conn4Net import Conn4Net
import multiprocessing as mp
from Conn4Game import Conn4Game

def get_participants(dir='models_conv/', age_delta=1):
	files = glob.glob(dir + 'best_*.hdf5')
	files = sorted(files[::age_delta] if age_delta < len(files) else files)
	matches = []

	for i in range(len(files) - 1):
		for j in range(i, len(files)):
			if i != j:
				matches.append((files[i], files[j]))

	return matches, files


results = {}
def log_results(result):
	net1, net2, score = result
	results[(net1, net2)] = score
	results[(net2, net1)] = 1 - score


def run_match(net1, net2, n_sims, n_games, rows=6, cols=7):
	print('Running match between {} and {}'.format(net1, net2))
	game = Conn4Game(rows=rows, cols=cols)
	input_dim = game.encoded_board_dim()
	output_dim = game.action_size()

	_net1 = Conn4Net(input_dim, output_dim, load=True, fname=net1)
	_net2 = Conn4Net(input_dim, output_dim, load=True, fname=net2)
	pit = Conn4Pit(_net1, _net2, n_sims, n_games, rows, cols)
	score = pit.run()
	return net1, net2, score


def run():
	n_cpu = max(1, mp.cpu_count() - 1)
	print('Running using {} cpus'.format(n_cpu))
	pairs, participants = get_participants()	

	pool = mp.Pool(n_cpu)
	for net1, net2 in pairs:
		pool.apply_async(run_match, args=(net1, net2, 30, 4, 6, 7), callback=log_results)
	
	pool.close()
	pool.join()	
	files_to_version = {p:int(p.split('_')[-1][:-5]) for i, p in enumerate(participants)}

	points = {}
	for pi in participants:
		for (p1, p2), score in results.items():
			if p1 == pi:
				points.setdefault(files_to_version[pi], []).append(score)

	points = {version: 1.0 * sum(score)/len(score) for version, score in points.items()}
	return points


if __name__ == '__main__':	
	points = run()
	version = sorted(points.keys())
	for v in version:
		print('{},{}'.format(v, points[v]))
