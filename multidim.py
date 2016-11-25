#coding:utf-8
import numpy
import os,sys,time
import json,csv
numpy.random.seed(1337)

class Delay(object):
	def __init__(self):
		pass

	def train(self, train_seq, test_seq, superparams, initparams,max_iter=0,max_outer_iter=100):

		model = {}
		model['superparams'] = superparams
		I = len(train_seq)
		K = len(train_seq[0]['features'])
		P = len(train_seq[0]['varying'])
		M = 2
		sigma = superparams['sigma']
		model['beta'] = numpy.random.random((M,K))
		model['alpha'] = numpy.random.random((M,M,P))
		model['Gamma'] = numpy.random.random((M**2,I))
		model['theta'] = initparams['theta']
		model['w'] = initparams['w']
		model['b'] = initparams['b']
		model['M'] = M
		model['K'] = K
		model['I'] = I

		reg_alpha = numpy.zeros(model['Gamma'].shape)
		reg_beta = numpy.zeros(model['beta'].shape)

		init_time = time.time()
		init_clock = time.clock()
		iteration = 1
		outer_times = 1

		L_converge = False
		while not L_converge:
			Q_converge = False
			while not Q_converge:
				LL = 0.0
				Gamma = numpy.array(model['Gamma'])
				beta = numpy.array(model['beta'])
				alpha = numpy.array(model['alpha'])
				intrinsic1 = numpy.zeros((M,K))
				intrinsic2 = numpy.zeros((M,K))

				for i in range(I):
					T = train_seq[i]['times'][-1] + 0.01
					times = numpy.array(train_seq[i]['times'],dtype=float)
					dims = numpy.array(train_seq[i]['dims'],dtype=int)
					features = numpy.array(train_seq[i]['features'],dtype=float)
					v = numpy.array(train_seq[i]['varying'],dtype=float)
					N = len(times)
			
					B = sigma * reg_alpha[:,i]
					B = B.reshape((M,M))
					C = numpy.zeros(B.shape)
					D = numpy.zeros(model['beta'].shape)
					init_intensity = numpy.dot(model['beta'], features)
					init_intensities = model['beta'] * numpy.tile(features,(M,1))
					gamma = model['Gamma'][:,i].reshape(M,M)
					theta = model['theta']
					w = model['w']
					b = model['b']
			
					for j in range(0,N):
						t_j = times[j]
						m_j = dims[j]
						int_g = self.G(w,T - times[j],b)
						B[:,m_j] += int_g
						LL += numpy.sum(int_g * gamma[:,m_j])
						_lambda = init_intensity[m_j] * self.g(theta,t_j,b)
						if j == 0:
							psi = init_intensities[m_j,:] * self.g(theta,t_j,b) / _lambda
						else:
							psi = init_intensities[m_j,:] * self.g(theta,t_j,b)
							t_k = times[0:j]
							m_k = dims[0:j]
							# idx = [_idx for _idx in range(len(t_k)) if t_j - t_k[_idx] <= superparams['impact_period']]
							if len(m_k) == 0:
								psi /= _lambda
							else:
								_g = self.g(w,t_j - t_k,b)
								_a = gamma[m_j, m_k]
								if len(v) > 0 :
									_a += numpy.dot(alpha[m_j,m_k],v[:,t_k])
								phi = _g * _a
								_lambda += numpy.sum(phi)
								psi /= _lambda
								phi /= _lambda
								
								for k in range(len(m_k)):
									C[m_j,m_k[k]] += phi[k]
							LL -= numpy.log(_lambda)
						D[m_j,:] += psi
					LL += self.G(theta,T,b) * numpy.sum(init_intensity)
					
					intrinsic1 += D
					intrinsic2 += numpy.tile(features * self.G(theta,T,b), (M,1) )
					gamma = (- B + numpy.sqrt(B ** 2 + 8 * sigma * C )) / (4 * sigma) * numpy.sign(C)
					alpha = (- D + numpy.sqrt(D ** 2 + 4 * sigma * intrinsic2)) / (2 * sigma) * numpy.sign(intrinsic2)
					model['alpha'] = alpha
					Gamma[:,i] = gamma.reshape(M**2)
				# update beta
				B = intrinsic2 + sigma * reg_beta
				beta = (- B + numpy.sqrt(B ** 2 + 4 * sigma * intrinsic1)) / (2 * sigma) * numpy.sign(intrinsic1)
				
				# check convergence
				error = numpy.sum(numpy.abs(model['Gamma'] - Gamma)) / numpy.sum(model['Gamma'])
				print json.dumps({
					'iter':iteration,
					'outer_iter':outer_times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':LL,
					'w':w,
					'theta':theta,
					'b':b,
					'error':error,
					'mean_alpha':numpy.mean(Gamma),
				})
				iteration += 1

				model['Gamma'] = Gamma
				model['beta'] = beta
				if iteration > max_iter or error < superparams['thres']:
					Q_converge = True
					break
				else:
					Q_converge = False

			# update theta & w & b
			LL_old = LL
			step = 10**-4
			for i in range(I):
				T = train_seq[i]['times'][-1] + 0.01
				times = numpy.array(train_seq[i]['times'],dtype=float)
				dims = numpy.array(train_seq[i]['dims'],dtype=int)
				features = numpy.array(train_seq[i]['features'],dtype=float)
				N = len(times)

				init_intensity = numpy.dot(model['beta'], features)
				init_intensities = model['beta'] * numpy.tile(features,(M,1))
				gamma = model['Gamma'][:,i].reshape(M,M)
				theta = model['theta']
				w = float(model['w'])
				b = float(model['b'])

				Lw = 0.0
				for j in range(0,N):
					t_j = times[j]
					m_j = dims[j]
					grad1 = (self.f(w,b) + self.f(w,T - t_j - b)) * w - \
							(2.0 - numpy.exp(- w * b) - numpy.exp(- w *(T - t_j - b)))
					grad1 /= w ** 2
					grad1 *= numpy.sum(gamma[:,m_j])

					_lambda = init_intensity[m_j] * self.g(theta,t_j,b)
					grad2 = 0.0
					if j == 0:
						pass
					else:
						t_k = times[0:j]
						m_k = dims[0:j]
						if len(m_k) == 0:
							pass
						else:
							_g = self.g(w,t_j - t_k,b)
							_a = gamma[m_j, m_k]
							_f = self.f(w,numpy.abs(t_j - t_k - b))
							_lambda += numpy.sum(_g * _a)
							grad2 += numpy.sum(_f * _a)
					grad2 /= _lambda
					Lw += grad1 + grad2

				model['w'] -= step * numpy.sign(Lw)
			if outer_times > max_outer_iter:
				L_converge = True
			else:
				L_converge = False
			outer_times += 1

		return model

	def G(self,w,t,b):
		return (1 - numpy.exp(- w * b) + 1 - numpy.exp(- w * (t - b))) / w

	def g(self,w,t,b):
		return numpy.exp(- w * numpy.abs(t - b))

	def f(self,w,x):
		return x * numpy.exp(- w * x)

	def predict(self, model, train_seq, test_seq, superparams, initparams):
		M = model['M']
		pred_seqs = []
		pred_seqs_self = []
		pred_seqs_nonself = []

		real_seqs = []
		real_seqs_self = []
		real_seqs_nonself = []		
		mapes = []

		mapes_self = []
		mapes_nonself = []
		I = len(train_seq)

		for i in range(I):
			T = train_seq[i]['times'][-1] + 0.01
			times = numpy.array(train_seq[i]['times'],dtype=float)
			dims = numpy.array(train_seq[i]['dims'],dtype=int)
			features = numpy.array(train_seq[i]['features'],dtype=float)
			N = len(times)
			N_self = len([x for x in dims if x == 0])
			N_nonself = len([x for x in dims if x == 1])
			if N != N_self + N_nonself:
				print 'N != N_self + N_nonself'
				exit()

			Musum = numpy.dot(model['beta'], features)
			gamma = model['Gamma'][:,i].reshape(M,M)
			theta = model['theta']
			w = model['w']
			b = model['b']
			
			duration = model['superparams']['duration']
			mape = []
			mape_self = []
			mape_nonself = []

			pred_seq = []
			pred_seq_self = []
			pred_seq_nonself = []

			real_seq = []
			real_seq_self = []
			real_seq_nonself = []
			
			for year in range(duration+1):
				LL = 0
				LL_self = 0
				LL_nonself = 0
				for j in range(N):
					m_j = dims[j]
					int_g = self.G(w,T + year - times[j],b) - self.G(w,T - times[j],b)
					LL += numpy.sum(int_g * gamma[:,m_j])
					LL_self += int_g * gamma[0,m_j]
					LL_nonself += int_g * gamma[1,m_j]
				LL += (self.G(theta,T + year,b) - self.G(theta,T,b)) * numpy.sum(Musum)
				LL_self += (self.G(theta,T + year,b) - self.G(theta,T,b)) * numpy.sum(Musum)
				LL_nonself += (self.G(theta,T + year,b) - self.G(theta,T,b)) * numpy.sum(Musum)

				pred = N + LL
				pred_self = N_self + LL_self
				pred_nonself = N_nonself + LL_nonself
				real = N + len([x for x in test_seq[i]['times'] if x + model['superparams']['cut_point'] < T + year])
				real_self = N_self + len([x for _x,x in enumerate(test_seq[i]['times']) if x + \
							model['superparams']['cut_point'] < T + year and test_seq[i]['dims'][_x] == 0])
				real_nonself = N_nonself + len([x for _x,x in enumerate(test_seq[i]['times']) if x + \
							model['superparams']['cut_point'] < T + year and test_seq[i]['dims'][_x] == 1])
				mape.append(abs(pred - real) / float(real + 0.001))
				mape_self.append(abs(pred_self - real_self) / float(real + 0.001))
				mape_nonself.append(abs(pred_nonself - real_nonself) / float(real_nonself + 0.001))

				pred_seq.append(pred)
				pred_seq_self.append(pred_self)
				pred_seq_nonself.append(pred_nonself)

				real_seq.append(real)
				real_seq_self.append(real_self)
				real_seq_nonself.append(real_nonself)

			mapes.append(mape)
			mapes_self.append(mape_self)
			mapes_nonself.append(mape_nonself)

			pred_seqs.append(pred_seq)
			pred_seqs_self.append(pred_seq_self)
			pred_seqs_nonself.append(pred_seq_nonself)

			real_seqs.append(real_seq)
			real_seqs_self.append(real_seq_self)
			real_seqs_nonself.append(real_seq_nonself)

		av_mape = numpy.mean(numpy.array(mapes),0)
		av_acc = numpy.mean(numpy.array(mapes) < model['superparams']['epsilon'],0)

		av_mape_self = numpy.mean(numpy.array(mapes_self),0)
		av_acc_self = numpy.mean(numpy.array(mapes_self) < model['superparams']['epsilon'],0)

		av_mape_nonself = numpy.mean(numpy.array(mapes_nonself),0)
		av_acc_nonself = numpy.mean(numpy.array(mapes_nonself) < model['superparams']['epsilon'],0)

		return {
			'av_mape':av_mape.tolist(),
			'av_acc':av_acc.tolist(),
			'av_mape_self':av_mape_self.tolist(),
			'av_acc_self':av_acc_self.tolist(),
			'av_mape_nonself':av_mape_nonself.tolist(),
			'av_acc_nonself':av_acc_nonself.tolist(),
			# 'mapes':mapes,
			# 'mapes_self':mapes_self,
			# 'mapes_nonself':mapes_nonself,
			# 'pred_seqs':pred_seqs,
			# 'pred_seqs_self':pred_seqs_self,
			# 'pred_seqs_nonself':pred_seqs_nonself,
			# 'real_seqs':real_seqs,
			# 'real_seqs_self':real_seqs_self,
			# 'real_seqs_nonself':real_seqs_nonself,
			}

	def predict_one(self,patent_id):
		pass

	def load(self,f):
		data = []
		for i,row in enumerate(csv.reader(file(f,'r'))):
			if i % 4 == 2:
				row = [float(row[1])]
			elif i % 4 == 0 or i % 4 == 1:
				row = [float(x) for x in row[1:]]
			elif i % 4 == 3:
				_row = [float(x) for x in row[1:]]
				_max = max(_row)
				_min = min(_row)
				row = [(x - _min)/float(_max - _min) for x in _row]
			data.append(row)

		T = 15
		lines = 4
		I = int(len(data)/lines)
		train_seq = []
		test_seq = []
		for i in range(I):
			publish_year = data[i * lines + 2]
			self_seq = data[i * lines]
			nonself_seq = data[i * lines + 1]
			feature = data[i * lines + 3]
			varying = data[(i * lines + 4):(i * lines + lines)]

			time_seq = self_seq + nonself_seq
			dim_seq = ([0] * len(self_seq)) + ([1] * len(nonself_seq))
			S = zip(time_seq,dim_seq)
			S = sorted(S,key=lambda x:x[0])
			Y = [x[0] for x in S]
			dim_seq = [x[1] for x in S]
			cut_point = T
			time_train = [y for y in Y if y <= cut_point]
			dim_train = [e for i,e in enumerate(dim_seq) if Y[i] <= cut_point]
			if len(time_train) < 5:
				continue
			_dict = {}
			_dict['times'] = time_train
			_dict['dims'] = dim_train
			_dict['features'] = feature
			_dict['varying'] = varying
			_dict['publish_year'] = publish_year
			train_seq.append(_dict)

			_dict = {}
			_dict['times'] = [y - cut_point for y in Y if y > cut_point]
			_dict['dims'] = [e for i,e in enumerate(dim_seq) if Y[i] > cut_point]
			_dict['features'] = feature
			_dict['varying'] = varying
			_dict['publish_year'] = publish_year
			test_seq.append(_dict)


		superparams = {}
		superparams['M'] = 2
		superparams['outer'] = 20
		superparams['inner'] = 10
		superparams['K'] = len(train_seq[0]['features'])
		superparams['impact_period'] = 5
		superparams['sigma'] = 1
		superparams['thres'] = 1e-3
		superparams['cut_point'] = cut_point
		superparams['duration'] = 10
		superparams['epsilon'] = 0.3

		initparams = {}
		initparams['theta'] = 0.2
		initparams['w'] = 1.0
		initparams['b'] = 0.5


		return train_seq,test_seq,superparams,initparams


if __name__ == '__main__':
	predictor = Delay()
	loaded = predictor.load('train_sequence_large_long_new.txt')
	# model = predictor.train(*loaded)
	result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
	print result

	pass