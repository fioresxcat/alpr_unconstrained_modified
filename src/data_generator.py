
import numpy as np

from threading import Semaphore, Thread
from time import sleep
from random import choice, randint
from pdb import set_trace as pause

class DataGenerator(object):

	def __init__(	self, data, process_data_item_func, xshape, yshape, \
					data_item_selector	= choice, 	\
					nthreads			= 2,		\
					pool_size			= 1000,		\
					min_nsamples		= 1,		\
					dtype 				= 'single' ):

		assert pool_size >= min_nsamples, \
			'Min. samples must be equal or less than pool_size'
		assert min_nsamples > 0 and pool_size > 0, \
			'Min. samples and pool size must be positive non-zero numbers'

		self._data = data
		self._process_data_item = process_data_item_func
		self._data_item_selector = data_item_selector
		self._xshape = xshape
		self._yshape = yshape
		self._nthreads = nthreads
		self._pool_size = pool_size
		self._min_nsamples = min_nsamples
		self._dtype = dtype
		
		self._count = 0
		self._stop = False
		self._threads = []
		self._sem = Semaphore()

		self._X, self._Y = self._get_buffers(self._pool_size)


	def _get_buffers(self,N):
		X = np.empty((N,) + self._xshape, dtype=self._dtype)
		Y = np.empty((N,) + self._yshape, dtype=self._dtype)
		return X,Y

	def _compute_sample(self):
		'''
		Hàm này sẽ return 2 giá trị: XX và YY
		XX là ảnh đã augment
		YY chính là nhãn cho ảnh đó (Ytrue): là matrix có shape M x N x 9 (channel đầu tiên gồm toàn 1 vs 0, 8 channel sau là 8 tọa độ của 4 đỉnh của LP tương ứng trong output feature map, chính là A_mn trong paper)
		'''
		d = self._data_item_selector(self._data)
		return self._process_data_item(d)

	def _insert_data(self,x,y):

		self._sem.acquire()

		if self._count < self._pool_size:
			self._X[self._count] = x
			self._Y[self._count] = y
			self._count += 1
		else:
			idx = randint(0,self._pool_size-1)
			self._X[idx] = x
			self._Y[idx] = y

		self._sem.release()

	def _run(self):
		'''
		khi bắt đầu run, liên tục lấy example từ self._data, đưa vào self._X và self._Y
		'''
		while True:
			# lấy một example từ self._data. Example gồm (ảnh, label).
			# output sau khi process example này, ta được 1 bộ example mới: XX và YY
			# XX là ảnh đã augment
			# YY là matrix có shape M x N x 9 (channel đầu tiên gồm toàn 1 vs 0, 8 channel sau là 8 tọa độ của 4 đỉnh của LP tương ứng trong output feature map, chính là A_mn trong paper)
			x,y = self._compute_sample() 
			
			# insert một bộ example vào self._X và self._Y
			self._insert_data(x,y)

			if self._stop:
				break

	def stop(self):
		'''
		kết thúc quá trình sử dụng tài nguyên để thực hiện lấy data
		'''
		self._stop = True
		for thread in self._threads:
			thread.join()

	def start(self):
		'''
		bắt đầu quá trình sử dụng tài nguyên để thực hiện lấy data
		'''
		self._stop = False
		self._threads = [Thread(target=self._run) for n in range(self._nthreads)]
		for thread in self._threads:
			thread.setDaemon(True)
			thread.start()

	def get_batch(self,N):
		'''
		lấy 1 batch với N example
		'''
		# Wait until the buffer was filled with the minimum
		# number of samples
		while self._count < self._min_nsamples:
			sleep(.1)

		X,Y = self._get_buffers(N)
		self._sem.acquire()
		for i in range(N):
			idx = randint(0,self._count-1)
			X[i] = self._X[idx]
			Y[i] = self._Y[idx]
		self._sem.release()
		return X,Y


