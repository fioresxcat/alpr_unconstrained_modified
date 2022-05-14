
import numpy as np

from os.path import isfile


class Label:
	'''
	Là một class đại diện cho nhãn của một ảnh

	Có các trường giá trị:
	tl: top_left của bb trong ảnh đó (tương đối)
	br: bottom_right của bb trong ảnh đó (tương đối)
	cl: class của bb đó (có thể là LP, 0, I, J, K, M, N, ...)

	Class này sinh ra có liên quan mật thiết đến pts (vì pts để xác định tl và br của nó)
	Có cảm giác nó hơi thừa, nhưng thực ra nó giúp việc tính toán cc, wh các thứ được dễ dàng hơn so với pts
	'''
	def __init__(self,cl=-1,tl=np.array([0.,0.]),br=np.array([0.,0.]),prob=None):
		self.__tl 	= tl
		self.__br 	= br
		self.__cl 	= cl
		self.__prob = prob

	def __str__(self):
		return 'Class: %d, top_left(x:%f,y:%f), bottom_right(x:%f,y:%f)' % (self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

	def copy(self):
		return Label(self.__cl,self.__tl,self.__br)

	def wh(self): return self.__br-self.__tl

	def cc(self): return self.__tl + self.wh()/2

	def tl(self): return self.__tl
 
	def br(self): return self.__br

	def tr(self): return np.array([self.__br[0],self.__tl[1]])

	def bl(self): return np.array([self.__tl[0],self.__br[1]])

	def cl(self): return self.__cl

	def area(self): return np.prod(self.wh())

	def prob(self): return self.__prob

	def set_class(self,cl):
		self.__cl = cl

	def set_tl(self,tl):
		self.__tl = tl

	def set_br(self,br):
		self.__br = br

	def set_wh(self,wh):
		cc = self.cc()
		self.__tl = cc - .5*wh
		self.__br = cc + .5*wh

	def set_prob(self,prob):
		self.__prob = prob


def lread(file_path,label_type=Label):
	'''
	Hàm này để đọc từ file label có dạng csv với cấu trúc như sau:
	class, tọa độ x của điểm center, tọa độ y của điểm center, chiều rộng bb, chiều cao bb, prob
	'''
	if not isfile(file_path):
		return []

	objs = []
	with open(file_path,'r') as fd:
		for line in fd:
			v 		= line.strip().split()
			cl 		= int(v[0])
			ccx,ccy = float(v[1]),float(v[2])
			w,h 	= float(v[3]),float(v[4])
			prob 	= float(v[5]) if len(v) == 6 else None

			cc 	= np.array([ccx,ccy])
			wh 	= np.array([w,h])

			objs.append(label_type(cl,cc-wh/2,cc+wh/2,prob=prob))

	return objs

def lwrite(file_path,labels,write_probs=True):
	'''
	Hàm này để write từ Label() thu được ra file dạng csv với cấu trúc như sau:
	class, tọa độ x của điểm center, tọa độ y của điểm center, chiều rộng bb, chiều cao bb, prob
	'''
	with open(file_path,'w') as fd:
		for l in labels: # duyệt từng label và ghi chúng ra file. Mỗi label là 1 dòng
			cc,wh,cl,prob = (l.cc(),l.wh(),l.cl(),l.prob())
			if prob != None and write_probs:
				fd.write('%d %f %f %f %f %f\n' % (cl,cc[0],cc[1],wh[0],wh[1],prob))
			else:
				fd.write('%d %f %f %f %f\n' % (cl,cc[0],cc[1],wh[0],wh[1]))


def dknet_label_conversion(R,img_width,img_height):
	WH = np.array([img_width,img_height],dtype=float)
	L  = []
	for r in R:
		center = np.array(r[2][:2])/WH
		wh2 = (np.array(r[2][2:])/WH)*.5
		L.append(Label(ord(r[0]),tl=center-wh2,br=center+wh2,prob=r[1]))
	return L


class Shape():
	'''
	là class chứa dữ liệu đọc từ file label
	Bao gồm các thành phần:
	pts: tọa độ 4 đỉnh của bb
	text: là chữ cái/ chữ số nếu đó là label cho chữ cái/ chữ số trên biển
	'''
	def __init__(self,pts=np.zeros((2,0)),max_sides=4,text=''):
		self.pts = pts
		self.max_sides = max_sides
		self.text = text
		self.lp_type = 1  ######### fix

	def isValid(self):
		return self.pts.shape[1] > 2

	def write(self,fp):
		fp.write('%d,' % self.pts.shape[1])
		ptsarray = self.pts.flatten()
		fp.write(''.join([('%f,' % value) for value in ptsarray]))
		fp.write('%s,' % self.text)
		fp.write('\n')

	def read(self,line):
		data 		= line.strip().split(',')
		ss 			= int(data[0])   # là số đỉnh, luôn là 4
		lp_type     = float(data[1])  ####### fix
		values 		= data[2:(ss*2 + 2)]  ### fix
		text 		= data[(ss*2 + 2)] if len(data) >= (ss*2 + 3) else ''  ### fix # là các chữ số, chữ cái trên biển số. Là element cuối cùng trong 1 dòng trong file CSV
		self.pts 	= np.array([float(value) for value in values]).reshape((2,ss))  # là tọa độ 4 đỉnh của LP trong ảnh
		self.text   = text
		self.lp_type=lp_type  ##### fix

def readShapes(path,obj_type=Shape):
	shapes = []
	with open(path) as fp:
		for line in fp:
			shape = obj_type()
			shape.read(line)
			shapes.append(shape)
			break  #### fix: chỉ đọc dòng đầu tiên, nếu ko sẽ phải label lại hết các dòng trong file
	return shapes

def writeShapes(path,shapes):
	if len(shapes):
		with open(path,'w') as fp:
			for shape in shapes:
				if shape.isValid():
					shape.write(fp)

