import numpy as np
import json
import sys
import math
import cv2
import time
import urllib
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.4,
                       minDistance = 7,
                       blockSize = 7 )

class IPCameraRead:

	def __init__( self, ip_addr ):
		self.stream=urllib.urlopen(ip_addr)
		print "Reading from ", ip_addr 
		self.bytes = ''
	
	def readip( self ):
		valid = False
		while ( not valid ):
			self.bytes += self.stream.read( 3072 )
			a = self.bytes.find('\xff\xd8')
			b = self.bytes.find('\xff\xd9')
			if a!=-1 and b!=-1:
				valid = True
				jpg = self.bytes[ a:b+2 ]
				self.bytes = self.bytes[ b+2: ]
				frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
		return True,frame

class SensorRead :
	def __init__(self, ip_addr ):
		self.ip_addr = ip_addr+"%s"
		self.timstmp = 0 
		self.valid = 0

	def getvalues( self, sense ):
		valid = False
		while not valid:
			if self.timstmp != 0: 
				params = urllib.urlencode({'from' : self.timstmp ,'sense' :sense } )
			else : 
				params = urllib.urlencode({'sense' : sense } )
			self.handle = urllib.urlopen( self.ip_addr % params )
			values = json.loads( self.handle.read() )
			sensors = sense.split(',')
			data = {}
			for sensor in sensors:
				if len( values[sensor]['data'] ) > 0:
					self.timstmp = values[sensor]['data'][ -1 ][ 0 ] 
					valid = True
					data[ sensor ] = values[sensor]['data'][-1][-1]
			if len(data)>0:
				return data
		

class App:
	def __init__(self, video_src, ip_addr=0):
		self.track_len = 10
		self.vid_src = video_src
		self.detect_interval = 10 
		self.tracks = []
		self.timestamp = [] 
		self.profiler = []
		self.profilerIP = []
		if self.vid_src == 0 :
			print "Initializing Laptop Camera "
			self.cam = cv2.VideoCapture(self.vid_src)
		elif self.vid_src == 1 :
			print "Initializing IP Camera "
			self.cam = IPCameraRead( ip_addr )		
		self.frame_idx = 0

	def Vect( self, tmstmp, pos ):

		cords = [(cord[ -1] , cord[ 0 ] ) for cord in pos ]
		disp = map( lambda (a,b) : (a[0]-b[0], a[1]-b[1]),cords )
		timespread = [ times[-1]-times[0] for times in tmstmp ]		
		velocity = map( lambda ((x,y),t):  (x/t,y/t) if t>0 else (x/0.001,y/0.0001) , zip( disp, timespread ) ) 

		weights = self.GaussianWeights( [ cord[ -1 ] for cord in pos ] , 1 , ( 320,270 ) )
		weighted_velocity = [ (weight*vel[0],weight*vel[1]) for weight, vel in zip( weights, velocity ) ]

	#	print weights  
	#	print [ cord[ -1] for cord in pos ] 

		x_vel = sum([ vel[ 0 ] for vel in weighted_velocity ])
		y_vel = sum([ vel[ 1 ] for vel in weighted_velocity ])

		if len(timespread)>0:
			print "X:",x_vel/len(timespread),"Y:", y_vel/len(timespread) 
			print "Orientation:",math.degrees(math.atan2(-y_vel,x_vel))	
		return x_vel, y_vel

	def GaussianWeights( self, cords, var, mean):
		weights = map( lambda (x,y): (100.0/(2*np.pi* var))*( np.exp( -0.25 *( (( x - mean[ 0 ])**2)+((y - mean[ 1 ])**2)) /( 100*var ) ) ), cords )    
		return weights
					

	def run(self):
		while True:
			if self.vid_src == 0 : 
#				init = time.time()
				ret, frame = self.cam.read()
#				final = time.time()
#				self.profiler.append( final - init )
			elif self.vid_src == 1:
#
				init = time.time()
				ret, frame = self.cam.readip()
				final = time.time()
				self.profilerIP.append( final - init )
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			vis = frame.copy()

			if len(self.tracks) > 0:
				img0, img1 = self.prev_gray, frame_gray
				p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
				p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
				p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
				d = abs(p0-p0r).reshape(-1, 2).max(-1)
				good = d < 1
				new_tracks = []
				new_time = [] 
				for tmstmp, tr, (x, y), good_flag in zip(self.timestamp,self.tracks, p1.reshape(-1, 2), good):
					if not good_flag:
						continue
					tr.append((x, y))
					tmstmp.append( time.time() )

					if len(tr) > self.track_len:
						del tr[0]
						del tmstmp[0]

					new_tracks.append(tr)
					new_time.append( tmstmp )
					cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

				self.tracks = new_tracks
				self.timestamp = new_time

				lines = [np.int32(tr) for tr in self.tracks]

				cv2.polylines(vis, lines, False, (0, 255, 0))

			if self.frame_idx % self.detect_interval == 0:
				mask = np.zeros_like(frame_gray)
				mask[:] = 255
				for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
					cv2.circle(mask, (x, y), 5, 0, -1)
				p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
				if p is not None:
					for x, y in np.float32(p).reshape(-1, 2):
						self.tracks.append([(x, y)])
						self.timestamp.append([time.time()])


			self.frame_idx += 1
			self.prev_gray = frame_gray
			grid = [ ( x , y ) for x in range( 100,500 , 50 ) for y in range( 100 , 500, 50 ) ]

			weights = self.GaussianWeights( grid , 1 , ( 320,270 ) )

			for weight,point in zip(weights,grid):
				if weight > 0.000001 :
					cv2.circle(vis, point , 2, (255, 255, 0), -1)
			cv2.imshow('lk_track', vis)
			self.Vect( self.timestamp, self.tracks )
#			if self.vid_src == 0:
#				print "Average Time for capture :", sum( self.profiler )/len( self.profiler)
			if self.vid_src == 1 :
				print "Average Time for capture :", sum( self.profilerIP )/len( self.profilerIP)

			ch = 0xFF & cv2.waitKey(1)
			if ch == 27:
				break
class AnalogData:
# constr
	def __init__(self, maxLen):
		self.ax = deque([0.0]*maxLen)
		self.ay = deque([0.0]*maxLen)
		self.az = deque([0.0]*maxLen)
		self.maxLen = maxLen
 
  # ring buffer
	def addToBuf(self, buf, val):
		if len(buf) < self.maxLen:
			buf.append(val)
		else:
			buf.popleft()
			buf.append(val)
 
  # add data
	def add(self, data):
		self.addToBuf(self.ax, data[0])
		self.addToBuf(self.ay, data[1])
		self.addToBuf(self.az, data[2])
    
# plot class
class AnalogPlot:
  # constr
	def __init__(self, analogData):
	# set plot to animated
		plt.ion() 
		self.axline, = plt.plot(analogData.ax)
		self.ayline, = plt.plot(analogData.ay)
		self.azline, = plt.plot(analogData.az)
		plt.ylim([-100, 100])

	# update plot
	def update(self, analogData):
		self.axline.set_ydata(analogData.ax)
		self.ayline.set_ydata(analogData.ay)
		self.azline.set_ydata(analogData.az)
		plt.draw()
 

def main():	
	args = sys.argv[ 1: ]
	sensor = SensorRead(args[ 0 ]+"/sensors.json?" )#"http://192.168.1.102:8080/sensors.json?")
#	while True:
#		try :
#
#			Mag = sensor.getvalues('mag,accel')
#			print Mag
#
#		except KeyboardInterrupt:
#			print "closing.."
#			break
	
	analogData = AnalogData(100)
	analogPlot = AnalogPlot(analogData)
	valid = False
	while True:
		try:
			values = sensor.getvalues('accel')
			data = [ float( val ) for val in values[ 'accel' ] ]
			print data
			analogData.add( data )
			analogPlot.update( analogData )
			if not valid :
				plt.show(block=False)
				valid = True
				print "Magnetometer Output .. "
		except KeyboardInterrupt :
			print "Closing .. "
			break

if __name__ == '__main__':
	main()

