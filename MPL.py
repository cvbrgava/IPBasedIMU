from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy
from sensor import SensorRead
import sys
from collections import deque 
import time

path = []

def rot( r, t ) :
	# Sample Rotation Basis
	return [ 0 , r*numpy.cos( t/10.0 )],[ 0,r*numpy.sin( t/10.0 )],[ 0,r*numpy.sin( t/0.02 )]  

def Rotx( angle ):
	# Sample Rotation basis
	x = [ 1, 0 , 0 ]
	y = [ 0 , numpy.cos( angle ) , -numpy.sin( angle ) ]
	z = [ 0 , numpy.sin( angle ) , numpy.cos( angle ) ]
	Rx = numpy.row_stack((x,y,z))
	return Rx

def cube( l ,b , h, format = 'l' ):
	# Euclidean Cube co-ordinates 
	x = [l,l,l,l,-l,-l,-l,-l,l]+[l,l,l,l,-l,-l,-l,-l,l]
	y = [-b,-b,b,b,b,b,-b,-b,-b]+[-b,-b,b,b,b,b,-b,-b,-b]
	z = [-h,h,h,-h,-h,h,h,-h,-h]+[h,-h,-h,h,h,-h,-h,h,h]
	if format == 'l':
		return x, y ,z 
	if format == 'm':
		return numpy.row_stack((x,y,z))

class MovingAvgFilter():

	def __init__( self, depth ):
		self.depth = depth
		self.pipe = deque([0.0])
		self.avg = 0

	def Filter( self, value ,debug=False):
		value_lost = 0.0
		number_of_values = len(self.pipe )

		if number_of_values < self.depth :
			self.pipe.append( value )
			self.avg = ( self.avg*(number_of_values  ) +  value   - value_lost   )/ (number_of_values+1)
		else :
			value_lost = self.pipe.popleft()
			self.pipe.append( value )
			self.avg = ( self.avg*(number_of_values ) +  value   - value_lost   )/ number_of_values
#	print number_of_values	

		return self.avg


class DirectionCosineMatrix( ) :
	def __init__( self, sensor, timestep=0.01 ,renormerror=0.001 ):
		# Initialize 
		self.sensor = sensor
		self.timestep = timestep
		self.renormError = renormerror
		self.omega = [ 0, 0 , 0 ]
		self.RotMatrixOld = numpy.array([[1,0,0],[0,1,0],[0,0,1]]) 
		self.RotMatrix = numpy.array([[1,0,0],[0,1,0],[0,0,1]]) 
		self.GonEarth = numpy.matrix([[-0.62135315],[0.73672485],[-9.697952]])
		self.displacement = numpy.matrix([[0.0],[0.0],[0.0]])
		self.BonEarth = numpy.matrix([[22.0],[20.0],[-0.0]])
		self.GFilterX =  MovingAvgFilter(15)
		self.GFilterY =  MovingAvgFilter(15)
		self.GFilterZ =  MovingAvgFilter(15)

	def Calibrate( self ):
		print "Calibrating Acclerometers..."
		for i in xrange( 50 ):
			inp = sensor.getvalues( 'accel' )
			self.accel = [ 1.0*float( val ) for val in inp[ 'accel' ] ]
			self.accel[ 0 ] = self.GFilterX.Filter( self.accel[ 0 ] )
			self.accel[ 1 ] = self.GFilterY.Filter( self.accel[ 1 ] )
			self.accel[ 2 ] = self.GFilterZ.Filter( self.accel[ 2 ] )
		self.GonPhone = self.accel


	def Renorm( self, RotMatrix ) :
		# Enforce Unitary conditions on the Rotation Matrix
		Rx = RotMatrix[ :, 0 ]
		Ry = RotMatrix[ :, 1 ]
		Rz = RotMatrix[ :, 2 ]
		error = numpy.dot( Rx, Ry )
		RxOrtho = Rx - ( (error/2)* Ry )
		RyOrtho = Ry - ( (error/2)* Rx )
		RzOrtho = numpy.cross( RxOrtho, RyOrtho )
		RxNorm = ( 0.5 )* ( 3 - numpy.dot( RxOrtho, RxOrtho ) ) * RxOrtho
		RyNorm = ( 0.5 )* ( 3 - numpy.dot( RyOrtho, RyOrtho ) ) * RyOrtho
		RzNorm = ( 0.5 )* ( 3 - numpy.dot( RzOrtho, RzOrtho ) ) * RzOrtho
#		print "Error :", error,numpy.dot( RxNorm, RyNorm )
		return numpy.column_stack( (RxNorm,RyNorm,RzNorm) )
	
	def BRotPlane( self ):
		# Calculate the rotation error between B and B in phone basis 
		BinPhone =  numpy.array( numpy.dot(numpy.linalg.pinv(self.RotMatrix),self.BonEarth ) )
		BinPhonelist = [val[ 0 ]  for val in BinPhone ] 
		init = time.time()
		inp = sensor.getvalues( 'mag' )
		#print "B download:", time.time()-init
		self.mag = [ 1.0*float( val ) for val in inp[ 'mag' ] ]
		Bplane = numpy.cross( BinPhonelist,self.mag )
#		print BinPhonelist
#		print self.mag
		return Bplane

	def GRotPlane( self, filter=True ):
		# Calculate the rotation error between G and G in phone basis 
		GinPhone =  numpy.array( numpy.dot(numpy.linalg.pinv(self.RotMatrix),self.GonEarth ) )
		GinPhonelist = [val[ 0 ]  for val in GinPhone ] 
		init = time.time()
		inp = sensor.getvalues( 'accel' )
		#print "G download :", time.time()-init
		self.accel = [ 1.0*float( val ) for val in inp[ 'accel' ] ]
#		print "Pre Filtering : ",self.accel
		if filter :
			self.accel[ 0 ] = self.GFilterX.Filter( self.accel[ 0 ] )
			self.accel[ 1 ] = self.GFilterY.Filter( self.accel[ 1 ] )
			self.accel[ 2 ]  = self.GFilterZ.Filter( self.accel[ 2 ],True )
#print "Post Filtering : ",self.accel[ 2 ] 
		#print numpy.sqrt( self.accel[0]**2 +  self.accel[1]**2+ self.accel[2]**2 )
		Gplane = numpy.cross( GinPhonelist,self.accel )
		return Gplane

	def PID( self, Kp, Kd ):
		# Evaluate omega correction from PID 
		Gplane = self.GRotPlane()
		Bplane = self.BRotPlane()
		correction = [ (1.2 * gval) + (0.2 * bval)  for gval,bval in zip(Gplane,Bplane) ]
		wcorrection = [ (Kp*val) + (Kd*self.timestep*val) for val in correction ]
		return wcorrection 

	def RotMatrixGyro(self, renorm = True, feedback = True, record=True ):
		# Integrate omega values from gyro to get the Rotation basis, correct with PID
		self.RotMatrixOld = self.RotMatrix
		init = time.time()
		inp = sensor.getvalues( 'gyro' )
		#print "Gyro download:", time.time()- init
		self.omega = [ 1.0*float( val ) for val in inp[ 'gyro' ] ]
		if feedback :
			omegaPID = self.PID( 0.2,0.5 )
			self.omega = [ val1 + val2 for val1,val2 in zip(self.omega,omegaPID) ]
		Incx = [ 1,  -1 *self.timestep * self.omega [ 2 ], self.timestep * self.omega[1] ]
		Incy = [ self.timestep * self.omega[ 2 ], 1 , -1* self.timestep * self.omega[ 0 ]]
		Incz = [ -1* self.timestep * self.omega[ 1 ], self.timestep * self.omega[ 0 ], 1 ]
		IntMatrix = numpy.row_stack(( Incx,Incy,Incz ) )
		self.RotMatrix = numpy.dot( self.RotMatrixOld, IntMatrix )	

		if renorm :
			self.RotMatrix = self.Renorm( self.RotMatrix )
	#	print self.RotMatrix
		if record :
			self.RecordMovement()
		return self.RotMatrix

	def RecordMovement( self ):
		PhoneBasisDisp = [ 0.5 * (val-self.GonPhone[ index ]) * self.timestep**2 for (index,val) in enumerate(self.accel) ]
#	print [ val for val in self.accel ]
#		print [ val for val in self.GonPhone ]
#print numpy.dot( self.RotMatrix, numpy.row_stack( (PhoneBasisDisp) ) )
		#self.displacement = self.displacement + ( numpy.dot( self.RotMatrix, numpy.row_stack( (PhoneBasisDisp) ) )  + (self.GonEarth * 0.5 * self.timestep**2)  ) 
		self.displacement = self.displacement + ( numpy.dot( self.RotMatrix, numpy.row_stack( (PhoneBasisDisp) ) ) )# + (self.GonEarth * 0.5 * self.timestep**2)  ) 
		#print   (numpy.dot( numpy.linalg.pinv(self.RotMatrix), self.GonEarth ) + numpy.row_stack( (self.accel) ) ) 
		path.append( self.displacement )
#		print path[-1]




# Initialize code 
fig = plt.figure()
ax = fig.add_axes([ 0,0,1,1],xlim=(-5,5),ylim=(-5,5),zlim=(-5,5), projection = '3d' )
line, = ax.plot( [], [] , [] , 'o-', lw=2 )
ax.grid()
args = sys.argv[ 1: ]
sensor = SensorRead( args[ 0 ] + "/sensors.json?" ) 
DCM = DirectionCosineMatrix( sensor )
DCM.Calibrate()

# Animation initialize
def init():
	line.set_data( [],[] )
	line.set_3d_properties( [] )
	return [line]

# Re-draw cube with the Rotation basis
def animate( i ) :
	CubeMatrix = cube( 2 , 4 ,1, 'm' )
	Rx = DCM.RotMatrixGyro(renorm=True,feedback=True )
	NewCube =numpy.dot( Rx, CubeMatrix)
	line.set_data( NewCube[ 0, :], NewCube[ 1, :] )
	line.set_3d_properties( NewCube[ 2,:] ) 
	fig.canvas.draw()
	return [line]

try :
	anim = animation.FuncAnimation(fig, animate, init_func=init,frames=2000, interval = 1,  blit=True)
	plt.show()

except KeyboardInterrupt:
	pass
print "Plotting.."
x = [ numpy.float( val[ 0 ] ) for val in path ]
y = [ numpy.float( val[ 1 ] ) for val in path ]
z = [ numpy.float( val[ 2 ] ) for val in path ]
fig = plt.figure()
ax = fig.add_subplot( 111, projection = '3d' )
ax.plot( x, y,z )
plt.show()



