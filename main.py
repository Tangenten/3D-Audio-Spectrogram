from __future__ import unicode_literals

import collections
import math
import multiprocessing
import sys
import os

import numpy
import numba

from pynput import keyboard

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

import pyaudio
import soundfile
import youtube_dl
from presets import Preset
import librosa as _librosa

librosa = Preset(_librosa)
librosa['sr'] = 44100


def youtubeToWav(youtubeLink):
	ydl_opts = {
		'format': 'bestaudio/best',
		'postprocessors': [{
			'key': 'FFmpegExtractAudio',
			'preferredcodec': 'wav',
			'preferredquality': '192',
		}],
		'prefer_ffmpeg': True,
		'keepvideo': False,
		'quiet': False
	}
	
	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
		info_dict = ydl.extract_info(youtubeLink, download = False)
		video_title = info_dict.get('title', None)
		
		path = os.path.dirname(os.path.realpath(__file__))
		file_path = sys.path[0] + "\\" + video_title + ".wav"
		
		if os.path.isfile(file_path):
			print("File exist")
		else:
			ydl.download([youtubeLink])
			
			for f_name in os.listdir(path):
				if f_name.startswith(video_title) and f_name.endswith('.wav'):
					os.rename(f_name, file_path)
					break
	
	return file_path


@numba.jit(nopython = True)
def linearInterpolation(y1, y2, frac):
	return y1 * (1.0 - frac) + y2 * frac


@numba.jit(nopython = True)
def cosineInterpolation(y1, y2, frac):
	frac2 = (1.0 - math.cos(frac * 3.14)) / 2
	return (y1 * (1.0 - frac2) + y2 * frac2)


@numba.jit(nopython = True)
def cubicInterpolation(y0, y1, y2, y3, mu):
	mu2 = mu * mu
	a0 = y3 - y2 - y0 + y1
	a1 = y0 - y1 - a0
	a2 = y2 - y0
	a3 = y1
	
	return a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3


@numba.jit(nopython = True)
def hermiteInterpolation(y0, y1, y2, y3, mu, tension, bias):
	mu2 = mu * mu
	mu3 = mu2 * mu
	
	m0 = (y1 - y0) * (1 + bias) * (1 - tension) / 2
	m0 += (y2 - y1) * (1 - bias) * (1 - tension) / 2
	m1 = (y2 - y1) * (1 + bias) * (1 - tension) / 2
	m1 += (y3 - y2) * (1 - bias) * (1 - tension) / 2
	a0 = 2 * mu3 - 3 * mu2 + 1
	a1 = mu3 - 2 * mu2 + mu
	a2 = mu3 - mu2
	a3 = -2 * mu3 + 3 * mu2
	
	return a0 * y1 + a1 * m0 + a2 * m1 + a3 * y2;


@numba.jit(nopython = True)
def logarithmicInterpolation(min, max, current, destination):
	return min * math.pow(max / min, current / (destination - 1))


@numba.jit(nopython = True)
def getColors(list, scalar):
	colorList = [(0.0, 0.0, 0.0, 0.0) for x in range((len(list) - 1) * 2)]
	offset = 0
	for i in range(0, (len(list) - 1) * 2, 2):
		z = list[offset]
		
		colorR = numpy.interp(z, [0.0, scalar], [0.3, 0.9])
		colorG = numpy.interp(z, [0.0, scalar], [0.1, 0.5])
		colorB = numpy.interp(z, [0.0, scalar], [0.5, 0.4])
		
		# colorR = logarithmicInterpolation(0.1, 1, z, scalar)
		# colorG = logarithmicInterpolation(0.1, 0.6, z, scalar)
		# colorB = logarithmicInterpolation(0.1, 0.1, z, scalar)
		
		colorList[i] = (colorR, colorG, colorB, 1)
		colorList[i + 1] = (colorR - 0.01, colorG - 0.01, colorB - 0.01, 1)
		offset += 1
	
	return colorList


class spectrogram:
	
	def __init__(self, audioHandler, bins, scalar):
		self.sampleRate = audioHandler.sampleRate
		self.frequencyBins = bins
		self.scalarValue = scalar
		self.scalar = self.frequencyBins * self.scalarValue
		self.freqScalarTop = int(self.scalar * 0.5)
		self.freqScalarBottom = int(self.scalar * 0.0005)
		
		self.logScale = [0.0] * self.frequencyBins
		for i in range(0, self.frequencyBins, 1):
			self.logScale[i] = int(logarithmicInterpolation(self.freqScalarBottom + 1, self.freqScalarTop + 1, i, self.frequencyBins + 1) - 1)
	
	def output(self, samples):
		logFFT = numpy.zeros(self.frequencyBins, dtype = numpy.float32)
		
		# mel = librosa.feature.melspectrogram(numpy.asarray(samples), n_fft = self.frequencyResolution, hop_length = len(samples) + 1, window=numpy.hamming, sr = self.sampleRate, n_mels = self.frequencyBins)
		# samples = librosa.amplitude_to_db(mel, ref = numpy.max)
		# #samples = librosa.power_to_db(S, ref = numpy.max)
		# samples = samples + 81
		# samples = samples / 81
		#
		# for i in range(0, len(samples), 1):
		# 	logFFT[i] = samples[i][0]
		
		samples_fft = numpy.fft.rfft(samples, n = self.scalar)
		samples_fft = numpy.abs(samples_fft)
		
		for i in range(0, self.frequencyBins, 1):
			y1 = samples_fft[int(self.logScale[i])]
			y1 = math.log2(y1 + 1) - 1
			
			# y1 = ((1 + y1) / (((self.frequencyBins) + 1) / (1 + i)))
			# y1 = (1 + y1) * ((1 + i) / ((self.frequencyBins) + 1))
			
			logFFT[i] = y1
		
		return logFFT
	
	def init(self, bins, scalar):
		self.frequencyBins = bins
		self.scalarValue = scalar
		self.scalar = self.frequencyBins * self.scalarValue
		self.freqScalarTop = int(self.scalar * 0.2)
		self.freqScalarBottom = int(self.scalar * 0.0005)
		
		self.logScale = [0.0] * self.frequencyBins
		for i in range(0, self.frequencyBins, 1):
			self.logScale[i] = int(logarithmicInterpolation(self.freqScalarBottom + 1, self.freqScalarTop + 1, i, self.frequencyBins + 1) - 1)


class audioHandler(multiprocessing.Process):
	
	def __init__(self, inputFile):
		super(multiprocessing.Process, self).__init__()
		self.daemon = True
		
		self.callbackPipeChild, self.callbackPipeParent = multiprocessing.Pipe(duplex = False)
		self.samplesQueue = multiprocessing.Queue()
		self.messageQueue = multiprocessing.Queue()
		
		self.deviceInfo = None
		
		self.bitDepth = 32
		# self.sampleRate = int(self.deviceInfo['defaultSampleRate'])
		self.sampleRate = 44100
		self.channels = 2
		self.frameCount = self.sampleRate // 16
		self.bufferSize = self.frameCount * self.channels
		
		self.running = False
		self.latency = 0
		
		self.inputFile = inputFile
	
	def audioCallback(self, in_data, frame_count, time_info, status):
		if status == pyaudio.paOutputOverflow or status == pyaudio.paOutputUnderflow:
			print("Underflow / Overflow")
		
		samples = self.callbackPipeChild.recv()
		self.samplesQueue.put_nowait(samples)
		
		return numpy.array(samples, dtype = numpy.float32), pyaudio.paContinue
	
	def input(self):
		while not self.messageQueue.empty():
			message = self.messageQueue.get_nowait()
			if message == "STOP":
				self.running = False
	
	def run(self):
		pyAudio = pyaudio.PyAudio()
		audioCallback = pyAudio.open(format = pyaudio.paFloat32, channels = self.channels, rate = self.sampleRate, frames_per_buffer = self.frameCount, stream_callback = self.audioCallback, output = True, start = False)
		audioCallback.start_stream()
		
		self.deviceInfo = pyAudio.get_default_output_device_info()
		self.latency = audioCallback.get_output_latency()
		
		outputData, outputSampleRate = soundfile.read(self.inputFile)
		# if outputSampleRate != self.sampleRate:
		
		# outputData = librosa.resample(numpy.asfortranarray(outputData), outputSampleRate, self.sampleRate, res_type = 'scipy', fix = True, scale = True)
		# outputData = numpy.reshape(outputData, (-1, 2))
		
		dataLength = len(outputData)
		samples = [0.0] * self.bufferSize
		offset = self.sampleRate * 0
		
		self.running = True
		while self.running:
			self.input()
			for i in range(0, self.bufferSize, 2):
				samples[i] = outputData[offset][0]
				samples[i + 1] = outputData[offset][1]
				
				offset += 1
				
				if offset >= dataLength:
					offset = 0
			
			self.callbackPipeParent.send(samples)  # Blocking if size == 1
		
		audioCallback.stop_stream()
		audioCallback.close()
		pyAudio.terminate()
		self.terminate()
	
	def stop(self):
		self.messageQueue.put_nowait("STOP")


class graphicHandler(object):
	
	def __init__(self, audioHandler):
		self.app = QtGui.QApplication(sys.argv)
		self.w = gl.GLViewWidget()
		self.w.setGeometry(50, 50, 720, 720)
		self.w.setWindowTitle('3D Spectrogram')
		self.w.show()
		
		listener = keyboard.Listener(on_press = self.on_press)
		listener.start()
		self.events = []
		
		self.m1 = gl.GLMeshItem(
			edgeColor = (0, 0, 0, 1),
			smooth = False,
			drawEdges = False,
			drawFaces = True,
			computeNormals = False
		)
		
		self.setLengths(128, 128, 64, 64)
		
		self.w.addItem(self.m1)
		
		self.zTop = 128
		self.colorThreshold = 3000
		self.zFallingRate = 1.1
		self.smoothingPasses = 2
		
		self.yOffset = 0
		
		self.deltaTime = 0
		self.timestamp = QtCore.QTime()
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.render)
		
		self.autoScroll = True
		
		self.audioHandler = audioHandler
		self.spectrogram = spectrogram(self.audioHandler, self.xLength, 64)
		self.samples = collections.deque(maxlen = int(self.audioHandler.bufferSize))
		self.samples.extend([0.0] * int(self.audioHandler.bufferSize))
	
	def on_press(self, key):
		try:
			self.events.append(key.char)
		except AttributeError:
			pass
	
	def input(self):
		for event in self.events:
			if event == "q":
				self.setLengths(self.xLength // 2, self.yLength, self.xSize, self.ySize)
				self.spectrogram.init(self.xLength, self.spectrogram.scalarValue)
			if event == "w":
				self.setLengths(self.xLength * 2, self.yLength, self.xSize, self.ySize)
				self.spectrogram.init(self.xLength, self.spectrogram.scalarValue)
			
			if event == "a":
				self.setLengths(self.xLength, self.yLength // 2, self.xSize, self.ySize)
			if event == "s":
				self.setLengths(self.xLength, self.yLength * 2, self.xSize, self.ySize)
			
			if event == "e":
				self.setLengths(self.xLength, self.yLength, self.xSize // 2, self.ySize)
			if event == "r":
				self.setLengths(self.xLength, self.yLength, self.xSize * 2, self.ySize)
			
			if event == "d":
				self.setLengths(self.xLength, self.yLength, self.xSize, self.ySize // 2)
			if event == "f":
				self.setLengths(self.xLength, self.yLength, self.xSize, self.ySize * 2)
			
			if event == "t":
				self.setHeight(self.zTop - 10)
				print(self.zTop)
			if event == "y":
				self.setHeight(self.zTop + 10)
				print(self.zTop)
			
			if event == "g":
				self.setColorThreshold(self.colorThreshold - 100)
				print(self.colorThreshold)
			if event == "h":
				self.setColorThreshold(self.colorThreshold + 100)
				print(self.colorThreshold)
			
			if event == "u":
				self.setDropOffRate(self.zFallingRate + 0.01)
				print(self.zFallingRate)
			if event == "i":
				self.setDropOffRate(self.zFallingRate - 0.01)
				print(self.zFallingRate)
			
			if event == "j":
				self.setSmoothingPasses(self.smoothingPasses - 1)
				print(self.smoothingPasses)
			
			if event == "k":
				self.setSmoothingPasses(self.smoothingPasses + 1)
				print(self.smoothingPasses)
			
			if event == "o":
				self.spectrogram.init(self.spectrogram.frequencyBins, self.spectrogram.scalarValue // 2)
				print(self.spectrogram.scalarValue)
			
			if event == "p":
				self.spectrogram.init(self.spectrogram.frequencyBins, self.spectrogram.scalarValue * 2)
				print(self.spectrogram.scalarValue)
			
			if event == "m":
				self.autoScroll ^= True
			
			self.events.clear()
	
	def setLengths(self, x, y, xSize, ySize):
		self.xLength = x
		self.xSize = xSize
		self.yLength = y
		self.ySize = ySize
		
		if self.xLength > self.yLength:
			self.colorOffset = max(self.xLength, self.yLength)
		else:
			self.colorOffset = min(self.xLength, self.yLength)
		
		self.verts = []
		for y in range(self.yLength):
			for x in range(self.xLength):
				self.verts.append([x * self.xSize, y * self.ySize, 0.0])
		
		self.faces = []
		self.colors = []
		for y in range(self.yLength - 1):
			for x in range(self.xLength - 1):
				self.faces.append([x + (y * self.colorOffset), (x + 1 + (y * self.colorOffset)), x + ((y + 1) * self.colorOffset)])
				self.colors.append([0.5, 0.5, 0.5, 1])
				self.faces.append([x + 1 + (y * self.colorOffset), (x + ((y + 1) * self.colorOffset)), x + 1 + ((y + 1) * self.colorOffset)])
				self.colors.append([0.5, 0.5, 0.5, 1])
		
		self.zFalling = [0.0] * self.xLength
		
		self.verts = numpy.array(self.verts, dtype = numpy.float32)
		self.faces = numpy.array(self.faces, dtype = numpy.uint32)
		self.colors = numpy.array(self.colors, dtype = numpy.float32)
		
		self.clearVerts = numpy.array(self.verts, dtype = numpy.float32)
		self.clearFaces = numpy.array(self.faces, dtype = numpy.uint32)
		self.clearColors = numpy.array(self.colors, dtype = numpy.float32)
		
		self.meshData = gl.MeshData(vertexes = self.verts, faces = self.faces, faceColors = self.colors)
		
		self.azimuth = 0
		
		self.m1.setMeshData(
			meshdata = self.meshData
		)
		
		self.w.setCameraPosition(pos = ((self.xLength * self.xSize) // 2, (self.yLength * self.ySize) // 2, 0))
	
	def getAudioSamplesForFrame(self):
		while not self.audioHandler.samplesQueue.empty():
			self.samples.extend(self.audioHandler.samplesQueue.get_nowait())
		
		samplesToRender = int(self.audioHandler.sampleRate * self.deltaTime)
		samples = []
		for i in range(samplesToRender):
			if self.samples:
				samples.append(self.samples.popleft())
			else:
				samples.append(0.0)
		
		return samples
	
	def getFFTBins(self, samples):
		fftBins = self.spectrogram.output(samples)
		fftBins *= self.zTop
		fftBins = fftBins[::-1]
		fftBins[0] = 0.0
		fftBins[len(fftBins) - 1] = 0.0
		
		return fftBins
	
	def updateTime(self):
		timeElapsed = self.timestamp.elapsed()
		self.deltaTime = timeElapsed / 1000.0 if timeElapsed != 0 else 0.016
		self.timestamp.start()
	
	def render(self):
		self.input()
		self.updateTime()
		
		samples = self.getAudioSamplesForFrame()
		fftBins = self.getFFTBins(samples)
		
		for i in range(self.smoothingPasses):
			smoothedBins = numpy.array(fftBins, dtype = numpy.float32)
			smoothedBins[0] = fftBins[0]
			for x in range(1, len(fftBins) - 1, 1):
				leftBin = fftBins[x - 1]
				rightBin = fftBins[x + 1]
				behindBin = self.verts[(len(self.verts) - self.xLength) + x][2]
				behindRightBin = self.verts[(len(self.verts) - self.xLength) + x + 1][2]
				behindLeftBin = self.verts[(len(self.verts) - self.xLength) + x - 1][2]
				
				sum = leftBin + rightBin + behindBin + behindLeftBin + behindRightBin
				sum /= 5
				sum = ((fftBins[x] + sum) / 2)
				smoothedBins[x] = sum
			
			smoothedBins[len(fftBins) - 1] = fftBins[len(fftBins) - 1]
			fftBins = smoothedBins
		
		self.zFalling = [x / self.zFallingRate for x in self.zFalling]
		for x in range(self.xLength):
			currZ = fftBins[x]
			if currZ < self.zFalling[x]:
				z = self.zFalling[x]
			else:
				z = currZ
				self.zFalling[x] = currZ
			
			fftBins[x] = z
		
		self.verts[: len(self.verts) - self.xLength, 2:] = self.verts[self.xLength:, 2:]
		self.verts[(len(self.verts) - self.xLength):, 2:] = numpy.asarray(fftBins, dtype = numpy.float32).reshape(-1, 1)
		
		colorList = getColors(fftBins, self.colorThreshold)
		self.colors[: len(self.colors) - ((self.colorOffset - 1) * 2)] = self.colors[((self.colorOffset - 1) * 2):]
		self.colors[len(self.colors) - ((self.colorOffset - 1) * 2):] = colorList
		
		self.meshData.setFaceColors(self.colors)
		self.meshData.setVertexes(self.verts, resetNormals = False)
		self.m1.meshDataChanged()
		
		if self.autoScroll:
			self.azimuth += 0.1
			self.w.setCameraPosition(azimuth = self.azimuth)
	
	def start(self):
		self.timer.start(16)
		if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
			QtGui.QApplication.instance().exec_()
	
	def stop(self):
		self.timer.stop(0)
	
	def setHeight(self, z):
		self.zTop = z
	
	def setColorThreshold(self, threshold):
		self.colorThreshold = threshold
	
	def setSmoothingPasses(self, passes):
		self.smoothingPasses = passes
	
	def setDropOffRate(self, rate):
		self.zFallingRate = rate


if __name__ == '__main__':
	a = audioHandler(youtubeToWav("https://www.youtube.com/watch?v=IUVFXhrnzPs"))
	g = graphicHandler(a)
	a.start()
	g.start()
