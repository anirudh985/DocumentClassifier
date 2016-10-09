


class ModelAccuracyEvaluator:

	def __init__(self):
		self.currentAccuracy = 0.0
		self.numberOfDocumentsEvaluated = 0
		self.numberOfDocumentsCorrectlyPredicted = 0

	def CheckForMatch(self,predictedResult, actualLabels):
		for i in range(0, len(predictedResult)):
			if predictedResult[i] == 1:
				if actualLabels[i] == 1:
					return True
		return False

	def CheckForStrictMatch(self,predictedResult, actualLabels):
		for i in range(0, len(actualLabels)):
			if actualLabels[i] == 1:
				if predictedResult[i] != 1:
					return False
			else:
				if predictedResult[i] != 0:
					return False
		return True

	def SetListOfTopics(self,listOfTopics):
		self.listOfTopics = listOfTopics

	def EvaluateDocument(self,predictedResult, actualLabels):
		self.numberOfDocumentsEvaluated += 1
		isPredictionCorrect = self.CheckForStrictMatch(predictedResult,actualLabels)
		if isPredictionCorrect == True:
			self.numberOfDocumentsCorrectlyPredicted += 1
		self.currentAccuracy = float((self.numberOfDocumentsCorrectlyPredicted * 100 ) / float(self.numberOfDocumentsEvaluated))
		# self.printCorrespodingTopics(predictedResult)
		# self.printCorrespodingTopics(actualLabels)
		# print "\n"

	def printCorrespodingTopics(self, outputPrediction):
		print map(lambda (x, y): y if x == 1 else '', zip(outputPrediction, self.listOfTopics))

	def GetAccuracy(self):
		return [self.currentAccuracy, self.numberOfDocumentsEvaluated, self.numberOfDocumentsCorrectlyPredicted]