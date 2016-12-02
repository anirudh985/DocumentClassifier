import time
from subprocess import check_call
import matplotlib.pyplot as plt

def parseRules():
    with open("./rulesFile") as fileObject:
        for line in fileObject:
            tempList = line.split(" ")
            rule = []
            rule.append(tempList[0])
            lineLength = len(tempList)
            for i in range(2,lineLength - 2):
                rule.append(tempList[i])
            rule.append(tempList[lineLength - 2][1:-1])
            rule.append(tempList[lineLength - 1][0:-2])
            rulesList.append(rule)

def cmpRules(a,b):
    if float(a[-1]) != float(b[-1]):
        return -1 if float(a[-1]) > float(b[-1]) else 1
    if float(a[-2]) != float(b[-2]):
        return -1 if float(a[-2]) > float(b[-2]) else 1
    return 0

def classifyTestDocs():
    unique_topics = set()
    with open("./testInputToCBAClassifier") as fileObject:
        for doc in fileObject:
            docSet = frozenset(doc[:-1].split(" "))
            matched_rules_count = 0
            predicted_classes = []
            for rule in rulesList:
                rule_matched = True
                for word in rule[1:-2]:
                    if word not in docSet:
                        rule_matched = False
                        break
                if rule_matched:
                    # print rule
                    predicted_classes.append(rule[0])
                    unique_topics.add(rule[0])
                    matched_rules_count += 1
                    if len(predicted_classes) >= 1:
                        break
            predictList.append(predicted_classes)
    print unique_topics

def parseTestTopics():
    with open("./testDataRules") as fileObject:
        for topicLine in fileObject:
            topics = topicLine[:-1].split(" ")
            testTopics.append(topics)

def predictAccuracy():
    match_count = 0
    iterator_position = 0
    print "Length********:",len(testTopics)
    print len(predictList)
    for topics in testTopics:
        # print topics
        matched = False
        for topic in topics:
            if topic in predictList[iterator_position]:
                matched = True
                break
        if matched:
            match_count += 1
        iterator_position += 1
    return match_count*100/float(len(predictList))

def plotGraph(plotAccuracy):
    split = 15
    supportList = [item[0] for item in configs[:split]]
    confidenceList = [item[1] for item in configs[split:]]
    plt.xlabel('support')
    plt.xticks(supportList)
    if plotAccuracy:
        plt.ylabel('Accuracy')
        plt.plot(supportList, accuracies[:split], linewidth=2.0, marker='o')
    else:
        plt.ylabel('Time taken')
        plt.plot(supportList, times[:split], linewidth=2.0, marker='o')
    plt.show()
    plt.xlabel('confidence')
    if plotAccuracy:
        plt.plot(confidenceList, accuracies[split:], linewidth=2.0, marker='o')
    else:
        plt.plot(confidenceList, times[split:], linewidth=2.0, marker='o')
    plt.show()
    # plt.setp(lines, color='r', linewidth=2.0)
    # print plot


configs = []
testTopics = []
parseTestTopics()
for i in range(1,16):
    configs.append((i,10))

accuracies = []
times = []
for config in configs:
    start_time = time.time()
    support = "-s" + str(config[0])
    confidence = "-c" + str(config[1])
    parsing_end_time = time.time()
    # #rulesFile = open("./rules.txt", "w+")
    returnValue = check_call(['./apriori', '-tr', support, confidence, '-Rappearances.txt', 'inputToCBAClassifier', './rulesFile'])
    print("\nTime taken to generate rules ----- %s" %(time.time() - parsing_end_time))
    if(returnValue == 0):
        print "Success"
    else:
        print "Failed"
    rulesList = []
    predictList = []
    parseRules()
    rulesList.sort(cmpRules)
    classifyTestDocs()
    # print rulesList
    print predictList
    acc = predictAccuracy()
    print acc
    accuracies.append(acc)
    times.append( time.time() - start_time)
plotGraph(True)
plotGraph(False)

print("\nTime taken to generate rules ----- %s" %(time.time() - parsing_end_time))




# cbaClassifier.predictClasses()
