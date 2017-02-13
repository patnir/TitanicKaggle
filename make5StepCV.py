import random

def getDataMap():
    with open("train.csv") as f:
        contents = f.readlines()[2:]
    dataMap = [line.strip().split(',') for line in contents]
    nDataMap = [[id,sv,c,fn[1:].strip(),ln.split('.')[0].strip(), ln.split('.')[1][:-1].strip(),s,a,ss,pg,t,f,ca,e] for id,sv,c,fn,ln,s,a,ss,pg,t,f,ca,e in dataMap]
    return nDataMap

def printMap(pMap):
    for i in pMap:
        print i

def separateData(data):
    r = [random.randint(0, 4) for i in range(len(data))]
    sData = [None] * 5
    for i in range(len(sData)):
        sData[i] = [[], []]
    print sData
    track = 0
    for i in r:
        for j in range(5):
            if i == j:
                sData[j][1].append(data[track])
            else:
                sData[j][0].append(data[track])
        track += 1
    return sData

def writeDataIntoFiles(data):
    for curr in range(len(data)):
        director = "cvSplit/data" + str(curr + 1)
        file = director + "/train.csv"
        print file
        f = open(file, "w+")
        for x in data[curr][0]:
            #f.write(x)
            s = ",".join(x)
            print s
            s = s + "\n"
            f.write(s)
        file = director + "/test.csv"
        print file
        f.close()
        f = open(file, "w+")
        for y in data[curr][1]:
            s = ",".join(y)
            print s
            s = s + "\n"
            f.write(s)
        f.close()
    return

def main():
    dataMap = getDataMap()
    printMap(dataMap)
    cvData = separateData(dataMap)
    #print cvData
    writeDataIntoFiles(cvData)
    return

if __name__ == "__main__":
    main()