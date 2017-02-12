import sys

def getDataMap():
    with open("test.csv") as f:
        contents = f.readlines()[2:]
    dataMap = [line.strip().split(',') for line in contents]
    nDataMap = [[id,c,fn[1:],ln.split('.')[0], ln.split('.')[1][:-1],s,a,ss,pg,t,f,ca,e] for id,c,fn,ln,s,a,ss,pg,t,f,ca,e in dataMap]
    return nDataMap

def printMap(pMap):
    for i in pMap:
        print i

def main():
    dataMap = getDataMap()
    printMap(dataMap)

    return

if __name__ == "__main__":
    main()