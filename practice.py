import numpy as np

def read_file():
    with open("test.csv") as f:
        contents = f.readlines()[2:]
    passengerInfoMap = [line.strip().split(',') for line in contents]

    return passengerInfoMap

def main():
    passengerMapInfo = read_file()
    print passengerMapInfo
    return

if __name__ == "__main__":
    main()