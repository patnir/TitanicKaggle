import glob
import re


def find_next_file(filename):
    files = [f for f in glob.glob(filename + "*")]
    numbers = [int(re.findall(r"[\d]+", i)[-1]) for i in files]
    n = max(numbers) + 1
    return filename + str(n) + ".csv"


def main():
    filename = "C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\Submissions\\LinearRegression"
    res = find_next_file(filename)
    print(res)

if __name__ == '__main__':
    main()