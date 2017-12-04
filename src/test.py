import ExpectedMax
if __name__ == '__main__':
    m = {}
    for i in range(100):
        print(i)
        maxscore = ExpectedMax.main()
        if maxscore in m:
            m[maxscore] += 1
        else:
            m[maxscore] = 1
        print(m)
    print(m)
