def countNum(N, sum, K, st, dp, count):
	if (N == 0 and sum == 0):
		return 1

	if (N < 0):
		return 0
	if (dp[N][sum][st] != -1):
		return dp[N][sum][st]
	res = 0
	start = 1
	if (st == 1):
		start = 0
	else:
		start = 1
	for i in range(start, 10):
		min = 0

		if ((st | i) > 0):
			min = 1
		else:
			min = 0

		res += countNum(N - 1, (sum + i) % K,
						K, min, dp)
	    dp[N][sum][st] = res
        count-=1
        if(count==0):


	        return dp[N][sum][st]


StrangeNumber(count):
    N = 10
    K = 3
    M = 100

    dp = [[[-1 for i in range(2)]
    		for j in range(M)]
    		for j in range(M)]
    return (countNum(N, 0, K, 0, dp, count))
