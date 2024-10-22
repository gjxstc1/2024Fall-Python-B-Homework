# Assignment #5: Greedy穷举Implementation

Updated 1939 GMT+8 Oct 21, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 04148: 生理周期

brute force, http://cs101.openjudge.cn/practice/04148

思路：
中国剩余定理即可，甚至可以提前算出各种数论逆元。$O(n)，$没有常数。


代码：

```python
cnt = 0
while True:
    a1, a2, a3, d = map(int, input().split())
    if a1 == -1: break
    cnt += 1
    a1 %= 23; a2 %= 28; a3 %= 33
    j = ((11 * (a2 - a1) % 28) + 28) % 28
    k = (2 * (a3 - a1 - 23 * j) % 33 + 33) % 33
    P = 23 * 28 * 33
    x = (23 * 28 * k + 23 * j + a1) % P
    d %= P
    if d < x: print("Case {}: the next triple peak occurs in {} days.".format(cnt, x - d))
    else: print("Case {}: the next triple peak occurs in {} days.".format(cnt, x + P - d))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat840189b19d6a9fcedfaa32dc0953313f-1.jpg)




### 18211: 军备竞赛

greedy, two pointers, http://cs101.openjudge.cn/practice/18211

思路：
显然买便宜的卖贵的，排序。尽可能买，实在不行了就卖，这样遍历所有较优的可能性。时刻更新最大值作为答案。


代码：

```python
money = int(input())
a = list(map(int, input().split()))
n = len(a)
a = sorted(a, reverse = True)
l = 0; r = n - 1
ans = 0
now = 0
while l <= r:
    if money - a[r] < 0:
        if not now: break
        now -= 1
        money += a[l]
        l += 1
        ans = max(ans, now)
        continue
    else:
        money -= a[r]
        now += 1
        ans = max(ans, now)
        r -= 1
print(ans)
```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChat842aefc1accf1a8bd52b2edd8a30954d-1.jpg)




### 21554: 排队做实验

greedy, http://cs101.openjudge.cn/practice/21554

思路：
显然让花时间少的人先做最优（可以通过分析最优情况不能交换来严格证明）。用上了enumerate。


代码：

```python
n = int(input())
a = [(i, v) for i, v in enumerate(list(map(int, input().split())))]
a = sorted(a, key = lambda x: x[1])
for i, v in a: print(i + 1, end = " ")
print()
ans = 0
for i, v in enumerate(a): ans += v[1] * (n - 1 - i)
print(f"{ans / n:.2f}")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat0305c24d7b02dd39f893658f505d0645-1.jpg)




### 01008: Maya Calendar

implementation, http://cs101.openjudge.cn/practice/01008/

思路：
简单数学题，有点像罗马数字的模拟。用上了字典。


代码：

```python
month1 =\
    "pop, no, zip, zotz, tzec, xul, yoxkin, mol, chen, yax, zac, ceh, mac, kankin, muan, pax, koyab, cumhu".split(", ")
month2 =\
    "imix, ik, akbal, kan, chicchan, cimi, manik, lamat, muluk, ok, chuen, eb, ben, ix, mem, cib, caban, eznab, canac, ahau".split(", ")
chk1 = {}
for i in range(18): chk1[month1[i]] = i
chk1["uayet"] = 18
n = int(input()); print(n)
for i in range(n):
    x, y1, y2 = input().split()
    day = int(y2) * 365 + chk1[y1] * 20 + int(x.rstrip('.'))
    print("{} {} {}".format(day % 13 + 1, month2[day % 20], day // 260))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChate6847bc4822d86267ec2b53e6842d6ef-1.jpg)




### 545C. Woodcutters

dp, greedy, 1500, https://codeforces.com/problemset/problem/545/C

思路：
贪心，关键点在于如果一个树要往右边倒，必须要不盖住下一个点才行（否则直接盖住下一个点，而这是非法的，我在这里困扰了一阵）。因此之后一段一直向右边倒，直到有一棵树能向左边倒结束。注意特判n=1。

本题dp更显然。也附上了dp做法。


代码：

```python
n = int(input())
a = []
for i in range(n):
    x, h = map(int, input().split())
    a.append([x, h])
ans = 2
if n == 1:
    print(1)
    exit(0)
for i in range(1, n - 1):
    if a[i][0] - a[i - 1][0] > a[i][1]: ans += 1
    elif a[i][0] + a[i][1] < a[i + 1][0]:
        ans += 1
        a[i][0] += a[i][1]
print(ans)
```


```python
n = int(input())
a = []
for i in range(n):
    x, h = map(int, input().split())
    a.append([x, h])
a.append([float('inf'), 0])
dp = [[0, 0, 0] for _ in range(n)] # 0 不变 1 left 2 right
if n == 1:
    print(1)
    exit(0)
dp[0][0] = dp[0][1] = 1
if a[0][0] + a[0][1] < a[1][0]: dp[0][2] = 1
for i in range(1, n):
    dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
    if a[i - 1][0] + a[i - 1][1] < a[i][0]: dp[i][0] = max(dp[i][0], dp[i - 1][2])
    if a[i - 1][0] + a[i][1] < a[i][0]:
        dp[i][1] = max(dp[i - 1][0] + 1, dp[i - 1][1] + 1)
        if a[i - 1][0] + a[i - 1][1] + a[i][1] < a[i][0]: dp[i][1] = max(dp[i][1], dp[i - 1][2] + 1)
    if a[i][0] + a[i][1] < a[i + 1][0]:
        dp[i][2] = max(dp[i - 1][0], dp[i - 1][1]) + 1
        if a[i - 1][0] + a[i - 1][1] < a[i][0]: dp[i][2] = max(dp[i][2], dp[i - 1][2] + 1)
mx = 0
mx = max(mx, dp[n - 1][0], dp[n - 1][1], dp[n - 1][2])
print(mx)
```


代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatd54422bd13592dcdb4aece6e36a7297e-1.jpg)

![alt text](WeChatff6ab6c013928fccd7748b9ddf2044b4-1.jpg)


### 01328: Radar Installation

greedy, http://cs101.openjudge.cn/practice/01328/

思路：
简单题目。每个点对应合法的雷达选点是一个区间，故转化成选若干个点使得每个区间包含至少一个点，这个显然贪心即可。


代码：

```python
cnt = 0
while True:
    cnt += 1; flag = True
    n, d = map(int, input().split())
    if not n: break
    print("Case {}: ".format(cnt), end="")
    a = []
    for i in range(n):
        x, y = map(int, input().split())
        if not flag: continue
        if abs(y) > d:
            flag = False
            continue
        a.append((x - (d * d - y * y) ** 0.5, x + (d * d - y * y) ** 0.5))
    input()
    if not flag:
        print(-1)
        continue
    a = sorted(a, key = lambda x: x[1])
    l = 0; ans = 0
    while l < n:
        ans += 1
        tmp = a[l][1]
        while l < n and a[l][0] <= tmp: l += 1
    print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatdc307d0ce8c0d4b9304a9d99c17bce48-1.jpg)




## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

本周学习了更深层的贪心算法。考虑贪心问题有一些常用的思维方式；以及有时候严格的证明容易被绕进去，感觉上没有反例就应该直接写;还有就是可以根据数据范围猜测是否用贪心。

同时我也用上了之前学习的enumerate、字典及其他语法，很有收获感。之后想学习一些heapq中的函数。

每日选做等期中后再集中赶上来。
