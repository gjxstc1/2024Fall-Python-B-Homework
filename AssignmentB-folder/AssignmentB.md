# Assignment #B: Dec Mock Exam大雪前一天

Updated 1649 GMT+8 Dec 5, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）⽉考： AC<mark>6</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E22548: 机智的股民老张

http://cs101.openjudge.cn/practice/22548/

思路：
简单贪心，直接做。


代码：

```python
a = list(map(int, input().split()))
n = len(a)
ans = 0
mx = [0] * (n + 1)
for i in range(n - 1, -1, -1): mx[i] = max(mx[i + 1], a[i])
for i in range(n): ans = max(ans, mx[i] - a[i])
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat29a902e5595cb80d3b658c9a548c5df3-1.jpg)




### M28701: 炸鸡排

greedy, http://cs101.openjudge.cn/practice/28701/

思路：
考试最难题，考场做法是乱猜的贪心，运气很好过了。不太会证明算法正确性。

先特判显然的情况：对$a$排序后，如果有$t \leq k-1$个鸡时间很长，能够耗掉剩余所有鸡，则答案已找到。具体为这$t$个鸡一直在锅里可以让剩余所有鸡都熟了，即$t$从大到小验证，直到碰到第一个使得$min(a[0$~$t-1])=a[t]>=其余的鸡在剩余k-t个位置的平均值=\frac{1}{k-t} \sum_{i=t}^{n-1}a[i]$。因为从大到小判断，这保证了剩余的鸡中没有"时间比其他鸡显著长"的鸡，故剩余鸡在剩下的$k-t$个位置中可以保证“平均分配”时间（猜的，应该能证明），这能保证时间最长，因此总时间由剩余鸡在剩下的$k-t$个位置的时间决定，即$\frac{1}{k-t} \sum_{i=t}^{n-1}a[i]$。如果不存在这样的$t$，则没有"时间比其他鸡显著长"的鸡，故可“平均分配”时间（猜的，应该能证明）


代码：

```python
n, k = map(int, input().split())
t = sorted(list(map(int, input().split())), reverse = True)
if k == 1:
    print(f"{sum(t):.3f}")
    exit(0)
x = sum(t)
y = sum(t[k:n])
for i in range(k - 2, -1, -1):
    y += t[i + 1]
    if t[i] >= y / (k - 1 - i):
        print(f"{y / (k - 1 - i):.3f}")
        exit(0)
print(f"{x / k:.3f}")
```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChat004760ca5686d44a0091dac808dc6b93-1.jpg)




### M20744: 土豪购物

dp, http://cs101.openjudge.cn/practice/20744/

思路：
不知道贪心有没有简单方法（最终取的一定是连续的正负交替段），但感觉非常难写

因此考场上写dp, $dp[i][0]=以i结尾，i必须取的连续一段，并且不退回的最大值;dp[i][1] = 以i结尾，i必须取的连续一段，并且可以退回1个的最大值$ $（dp[i][1]蕴含了dp[i][0]的情况，因为如果必须退回1个会写起来复杂。这仍然正确，因为dp[i][0]只会由dp[i-1][0]转移得到，不由dp[i-1][1]转移）$

代码：

```python
a = list(map(int, input().split(",")))
n = len(a)
mx = float("-inf")
dp = [[0, 0] for _ in range(n + 1)]
dp[-1][0] = dp[-1][1] = float("-inf")
for i in range(n):
    if a[i] >= 0:
        dp[i][0] = max(dp[i - 1][0] + a[i], a[i])
        dp[i][1] = max(dp[i - 1][1] + a[i], dp[i - 1][0] + a[i], a[i])
    else:
        dp[i][0] = max(dp[i - 1][0] + a[i], a[i])
        dp[i][1] = max(dp[i - 1][0], dp[i - 1][1] + a[i])
    mx = max(mx, dp[i][0], dp[i][1])
#print(dp)
print(mx)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat593e72efd766def469066b9e3555400e-1.jpg)




### T25561: 2022决战双十一

brute force, dfs, http://cs101.openjudge.cn/practice/25561/

思路：
有一点点细节的模拟。题意清晰，坑点都明确写在题目中了。读入又是非常不友好。


代码：

```python
def dfs(x):
    global ans
    if x == n:
        tmp0 = sum(choice)
        tmp0 -= (tmp0 // 300) * 50
        for i in range(m):
            tmp1 = 0
            for j in range(len(q[i])):
                if choice[i] >= q[i][j][0]: tmp1 = max(tmp1, q[i][j][1])
            tmp0 -= tmp1
        ans = min(ans, tmp0)
        return
    for i in range(len(a[x])):
        choice[a[x][i][0]] += a[x][i][1]
        dfs(x + 1)
        choice[a[x][i][0]] -= a[x][i][1]
n, m = map(int, input().split())
ans = float("inf")
a = []
q = []
for i in range(n):
    x = list(input().split())
    tmp = []
    cnt = 0
    for t in x:
        gjx, stc = t.split(":")
        dian = int(gjx)
        price = int(stc)
        tmp.append([dian - 1, price])
        cnt += 1
    a.append(tmp)
for j in range(m):
    x = list(input().split())
    tmp = []
    cnt = 0
    for t in x:
        gjx, stc = t.split("-")
        hello = int(gjx)
        world = int(stc)
        tmp.append([hello, world])
        cnt += 1
    q.append(tmp)
choice = [0] * m
dfs(0)
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatf38c35f98e43c0ba40f895876ad9051d-1.jpg)



### T20741: 两座孤岛最短距离

dfs, bfs, http://cs101.openjudge.cn/practice/20741/

思路：
初始随便找到一个岛，然后bfs扩张即可


代码：

```python
from collections import deque

def dfs(x, y):
    for nx, ny in directions:
        tx, ty = x + nx, y + ny
        if 0 <= tx < n and 0 <= ty < n and not flag[tx][ty] and a[tx][ty] == "1":
            flag[tx][ty] = True
            q.append((tx, ty, 0))
            dfs(tx, ty)
n = int(input())
a = []
directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
for i in range(n): a.append(list(input()))
flag = [[False] * n for i in range(n)]
q = deque()
tag = False
for i in range(n):
    for j in range(n):
        if a[i][j] == '1' and not flag[i][j]:
            flag[i][j] = True
            q.append((i, j, 0))
            dfs(i, j)
            tag = True
            break
    if tag: break
while q:
    tmp = q.popleft()
    for nx, ny in directions:
        tx, ty = tmp[0] + nx, tmp[1] + ny
        if 0 <= tx < n and 0 <= ty < n and not flag[tx][ty]:
            if a[tx][ty] == "1":
                print(tmp[2])
                exit(0)
            flag[tx][ty] = True
            q.append((tx, ty, tmp[2] + 1))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat4f6aa0379ca71b427c14ebea0401b407-1.jpg)




### T28776: 国王游戏

greedy, http://cs101.openjudge.cn/practice/28776

思路：推导贪心的式子, $(c,d)排在(e,f)前面（交换这两个会更劣）可以推出max(c/f,1/d)<=max(1/f,e/d)，其一个充分条件是cd \leq ef, 并且这是全序关系，故按乘积升序排列最优$。

后来发现六年级时练习高精度做过这题，很感慨。

代码：

```python
n = int(input())
x0, y0 = map(int, input().split())
a = [list(map(int, input().split())) for i in range(n)]
a = sorted(a, key = lambda x: x[0] * x[1])
ans = -1
for i in range(n):
    ans = max(ans, x0 // a[i][1])
    x0 *= a[i][0]
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatde2df30e3116ac62d38127973cf77e03-1.jpg)




## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

最近两周比较忙（有好多论文要写）所以落下了选做题，之后尽量补上来。很焦虑期末机考和笔试。

月考暴露的问题：对基础语法不熟练（因为平常做题老是有自动补全），比如deque的popleft尝试了一会儿；犹豫三维list能不能写len函数求某行有多少个二维数组（机房本地pycharm有warning但场下发现可以用）。以及之后要加强推式子的能力（dp、贪心），考场上推了较长时间。



