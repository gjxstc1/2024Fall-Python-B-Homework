# Assignment #8: 田忌赛马来了

Updated 1021 GMT+8 Nov 12, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 12558: 岛屿周⻓

matices, http://cs101.openjudge.cn/practice/12558/ 

思路：
搜索。


代码：

```python
n, m = map(int, input().split())
a = [[0] * (m + 2)]
for i in range(n): a.append([0] + list(map(int, input().split())) + [0])
a.append([0] * (m + 2))
flag = [[False] * (m + 2) for i in range(n + 2)]
def dfs(x, y):
    global flag
    if not x or not y or x == n + 1 or y == m + 1 or not a[x][y] or flag[x][y]: return 0
    flag[x][y] = True
    return 4 - a[x - 1][y] - a[x + 1][y] - a[x][y - 1] - a[x][y + 1]\
            + dfs(x - 1, y) + dfs(x + 1, y) + dfs(x, y - 1) + dfs(x, y + 1)
for i in range(1, n + 1):
    for j in range(1, m + 1):
        if a[i][j]:
            print(dfs(i, j))
            exit(0)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat9e063727b483f79470ec9f925165833a-1.jpg)




### LeetCode54.螺旋矩阵

matrice, https://leetcode.cn/problems/spiral-matrix/

与OJ这个题目一样的 18106: 螺旋矩阵，http://cs101.openjudge.cn/practice/18106

思路：
按填数顺序搜索。


代码：

```python
mov = [[0, 1], [1, 0], [0, -1], [-1, 0]]
def dfs(x, y, num, t): # t = 0 r, t = 1 d, t = 2 l, t = 3 u
    if num > n * n: return
    global a
    a[x][y] = num
    x1 = x + mov[t][0]; y1 = y + mov[t][1]
    if x1 < 0 or x1 >= n or y1 < 0 or y1 >= n or a[x1][y1]:
        t = (t + 1) % 4
        dfs(x + mov[t][0], y + mov[t][1], num + 1, t)
        return
    dfs(x1, y1, num + 1, t)
    return
n = int(input())
a = [[0] * n for _ in range(n)]
dfs(0, 0, 1, 0)
for row in a: print(*row)
```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChat9eaf4ba89e5705e19104eaa9820ce934-1.jpg)
![alt text](WeChatdf44ee370a126f36042fc211344d7381-1.jpg)




### 04133:垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/

思路：
注意到n,d很小而范围很大，所以只用枚举所有能清除到某个垃圾的所有点（包含可能的所有答案），注意特判重复。复杂度较低$O(n^2*d^2)$。

特别地如果不用求选点的个数而只求最大值则可以降低到$O(n ^ 2)$，因为可以调整选点到其恰好能扫到某个垃圾（故可以改成
```python
for i in [max(0, a[k][0] - d), min(a[k][0] + d, 1024) + 1]:
        for j in [max(0, a[k][1] - d), min(a[k][1] + d, 1024) + 1]:
```
）。


代码：

```python
from collections import defaultdict
d = int(input())
n = int(input())
a = [list(map(int, input().split())) for i in range(n)]
chk = defaultdict(bool)
mx = 0; cnt = 0
for k in range(n):
    for i in range(max(0, a[k][0] - d), min(a[k][0] + d, 1024) + 1):
        for j in range(max(0, a[k][1] - d), min(a[k][1] + d, 1024) + 1):
            if chk[i, j]: continue
            chk[i, j] = True
            ans = 0
            for l in range(n):
                if abs(a[l][0] - i) <= d and abs(a[l][1] - j) <= d: ans += a[l][2]
            if ans > mx: cnt = 1; mx = ans
            elif ans == mx: cnt += 1
print(" ".join(map(str, [cnt, mx])))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat49be213d543ceb9a3bc231c2b98693de-1.jpg)




### LeetCode376.摆动序列

greedy, dp, https://leetcode.cn/problems/wiggle-subsequence/

与OJ这个题目一样的，26976:摆动序列, http://cs101.openjudge.cn/routine/26976/

思路：

dp[i][0] 最长摆动序列长度 (最后一个 < 上一个)

dp[i][1] 最长摆动序列长度 (最后一个 > 上一个)


代码：

```python
n = int(input())
a = list(map(int, input().split()))
dp = [[1, 1] for _ in range(n)]
# dp[i][0] 最长摆动序列长度 (最后一个 < 上一个)
# dp[i][1] 最长摆动序列长度 (最后一个 > 上一个)
ans = 1
for i in range(1, n):
    if a[i] < a[i - 1]:
        dp[i][0] = dp[i - 1][1] + 1
        dp[i][1] = dp[i - 1][1]
    elif a[i] > a[i - 1]:
        dp[i][1] = dp[i - 1][0] + 1
        dp[i][0] = dp[i - 1][0]
    else:
        dp[i][1] = dp[i - 1][1]
        dp[i][0] = dp[i - 1][0]
print(max(dp[n - 1][0], dp[n - 1][1]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat1f2ac9a0b2cac82a595c4386df13c34f-1.jpg)
![alt text](WeChate292a0341e009f1124a3d043ffce0c78-1.jpg)



### CF455A: Boredom

dp, 1500, https://codeforces.com/contest/455/problem/A

思路：
简单转化到问题：“取若干不相邻的数，求最大的这些数权值之和”。dp。


代码：

```python
M = int(1e5)
a = [0] * (M + 1)
n = int(input())
for x in map(int, input().split()): a[x] += 1
dp = [[0, 0] for _ in range(M + 1)]
# dp[i][0] 不选i, dp[i][1] 选i
for i in range(1, M + 1):
    dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
    dp[i][1] = dp[i - 1][0] + a[i] * i
print(max(dp[M][0], dp[M][1]))

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat514aef5399b46f4f137009ba4927ba76-1.jpg)




### 02287: Tian Ji -- The Horse Racing

greedy, dfs http://cs101.openjudge.cn/practice/02287

思路：
很有想法的贪心，貌似区间dp也能做。每次比较两人最大的马，如果不相等则很容易贪心；关键在于相等情况：此时比较最小的马，如果田能赢则也很容易贪心；

否则让田最小马对王最大马，对剩余情况继续求最优解。这是因为最优解包含了此时田最大马对王最小马的情况，因此这时结果不差于田最大马打王最大马的结果（可以设此时田最小马打谁（必不胜），可以交换发现结果不劣）。故可以让田最小马对王最大马。


代码：

```python
while True:
    n = int(input())
    if not n: break
    a = sorted(list(map(int, input().split())))
    b = sorted((map(int, input().split())))
    l1 = 0; r1 = n - 1; l2 = 0; r2 = n - 1; ans = 0
    for ___ in range(n):
        if a[r1] > b[r2]:
            ans += 1
            r1 -= 1; r2 -= 1
            continue
        if a[r1] < b[r2]:
            ans -= 1
            l1 += 1; r2 -= 1
            continue
        #a[r1] = b[r2]
        if a[l1] <= b[l2]:
            if a[l1] < b[r2]: ans -= 1
            l1 += 1; r2 -= 1
        else:
            ans += 1
            l1 += 1; l2 += 1
    print(ans * 200)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChata308a57d1e4592e3f66f4f3d7917908a-1.jpg)




## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

不知道为啥leetcode提交的代码要写到这么奇怪的框架...

本周学习了递归+dp。大概过去一周的计概2024fall每日选做都补上来了。同时又学了一些不熟的语法（月考暴露的问题）、了解了一些厉害的库中的函数。



