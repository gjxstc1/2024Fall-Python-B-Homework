# Assignment #6: Recursion and DP

Updated 2201 GMT+8 Oct 29, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### sy119: 汉诺塔

recursion, https://sunnywhy.com/sfbj/4/3/119  

思路：
经典递归问题，本质是每次只能先将n-1个放到中间，再将最大的先挪到右边，然后转化成了n-1元情况。根据已经将多少个放到最右边来递归，注意左和中间柱子名字会变。根据本题输出可以得知是$2^n-1$步


代码：

```python
ansstr = ""
def g(m, left, mid, right):
    global ansstr
    if not m: return 0
    ans = 0
    ans += g(m - 1, left, right, mid) # 除了m以外left全挪到mid
    ansstr += left + '->' + right + '\n'
    ans += g(m - 1, mid, left, right)
    return ans + 1
n = int(input())
print(g(n, 'A', 'B', 'C'))
print(ansstr)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat786019219b99d092465f72827b03afe1-1.jpg)




### sy132: 全排列I

recursion, https://sunnywhy.com/sfbj/4/3/132

思路：
数据量太小可以大复杂度搜索。循环遍历+回溯即可。


代码：

```python
n = int(input())
def g(m, a, used):
    global n
    if m > n:
        print(" ".join(map(str, a)))
        return
    for i in range(1, n + 1):
        if not used[i]:
            used[i] = True
            a.append(i)
            g(m + 1, a, used)
            used[i] = False
            a.pop()
    return
a = []
g(1, a, [False] * (n + 1))
```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](WeChata36df5acc4c932afde251f2a17b89e85-1.jpg)



### 02945: 拦截导弹 

dp, http://cs101.openjudge.cn/2024fallroutine/02945

思路：
求最长单调不下降子序列长度，模版题。$g[j]$表示单调不降子序列长度为$j$时的子序列最后一项的最大值，每次找新来的数能放到的位置（且满足最优性），从大到小循环保证这个最大值会不减小。

代码：

```python
n = int(input())
a = list(map(int, input().split()))
num = 0
g = [float('inf')] * n
for i in range(n):
    for j in range(num, -1, -1):
        if a[i] <= g[j]:
            g[j + 1] = a[i]
            num = max(num, j + 1)
            break
print(num)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat4b12ad123f3baa346ee7e9944138065b-1.jpg)




### 23421: 小偷背包 

dp, http://cs101.openjudge.cn/practice/23421

思路：
0/1背包模版题。


代码：

```python
n, B = map(int, input().split())
v = list(map(int, input().split()))
w = list(map(int, input().split()))
dp = [0] * (B + 1)
for i in range(n):
    for j in range(B, w[i] - 1, -1):
        dp[j] = max(dp[j], dp[j - w[i]] + v[i])
mx = 0
for j in range(B + 1): mx = max(mx, dp[j])
print(mx)
```
![alt text](WeChat5d33e604764d6a179df3810aa181a51d-1.jpg)


代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### 02754: 八皇后

dfs and similar, http://cs101.openjudge.cn/practice/02754

思路：
带回溯的dfs，用两个坐标之和与差来判断斜线情况（+8平移来保证下标非负）。


代码：

```python
import sys
output = sys.stdout.write
ans = []
def g(m, a, used_column, used1, used2):
    global ans
    if m > 8:
        ans.append(a)
        return
    for i in range(1, 9):
        if used_column[i] and used1[m + i] and used2[i - m + 8]:
            used_column[i] = used1[m + i] = used2[i - m + 8] = False
            g(m + 1, a + str(i), used_column, used1, used2)
            used_column[i] = used1[m + i] = used2[i - m + 8] = True
    return
g(1, "", [True] * 9, [True] * 18, [True] * 18)
T =  int(input())
result = []
for __ in range(T): result.append(ans[int(input()) - 1])
output("\n".join(result) + "\n")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat351a943bf34716e2474e2c042c5ccb56-1.jpg)




### 189A. Cut Ribbon 

brute force, dp 1300 https://codeforces.com/problemset/problem/189/A

思路：
可以完全背包（更好，可做多个物品），也可以用扩欧（3个物品才比较优），枚举一个变量对另外两个解不定方程。


代码：

```python
n, a, b, c = map(int, input().split())
a = [a, b, c]
dp = [-1 for i in range(n + 1)]
dp[0] = 0
for i in range(3):
    for j in range(a[i], n + 1):
        if dp[j - a[i]] != -1: dp[j] = max(dp[j], dp[j - a[i]] + 1)
print(dp[n])
```

```python
def swap(a, b): return b, a
def exgcd(a, b):
    if not b:
        return a, 1, 0
    d, x, y = exgcd(b, a % b)
    return d, y, x - a // b * y
n, a, b, c = map(int, input().split()) # ax+by=d
if a < b: a, b = swap(a, b)
d, x0, y0 = exgcd(a, b); ans = 0
a //= d; b //= d
for i in range(n // c + 1):
    if (n - c * i) % d: continue
    C = (n - c * i) // d
    tmpx = (x0 * C % b + b) % b
    if C < a * tmpx: continue
    ans = max(ans, tmpx + (C - a * tmpx) // b + i)
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatd286ca631899c44b387831c41a9ee0a3-1.jpg)

![alt text](WeChat57ca6addc535d86565ca1be04d73cde4-1.jpg)


## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

本周我学习了dp的简单模版和搜索。

本周有多门期中考试，欠下的选做题考完就补上来。



# Assignment #6: Recursion and DP

Updated 2201 GMT+8 Oct 29, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### sy119: 汉诺塔

recursion, https://sunnywhy.com/sfbj/4/3/119  

思路：
经典递归问题，本质是每次只能先将n-1个放到中间，再将最大的先挪到右边，然后转化成了n-1元情况。根据已经将多少个放到最右边来递归，注意左和中间柱子名字会变。根据本题输出可以得知是$2^n-1$步


代码：

```python
ansstr = ""
def g(m, left, mid, right):
    global ansstr
    if not m: return 0
    ans = 0
    ans += g(m - 1, left, right, mid) # 除了m以外left全挪到mid
    ansstr += left + '->' + right + '\n'
    ans += g(m - 1, mid, left, right)
    return ans + 1
n = int(input())
print(g(n, 'A', 'B', 'C'))
print(ansstr)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat786019219b99d092465f72827b03afe1-1.jpg)




### sy132: 全排列I

recursion, https://sunnywhy.com/sfbj/4/3/132

思路：
数据量太小可以大复杂度搜索。循环遍历+回溯即可。


代码：

```python
n = int(input())
def g(m, a, used):
    global n
    if m > n:
        print(" ".join(map(str, a)))
        return
    for i in range(1, n + 1):
        if not used[i]:
            used[i] = True
            a.append(i)
            g(m + 1, a, used)
            used[i] = False
            a.pop()
    return
a = []
g(1, a, [False] * (n + 1))
```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](WeChata36df5acc4c932afde251f2a17b89e85-1.jpg)



### 02945: 拦截导弹 

dp, http://cs101.openjudge.cn/2024fallroutine/02945

思路：
求最长单调不下降子序列长度，模版题。$g[j]$表示单调不降子序列长度为$j$时的子序列最后一项的最大值，每次找新来的数能放到的位置（且满足最优性），从大到小循环保证这个最大值会不减小。

代码：

```python
n = int(input())
a = list(map(int, input().split()))
num = 0
g = [float('inf')] * n
for i in range(n):
    for j in range(num, -1, -1):
        if a[i] <= g[j]:
            g[j + 1] = a[i]
            num = max(num, j + 1)
            break
print(num)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat4b12ad123f3baa346ee7e9944138065b-1.jpg)




### 23421: 小偷背包 

dp, http://cs101.openjudge.cn/practice/23421

思路：
0/1背包模版题。


代码：

```python
n, B = map(int, input().split())
v = list(map(int, input().split()))
w = list(map(int, input().split()))
dp = [0] * (B + 1)
for i in range(n):
    for j in range(B, w[i] - 1, -1):
        dp[j] = max(dp[j], dp[j - w[i]] + v[i])
mx = 0
for j in range(B + 1): mx = max(mx, dp[j])
print(mx)
```
![alt text](WeChat5d33e604764d6a179df3810aa181a51d-1.jpg)


代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### 02754: 八皇后

dfs and similar, http://cs101.openjudge.cn/practice/02754

思路：
带回溯的dfs，用两个坐标之和与差来判断斜线情况（+8平移来保证下标非负）。


代码：

```python
import sys
output = sys.stdout.write
ans = []
def g(m, a, used_column, used1, used2):
    global ans
    if m > 8:
        ans.append(a)
        return
    for i in range(1, 9):
        if used_column[i] and used1[m + i] and used2[i - m + 8]:
            used_column[i] = used1[m + i] = used2[i - m + 8] = False
            g(m + 1, a + str(i), used_column, used1, used2)
            used_column[i] = used1[m + i] = used2[i - m + 8] = True
    return
g(1, "", [True] * 9, [True] * 18, [True] * 18)
T =  int(input())
result = []
for __ in range(T): result.append(ans[int(input()) - 1])
output("\n".join(result) + "\n")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat351a943bf34716e2474e2c042c5ccb56-1.jpg)




### 189A. Cut Ribbon 

brute force, dp 1300 https://codeforces.com/problemset/problem/189/A

思路：
可以完全背包（更好，可做多个物品），也可以用扩欧（3个物品才比较优），枚举一个变量对另外两个解不定方程。


代码：

```python
n, a, b, c = map(int, input().split())
a = [a, b, c]
dp = [-1 for i in range(n + 1)]
dp[0] = 0
for i in range(3):
    for j in range(a[i], n + 1):
        if dp[j - a[i]] != -1: dp[j] = max(dp[j], dp[j - a[i]] + 1)
print(dp[n])
```

```python
def swap(a, b): return b, a
def exgcd(a, b):
    if not b:
        return a, 1, 0
    d, x, y = exgcd(b, a % b)
    return d, y, x - a // b * y
n, a, b, c = map(int, input().split()) # ax+by=d
if a < b: a, b = swap(a, b)
d, x0, y0 = exgcd(a, b); ans = 0
a //= d; b //= d
for i in range(n // c + 1):
    if (n - c * i) % d: continue
    C = (n - c * i) // d
    tmpx = (x0 * C % b + b) % b
    if C < a * tmpx: continue
    ans = max(ans, tmpx + (C - a * tmpx) // b + i)
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatd286ca631899c44b387831c41a9ee0a3-1.jpg)

![alt text](WeChat57ca6addc535d86565ca1be04d73cde4-1.jpg)


## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

本周我学习了dp的简单模版和搜索。

本周有多门期中考试，欠下的选做题考完就补上来。



