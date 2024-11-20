# Assignment #9: dfs, bfs, & dp

Updated 2107 GMT+8 Nov 19, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 18160: 最大连通域面积

dfs similar, http://cs101.openjudge.cn/practice/18160

思路：
直接做


代码：

```python
T = int(input())
directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
for __ in range(T):
    n, m = map(int, input().split())
    a = [input() for _ in range(n)]
    flag = [[False] * m for _ in range(n)]
    cnt = 0; mx = 0
    for i in range(n):
        for j in range(m):
            if a[i][j] == '.':
                flag[i][j] = True
    for x in range(n):
        for y in range(m):
            if not flag[x][y]:
                q = [(x, y)]
                flag[x][y] = True
                cnt = 0
                while q:
                    cx, cy = q.pop()
                    cnt += 1
                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < n and 0 <= ny < m and not flag[nx][ny]:
                            flag[nx][ny] = True
                            q.append((nx, ny))
                mx = max(mx, cnt)
    print(mx)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatfd0febd89d5235ad4af536b3d3f4634b-1.jpg)




### 19930: 寻宝

bfs, http://cs101.openjudge.cn/practice/19930

思路：
直接做，bfs可以用队列写


代码：

```python
from collections import deque
n, m = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(n)]
flag = [[False] * m for _ in range(n)]
directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
q = deque()
q.append((0, 0, 0)); flag[0][0] = True
while q:
    x, y, z = q.popleft()
    if a[x][y] == 1:
        print(z)
        exit(0)
    for nx, ny in directions:
        tx, ty = x + nx, y + ny
        if 0 <= tx < n and 0 <= ty < m and not flag[tx][ty] and a[tx][ty] != 2:
            flag[tx][ty] = True
            q.append((tx, ty, z + 1))
print("NO")
```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChat0a1776de27dcba9b93baeae9d6a6ec04-1.jpg)




### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123

思路：
直接做


代码：

```python
def dfs(x, y, cnt):
    if cnt == n * m: return 1
    ans = 0
    for nx, ny in directions:
        tx, ty = x + nx, y + ny
        if 0 <= tx < n and 0 <= ty < m and not flag[tx][ty]:
            flag[tx][ty] = True
            ans += dfs(tx, ty, cnt + 1)
            flag[tx][ty] = False
    return ans
T = int(input())
directions = [(1, -2), (1, 2), (-1, 2), (-1, -2), (2, -1), (2, 1), (-2, 1), (-2, -1)]
for __ in range(T):
    n, m, sx, sy = map(int, input().split())
    flag = [[False] *  m for _ in range(n)]
    flag[sx][sy] = True
    print(dfs(sx, sy, 1))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat06c1e7417eaf7e1b88944444dd2d7dbe-1.jpg)




### sy316: 矩阵最大权值路径

dfs, https://sunnywhy.com/sfbj/8/1/316

思路：
直接做。


代码：

```python
def dfs(x, y, z):
    global mx, result
    if (x, y) == (n - 1, m - 1):
        if z > mx: mx, result = z, cur[:]
        return
    for nx, ny in directions:
        tx, ty = x + nx, y + ny
        if 0 <= tx < n and 0 <= ty < m and not flag[tx][ty]:
            flag[tx][ty] = True
            cur.append((tx, ty))
            dfs(tx, ty, z + a[tx][ty])
            cur.pop()
            flag[tx][ty] = False
n, m = map(int, input().split())
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
a = [list(map(int, input().split())) for _ in range(n)]
flag = [[False] *  m for _ in range(n)]
flag[0][0] = True
mx = -10000
result = []; cur = [(0, 0)]
dfs(0, 0, a[0][0])
for x in result: print(x[0] + 1, x[1] + 1)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat7a6438778c36049dc5fd4dab3737ea1a-1.jpg)






### LeetCode62.不同路径

dp, https://leetcode.cn/problems/unique-paths/

思路：
标数法。


代码：

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        f = [[0] * n for i in range(m)]
        for i in range(m): f[i][0] = 1
        for j in range(n): f[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                f[i][j] = f[i - 1][j] + f[i][j - 1]
        return f[-1][-1]
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChata07656f32a9a44223910e97d624719ca-1.jpg)




### sy358: 受到祝福的平方

dfs, dp, https://sunnywhy.com/sfbj/8/3/539

思路：
把合法的点位放到一个数组里，也许有点dp想法但不是dp。


代码：

```python
a = input()
n = len(a)
valid = [-1]
def check(x):
    if not x: return False
    tmp = int(x ** 0.5)
    return tmp * tmp == x
for i in range(n):
    for t in valid:
        if check(int(a[t + 1:i + 1])):
            valid.append(i)
            break
print("Yes" if n - 1 in valid else "No")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChataed42e80c3f5d3ac3b87b8dda5b50e64-1.jpg)




## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

做完本周所有选做题。

学到了list的拷贝（两个独立）和引用（两个变量引用同一个list，对一个列表的修改同时反映在另一个列表上）是不同的。

学习过程中还是不太清楚：什么时候必须用global什么时候不必用，我猜是不是list的深拷贝才必须用，浅拷贝不用？还是都要用？以及元素本身在递归中不能直接修改（需要global声明），但是list内部的数是可以修改的（不用global），这又很神奇...




