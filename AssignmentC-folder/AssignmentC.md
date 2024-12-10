# Assignment #C: 五味杂陈 

Updated 1148 GMT+8 Dec 10, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 1115. 取石子游戏

dfs, https://www.acwing.com/problem/content/description/1117/

思路：
数学题，关键在于注意到$x \geq 2y$时先手必胜（讨论（x%y, y）是当时的先手胜还是后手胜），因此只用考虑$<$,此时唯一取法。因此复杂度同辗转相除。


代码：

```python
def solve(x, y): # 1 = 先手胜
    if not y: return False
    if x >= 2 * y: return True
    return not solve(y, x - y)

def main():
    while True:
        a, b = map(int, input().split())
        if a == b == 0: break
        print("win" if solve(max(a, b), min(a, b)) else "lose")

if __name__ == '__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat8c8063da1797d35d931351545e9456d5-1.jpg)




### 25570: 洋葱

Matrices, http://cs101.openjudge.cn/practice/25570

思路：
前缀和练习，一个坑点是如果中间只有一个元素和是普通的层是不一样的！（因为会加四次减四次减没）


代码：

```python
def main():
    n = int(input())
    a = [list(map(int, input().split())) for __ in range(n)]
    row = [[0] * (n + 1) for __ in range(n + 1)]
    column = [[0] * (n + 1) for __ in range(n + 1)]
    for i in range(n):
        for j in range(n):
            row[i][j] = row[i][j - 1] + a[i][j]
    for i in range(n):
        for j in range(n):
            column[i][j] = column[i][j - 1] + a[j][i]
    ans = 0
    for i in range(n >> 1):
        ans = max(ans, - a[i][i] - a[i][n - 1 - i] - a[n - 1 - i][i] - a[n - 1 - i][n - 1 - i] \
                  + row[i][n - i - 1] - row[i][i - 1] + row[n - i - 1][n - i - 1] - row[n - i - 1][i - 1] \
                  + column[i][n - i - 1] - column[i][i - 1] + column[n - i - 1][n - i - 1] - column[n - i - 1][i - 1])
    if n & 1: ans = max(ans, a[n >> 1][n >> 1])
    print(ans)
if __name__ == '__main__':
    main()
```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](WeChat10872478175e347442230afafbd6037b-1.jpg)



### 1526C1. Potions(Easy Version)

greedy, dp, data structures, brute force, *1500, https://codeforces.com/problemset/problem/1526/C1

思路：
带反悔贪心模版题，用heapq（注意是从小到大排序！）


代码：

```python
import heapq
def main():
    n = int(input())
    a = list(map(int, input().split()))
    q = []
    ans = 0
    now = 0
    for x in a:
        if now + x >= 0:
            now += x
            ans += 1
            if x < 0: heapq.heappush(q, x)
        elif q:
            tmp = q[0]
            if x > tmp:
                now += x - tmp
                heapq.heappop(q)
                heapq.heappush(q, x)
    print(ans)

if __name__ == '__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat20b2f8ae5ebc0a1e1df8d0ddce194da0-1.jpg)




### 22067: 快速堆猪

辅助栈，http://cs101.openjudge.cn/practice/22067/

思路：
参考题解，学到了可以建立两个stack，另一个stack是对应时间的所有猪的min，在pop时同时也pop，正好得到之前时间的min值。


代码：

```python
def main():
    q = [] # 所有猪，stack
    mn = [] # 每次push后的min猪，和q对应 stack
    while True:
        try:
            s = input().split() # 这时s是list
            if s[0] == "pop":
                if q and mn: q.pop(); mn.pop()
            if s[0] == "push":
                x = int(s[1])
                q.append(x)
                if not mn: mn.append(x)
                else: mn.append(min(mn[-1], x))
            if s[0] == "min":
                if q and mn: print(mn[-1])
        except EOFError: break

if __name__ == '__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChateb8cc00fea57975ed8d39e041683bd2d-1.jpg)




### 20106: 走山路

Dijkstra, http://cs101.openjudge.cn/practice/20106/

思路：
用单源最短路算法是因可能最优解是绕一个大弯到终点，不是简单第一次碰到就是答案。第一个方法是spfa算法，能处理负边权，在稀疏图中比dij快，但是在稠密图中更慢，复杂度最差为$O(NM)$；第二个方法是dijkstra算法，s不能处理负边权，但是稳定$O((N + M)\log N)$


代码：

```python
from collections import deque
def solve(sx, sy, ex, ey):
    q = deque([(sx, sy)])
    if a[sx][sy] == '#' or a[ex][ey] == '#':
        print("NO")
        return
    flag = [[False] * m for _ in range(n)] # 是否 in q
    flag[sx][sy] = True
    d = [[float('inf')] * m for _ in range(n)]
    d[sx][sy] = 0
    while q:
        tmp = q.popleft()
        flag[tmp[0]][tmp[1]] = False #重要的一行！
        for nx, ny in directions:
            tx, ty = tmp[0] + nx, tmp[1] + ny
            if 0 <= tx < n and 0 <= ty < m and a[tx][ty] != "#":
                if abs(int(a[tmp[0]][tmp[1]]) - int(a[tx][ty])) + d[tmp[0]][tmp[1]] >= d[tx][ty]: continue
                d[tx][ty] = abs(int(a[tmp[0]][tmp[1]]) - int(a[tx][ty])) + d[tmp[0]][tmp[1]]
                if not flag[tx][ty]:
                    flag[tx][ty] = True
                    q.append((tx, ty))
    print("NO" if d[ex][ey] == float('inf') else d[ex][ey])

directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
n, m, T = map(int, input().split())
a = [input().split() for __ in range(n)]
for __ in range(T):
    startx, starty, endx, endy = map(int, input().split())
    solve(startx, starty, endx, endy)
```

```python
from collections import deque
import heapq
def solve(sx, sy, ex, ey):
    q = []
    if a[sx][sy] == '#' or a[ex][ey] == '#':
        print("NO")
        return
    flag = [[False] * m for _ in range(n)] # 是否曾in q过
    d = [[float('inf')] * m for _ in range(n)]
    heapq.heappush(q, (0, sx, sy))
    d[sx][sy] = 0
    while q:
        dist, x, y = heapq.heappop(q)
        if x == ex and y == ey: break
        if flag[x][y]: continue
        flag[x][y] = True
        for nx, ny in directions:
            tx, ty = x + nx, y + ny
            if 0 <= tx < n and 0 <= ty < m and a[tx][ty] != "#":
                if abs(int(a[x][y]) - int(a[tx][ty])) + dist >= d[tx][ty]: continue
                d[tx][ty] = abs(int(a[x][y]) - int(a[tx][ty])) + dist
                heapq.heappush(q, (d[tx][ty], tx, ty))
    print("NO" if d[ex][ey] == float('inf') else d[ex][ey])

directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
n, m, T = map(int, input().split())
a = [input().split() for __ in range(n)]
for __ in range(T):
    startx, starty, endx, endy = map(int, input().split())
    solve(startx, starty, endx, endy)
```


代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChate0b1c1384585c6547a7fe835bc8cc1b4-1.jpg)
![alt text](WeChat178334f3712f79f435a8b415d65373ec-1.jpg)


### 04129: 变换的迷宫

bfs, http://cs101.openjudge.cn/practice/04129/

思路：
本题不能直接用二维flag(或in queue)，因为可能会反复横跳，重复经过同一个点（为了等到K的倍数时经过旁边的#）；但是本题可以再额外记录时间mod K的信息，这样就可以用三维flag(或in queue)记录（因为t和t mod K时间的#状态相同,因此多次经过这样的同一个状态是可以剪枝的）

学习了如何找二维数组中的index

代码：

```python
from collections import deque
def solve():
    flag = [[[False] * m for _ in range(n)] for ___ in range(p)] # 是否曾in q过
    q = deque([(0, sx, sy)])
    flag[0][sx][sy] = True
    while q:
        d, x, y = q.popleft()
        if (x, y) == (ex, ey): return d
        for nx, ny in directions:
            tx, ty = x + nx, y + ny
            if 0 <= tx < n and 0 <= ty < m and not flag[(d + 1) % p][tx][ty] and (a[tx][ty] != "#" or (d + 1) % p == 0):
                flag[(d + 1) % p][tx][ty] = True
                q.append((d + 1, tx, ty))
    return "Oop!"

directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
T = int(input())
for __ in range(T):
    n, m, p = map(int, input().split())
    a = [input() for _ in range(n)]
    for rowx, row in enumerate(a):
        try:
            coly = row.index('S')
            sx, sy = rowx, coly
            break
        except ValueError: continue
    for rowx, row in enumerate(a):
        try:
            coly = row.index('E')
            ex, ey = rowx, coly
            break
        except ValueError: continue
    print(solve())
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatf409fa84da715d11783099f25b7e04ab-1.jpg)




## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

本次作业做了较长时间，而且有题目看了题解；本周学习了最短路算法dijkstra，巩固了bfs，复习了heapq，同时也看了一些笔试内容等。本周有考试因此补的选做题不多。





