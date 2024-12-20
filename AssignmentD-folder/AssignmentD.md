# Assignment #D: 十全十美 

Updated 1254 GMT+8 Dec 17, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 02692: 假币问题

brute force, http://cs101.openjudge.cn/practice/02692

思路：
暴力枚举


代码：

```python
T = int(input())
#dict = {"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "H":8, "I":9, "J":10, "K":11, "L":12}
for __ in range(T):
    s=[]
    for i in range(3): s.append(input().split())
    for alpha in "ABCDEFGHIJKL":
        for id in ["light", "heavy"]:
            flag = True
            for i in range(3):
                if alpha not in s[i][0] and alpha not in s[i][1] and s[i][2] != "even":
                    flag = False
                    break
                if id == "light":
                    if alpha in s[i][0] and s[i][2] != "down" or alpha in s[i][1] and s[i][2] != "up":
                        flag = False
                        break
                if id == "heavy":
                    if alpha in s[i][0] and s[i][2] != "up" or alpha in s[i][1] and s[i][2] != "down":
                        flag = False
                        break
            if flag:
                print(f"{alpha} is the counterfeit coin and it is {id}. ")
                break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat391050e562dc0e683098b9f6ac24dbc9-1.jpg)




### 01088: 滑雪

dp, dfs similar, http://cs101.openjudge.cn/practice/01088

思路：
dfs即可，状态相同的位置的搜索结果永远相同，故可以用数组记录（进而只用搜索一次）（其实本质就是@lru_cache(maxsize=None)，因此也可以只用这一行语句解决）


代码：

```python
import sys
sys.setrecursionlimit(1 << 30)
from functools import lru_cache
@lru_cache(maxsize=None)
def dfs(x, y):
    if d[x][y] > 0: return d[x][y] 
    ans = 1
    for nx, ny in directions:
        tx, ty = x + nx, y + ny
        if 0 <= tx < n and 0 <= ty < m and a[tx][ty] < a[x][y]:
            ans = max(ans, dfs(tx, ty) + 1)
    d[x][y] = ans
    return ans
n, m = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(n)]
d = [[0] * m for _ in range(n)]
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ans = 1
for i in range(n):
    for j in range(m):
        ans = max(ans, dfs(i, j))
print(ans)
```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChat82a120098b6b3a3d8d438fcc1b986aa1-1.jpg)




### 25572: 螃蟹采蘑菇

bfs, dfs, http://cs101.openjudge.cn/practice/25572/

思路：
1*2的人只记录其中一个位置即可，另一个位置只在判断能否移动时才发挥作用


代码：

```python
from collections import deque

def chk():
    for i in range(n):
        for j in range(n):
            if a[i][j] == 5:
                for nx, ny in directions:
                    tx, ty = i + nx, j + ny
                    if 0 <= tx < n and 0 <= ty < n and a[tx][ty] == 5:
                        return i, j, nx, ny

n = int(input())
a = [list(map(int, input().split())) for _ in range(n)]
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
sx, sy, adjustx, adjusty = chk()
flag = [[False] * n for _ in range(n)]
flag[sx][sy] = True
q = deque([(sx, sy)])
while q:
    x, y = q.popleft()
    if a[x][y] == 9 or a[x + adjustx][y + adjusty] == 9:
        print("yes")
        exit(0)
    for nx, ny in directions:
        tx, ty = x + nx, y + ny
        tx2, ty2 = tx + adjustx, ty + adjusty
        if 0 <= tx < n and 0 <= ty < n and 0 <= tx2 < n and 0 <= ty2 < n and\
                not flag[tx][ty] and a[tx][ty] != 1 and a[tx2][ty2] != 1:
            flag[tx][ty] = True
            q.append((tx, ty))
print("no")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatfeddd451bdaa897fc98f2a594de36490-1.jpg)




### 27373: 最大整数

dp, http://cs101.openjudge.cn/practice/27373/

思路：
dp[j]表示能凑到j位数的字符串中这个多位数最大的那个构造（不存在则为某个自定初值）。注意如果j位是可以构造的，那么构造所用的顺序一定是按照A+B>B+A排序好的顺序才使得数最大（之前作业题），故做dp前要先这样排序再做。最后位数从大到小即可。


代码：

```python
from functools import cmp_to_key
def cmp(x, y): # -1:不交换，x在y前；1:交换，y在x前
    if x + y > y + x: return -1
    return 1
m = int(input())
n = int(input())
s = input().split()
s = sorted(s, key = cmp_to_key(cmp))
ans = [""] + ["!"] * m
for i in range(n):
    for j in range(m, len(s[i]) - 1, -1):
        if ans[j - len(s[i])] != "!":
            if ans[j] == "!" or ans[j - len(s[i])] + s[i] > ans[j]: ans[j] = ans[j - len(s[i])] + s[i]
for j in range(m, -1, -1):
    if ans[j] != "!":
        print(ans[j])
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat3649935cc16340870eaf91b12d6c3c6a-1.jpg)




### 02811: 熄灯问题

brute force, http://cs101.openjudge.cn/practice/02811

思路：
本题较难，看了提示和题解才知道做法：枚举第一行的所有操作，之后根据第一行的操作后灯亮情况可以唯一决定第二行的操作（！），之后根据第二行的灯亮情况唯一决定第三行的操作，以此类推。因此只用枚举64种情况。看题解后也学习了product用法。


代码：

```python
from itertools import product
a = [list(map(int, input().split())) for _ in range(5)]
movement = [[0] * 6 for _ in range(5)]
for tmp in product(range(2), repeat = 6):
    movement[0] = list(tmp)
    b = [a[i][:] for i in range(5)]
    for j in range(6):
        if movement[0][j]:
            b[0][j] = not b[0][j]
            b[1][j] = not b[1][j]
            if j: b[0][j - 1] = not b[0][j - 1]
            if j != 5: b[0][j + 1] = not b[0][j + 1]
    for j in range(1, 5):
        for k in range(6):
            movement[j][k] = int(b[j - 1][k])
            if b[j - 1][k]:
                b[j - 1][k] = False
                b[j][k] = not b[j][k]
                if j != 4: b[j + 1][k] = not b[j + 1][k]
                if k: b[j][k - 1] = not b[j][k - 1]
                if k != 5: b[j][k + 1] = not b[j][k + 1]
    flag = True
    for k in range(6):
        if b[4][k]:
            flag = False
            break
    if flag:
        for row in movement: print(*row)
        exit(0)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat0ccb217f91e5a818a610b1e98599bda1-1.jpg)




### 08210: 河中跳房子

binary search, greedy, http://cs101.openjudge.cn/practice/08210/

思路：
二分答案，$O(logL)$，有了答案之后的检查可以贪心，因为起点必须保留（！），因此可以从前往后走去掉不合法石头（直到碰到第一个不用去掉的石头即可，因为石头越靠前越优）。


代码：

```python
def check(x):
    num = 0; pos = 0
    while pos < n:
        posr = pos + 1
        while posr < n and a[posr] - a[pos] < x:
            num += 1
            posr += 1
        pos = posr
    #print(x, num)
    return num <= m
w, n, m = map(int, input().split())
a = [0] + [int(input()) for i in range(n)] + [w]
n += 2
l, r, ans = 1, w, -1
while l <= r:
    mid = (l + r) >> 1
    if check(mid): ans, l = mid, mid + 1
    else: r = mid - 1
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat9efca83be681477742cde956c8f6d349-1.jpg)




## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

本周复习了programming综合（贪心，dp，dfs，bfs），发现自己还有很多常见操作还不太会（比如product），之前的一些语法和算法也有所遗忘(比如有的贪心原来本会的，现在却忘了怎么做的)，有点焦虑期末机考。本周补了不少之前欠下的选做题，熟练了很多。



