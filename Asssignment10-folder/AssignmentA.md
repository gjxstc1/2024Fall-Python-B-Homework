# Assignment #10: dp & bfs

Updated 2 GMT+8 Nov 25, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### LuoguP1255 数楼梯

dp, bfs, https://www.luogu.com.cn/problem/P1255

思路：
直接做


代码：

```python
def main():
    n = int(input())
    f = [0] + [1] + [2] + [0] * (n - 2)
    for i in range(3, n + 1): f[i] = f[i - 1] + f[i - 2]
    print(f[n])
if __name__ == '__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat19177a33c8ead26e165b284f6acc494e-1.jpg)




### 27528: 跳台阶

dp, http://cs101.openjudge.cn/practice/27528/

思路：
直接做。每一层要么到过要么没到过，2种选择。


代码：

```python
def main():
    n = int(input())
    print(2 ** (n - 1))
if __name__ == '__main__':
    main()
```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChat9756e33ada726a335339160d8cae98e0-1.jpg)




### 474D. Flowers

dp, https://codeforces.com/problemset/problem/474/D

思路：
dp，枚举最后一个连续k个白色段所在位置，得到$f[i]=\sum_{j=0}^{i-k} f[j]$,然后前缀和维护


代码：

```python
P = int(1e9) + 7

def main():
    n = int(1e5)
    T, k = map(int, input().split())
    f = [1] * (n + 1)
    s = [i + 1 for i in range(n + 1)]
    for i in range(k, n + 1):
        f[i] = (1 + s[i - k]) % P
        s[i] = (s[i - 1] + f[i]) % P
    for ___ in range(T):
        x, y = map(int, input().split())
        print(((s[y] - s[x - 1]) % P + P) % P)

if __name__ == "__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat4abfe9a27af4c3b5debf5b02d5fb3d22-1.jpg)




### LeetCode5.最长回文子串

dp, two pointers, string, https://leetcode.cn/problems/longest-palindromic-substring/

思路：
直接做，枚举中间位置


代码：

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        mx = 1; ans = s[0]
        for i in range(0, n):
            l, r = i, i
            while l >= 0 and r <= n - 1 and s[l] == s[r]: l -= 1; r += 1
            if r - l - 1 > mx:
                mx = r - l - 1
                ans = s[l + 1:r]
            l, r = i, i + 1
            while l >= 0 and r <= n - 1 and s[l] == s[r]: l -= 1; r += 1
            if r - l - 1 > mx:
                mx = r - l - 1
                ans = s[l + 1:r]
        return ans

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat3a2b08b188ee6b0c6f3f182bbe25382f-1.jpg)






### 12029: 水淹七军

bfs, dfs, http://cs101.openjudge.cn/practice/12029/

思路：
理解真实题意后题目本身很容易，但感觉题目有点模糊。难点在于python直接输入有问题，以及水可以平级流动、司令部可以与周围水等高(这些要点没直接出现在题目叙述中，有点莫名其妙，我本以为必须要一直严格单调更低才能传递)，因为这个我提交了将近10次RE和一堆WA😭。希望期末上机没有这种题目。

可以对放水点离线排序，先做放水点高的位置，避免后面覆盖的麻烦情况（水越高越优）。最后要排除掉放水点高度<=司令部高度的（不会影响司令部）。


代码：

```python
import sys
sys.setrecursionlimit(1 << 30)
input = sys.stdin.read
def dfs(x, y, c):
    flag[x][y] = True
    for nx, ny in directions:
        tx, ty = x + nx, y + ny
        if 0 <= tx < m and 0 <= ty < n and not flag[tx][ty] and a[tx][ty] <= c:
            dfs(tx, ty, c)

if __name__ == "__main__":
    data = input().split()
    index = 0
    T = int(data[index]); index += 1
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    result = []
    for __ in range(T):
        m, n = map(int, data[index:index + 2]); index += 2
        a = []
        for _ in range(m):
            a.append(data[index:index + n]); index += n
        sx, sy = map(int, data[index:index + 2]); index += 2
        sx -= 1; sy -= 1
        S = int(data[index]); index += 1
        flag = [[False] * n for _ in range(m)]
        query = []
        for i in range(S):
            x, y = map(int, data[index:index + 2]); index += 2
            x -= 1; y -= 1
            query.append((x, y, a[x][y]))
        query = sorted(query, key = lambda x: x[2], reverse = True)
        for x, y, z in query:
            if z <= a[sx][sy]: break
            dfs(x, y, z)
        print("Yes" if flag[sx][sy] else "No")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat42bbd337f5ac99376a742aa1d22cd5dc-1.jpg)




### 02802: 小游戏

bfs, http://cs101.openjudge.cn/practice/02802/

思路：
bfs，注意bfs要满足序关系（距离短的先进）。本题难度在于繁琐的输入输出（比如交换坐标、各种多个数据、还要奇怪地输出一行）。可以把str转化成list，使得其变成可变对象（便于只改变起点和终点的X，不用特殊判断起点终点）。


代码：

```python
from collections import deque
def main():
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    cnt = 0
    while True:
        cnt += 1
        m, n = map(int, input().split())
        if not m and not n: break
        a = [[" "] * (m + 2)] + [[" "] + list(input()) + [" "] for _ in range(n)] + [[" "] * (m + 2)]
        cnt0 = 0
        print(f"Board #{cnt}:")
        while True:
            cnt0 += 1
            starty, startx, endy, endx = map(int, input().split())
            if not startx and not starty: break
            q = deque()
            flag = [[False] * (m + 2) for i in range(n + 2)]
            q.append((startx, starty, 0))
            flag[startx][starty] = True
            a[startx][starty] = a[endx][endy] = " "
            ans = -1
            while q:
                tmp = q.popleft()
                if (endx, endy) == (tmp[0], tmp[1]):
                    ans = tmp[2]
                    break
                for nx, ny in directions:
                    tx, ty = tmp[0] + nx, tmp[1] + ny
                    while 0 <= tx <= n + 1 and 0 <= ty <= m + 1 and not flag[tx][ty] and a[tx][ty] != 'X':
                        q.append((tx, ty, tmp[2] + 1))
                        flag[tx][ty] = True
                        tx += nx; ty += ny
            print(f"Pair {cnt0}: {ans} segments." if ans != -1 else f"Pair {cnt0}: impossible.")
            a[startx][starty] = a[endx][endy] = "X"
        print()

if __name__ == "__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat1b159f9881df589ffa780911f3b9898e-1.jpg)




## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

本周学习了bfs，复习了dp，dfs。学习了更良好的代码习惯。

想请问一下如果某个题目必须要一次性读入，那我有没有方便的办法在本地测试样例（不用文件读入）? ai没有给出适用于mac的办法（貌似ctrl+D只是停止运行，并不能给EOF信号）

由于本周较忙所以没有写完本周的每日选做，争取在本周补完。现在特别慌张期末上机考试（比如莫名其妙的RE）和笔试（实在找不到往年python笔试题...）。



