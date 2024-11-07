# Assignment #7: Nov Mock Exam立冬

Updated 1646 GMT+8 Nov 7, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）⽉考： AC<mark>6</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E07618: 病人排队

sorttings, http://cs101.openjudge.cn/practice/07618/

思路：
按照x排序即可。


代码：

```python
from functools import cmp_to_key
n = int(input())
class Node:
    def __init__(self, c, age, pos):
        self.c = c
        self.age = age
        self.pos = pos
def cmp(x, y): # return 1 <-> 应该交换(x,y),即y应该排在x的前面
    if x.age != y.age: return y.age - x.age
    return x.pos - y.pos
a = []
b = []
cnt1 = 0; cnt2 = 0
for i in range(n):
    x, y = map(str, input().split())
    y = int(y)
    if y >= 60:
        cnt1 += 1
        a.append(Node(x, y, i))
    else:
        cnt2 += 1
        b.append(x)
a = sorted(a, key = cmp_to_key(cmp))
for i in range(cnt1): print(a[i].c)
for i in range(cnt2): print(b[i])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat479443bd498ac62ce6ab8624649e3ea8-1.jpg)




### E23555: 节省存储的矩阵乘法

implementation, matrices, http://cs101.openjudge.cn/practice/23555/

思路：
直接做。


代码：

```python
n, m1, m2 = map(int, input().split())
a = [[0] * n for i in range(n)]
b = [[0] * n for i in range(n)]
for _ in range(m1):
    r, c, x = map(int, input().split())
    a[r][c] = x
for _ in range(m2):
    r, c, x = map(int, input().split())
    b[r][c] = x
for i in range(n):
    for j in range(n):
        ans = 0
        for k in range(n): ans += a[i][k] * b[k][j]
        if ans: print(" ".join(map(str, [i, j, ans])))
```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChat6e99341815af903a023f586d01669b6c-1.jpg)




### M18182: 打怪兽 

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/

思路：
仍然是多分量的排序，排序后直接做。


代码：

```python
from functools import cmp_to_key
T = int(input())
class Node:
    def __init__(self, t, x):
        self.t = t
        self.x = x
def cmp(p, q):
    if p.t == q.t:
        return q.x - p.x
    return p.t - q.t
for __ in range(T):
    n, m, b = map(int, input().split())
    a = []
    for i in range(n):
        x, y = map(int, input().split())
        a.append(Node(x, y))
    a = sorted(a, key = cmp_to_key(cmp))
    b -= a[0].x
    if b <= 0:
        print(a[0].t)
        continue
    now = 1; flag = False
    for i in range(1, n):
        if a[i].t == a[i - 1].t: now += 1
        else: now = 1
        if now > m: continue
        b -= a[i].x
        if b <= 0:
            print(a[i].t)
            flag = True
            break
    if not flag: print("alive")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](WeChat0ee7eacfb6a03ce8c19ba311ea924c8d-1.jpg)



### M28780: 零钱兑换3

dp, http://cs101.openjudge.cn/practice/28780/

思路：
完全背包。考场上python超时了于是改写C++，不知道为啥场下再交又对了。


代码：

```python
n, m = map(int, input().split())
a = list(map(int, input().split()))
dp = [1000000] * (m + 1)
dp[0] = 0
for i in range(n):
    for j in range(a[i], m + 1):
        dp[j] = min(dp[j], dp[j - a[i]] + 1)
if dp[m] == 1000000: print(-1)
else: print(dp[m])
```
```cpp
# include <bits/stdc++.h>

# define f(i, a, b) for (int i = a; i <= b; i++)
# define _f(i, a, b) for (int i = a; i >= b; i--)
using namespace std;

const int NR = 1e6;
int n, m;
int x, dp[NR + 1];
int main() {
	scanf("%d%d", &n, &m);
        memset(dp, 127, sizeof(dp));
	dp[0] = 0;
	f(i,1,n) {
		scanf("%d", &x);
		f(j,x,m) dp[j] = min(dp[j], dp[j - x] + 1);
	}
	if (dp[m] < 1e7) printf("%d\n", dp[m]);
	else printf("-1\n");
	return 0;
}
```


代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat58afa415acf358fb8cb24b015def2881-1.jpg)
![alt text](WeChata3692d04c5a408748f03d7bc6e308002-1.jpg)




### T12757: 阿尔法星人翻译官

implementation, http://cs101.openjudge.cn/practice/12757

思路：
题意非常模糊，需要猜测题意，按照个人理解来做。是有一点细节的模拟，重点在于million后的thousand只乘上million后面的数。


代码：

```python
a = "negative, zero, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty, thirty, forty, fifty, sixty, seventy, eighty, ninety, hundred, thousand, million".split(", ")
mp = {}
mp[a[0]] = -1
for i in range(21): mp[a[i + 1]] = i
for i in range(22, 29): mp[a[i]] = 10 * (i - 19)
mp[a[29]] = 100; mp[a[30]] = 1000; mp[a[31]] = 1000000
c = list(map(str, input().split()))
op = 1; st = 0
if c[0] == a[0]: op = -1; st = 1
now = 0
c.append("F")
c.append("F")
cur = 0
ans = 0
while st <= len(c) - 3:
    if c[st + 1] == a[29]:
        now += mp[c[st]] * 100
        if c[st + 2] == a[30]: now *= 1000; st += 3
        elif c[st + 2] == a[31]:
            now *= 1000000; st += 3
            ans += now
            now = 0
        else: st += 2
    elif c[st + 1] == a[30]:
        now = (now + mp[c[st]]) * 1000; st += 2

    elif c[st + 1] == a[31]:
        now = (now + mp[c[st]]) * 1000000; st += 2
        ans += now
        now = 0
    else:
        now += mp[c[st]]; st += 1
ans += now
print(ans* op)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChateb37a307e82dfb7a4beef1b3b37d4540-1.jpg)




### T16528: 充实的寒假生活

greedy/dp, cs10117 Final Exam, http://cs101.openjudge.cn/practice/16528/

思路：
数据量过小，可以$O(nm)$dp。考场下才想到可以按照结束时间排序+贪心。


代码：

```python
n = int(input())
a = []
for i in range(n):
    x, y = map(int, input().split())
    a.append((x + 1, y + 1))
a = sorted(a, key = lambda _: _[0])
dp = [0] * 65
for i in range(n):
    for j in range(a[i][1], 62):
        dp[j] = max(dp[j], dp[a[i][0] - 1] + 1)
print(dp[61])
```
```python
n = int(input())
a = []
for i in range(n):
    x, y = map(int, input().split())
    a.append((x, y))
a = sorted(a, key = lambda _: _[1])
pos = 0; ans = 0
while pos < n:
    ans += 1
    tmp = pos
    while tmp < n and a[tmp][0] <= a[pos][1]: tmp += 1
    pos = tmp
print(ans)
```


代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat6381f47008b467bd6d7ad3562f318ffe-1.jpg)
![alt text](WeChatc56ac4966287b073725c82e2869733ab-1.jpg)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>
s
目前还在赶之前选做题的阶段，期中终于结束了。

月考的总结：通过月考我发现自己对python的语法还并不熟练（比如对字典中某元素是否存在、多个元素按照某些分量排序的语法、字典的语法，因而浪费很多时间）、对常见结构不熟悉（比如最后一道题显然可以贪心），之后应当多总结。



