# Assignment #4: T-primes + 贪心

Updated 0337 GMT+8 Oct 15, 2024

2024 fall, Complied by <mark>高景行 数学科学学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 34B. Sale

greedy, sorting, 900, https://codeforces.com/problemset/problem/34/B



思路：
直接排序。


代码

```python
# 
n, m = map(int, input().split())
a = list(map(int, input().split()))
a = sorted(a)
sum = 0
for i in range(m):
    if a[i] >= 0: break
    sum += a[i]
print(-sum)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatbc97574bf3451bc4473ba8d2fef0e8ff-1.jpg)




### 160A. Twins

greedy, sortings, 900, https://codeforces.com/problemset/problem/160/A

思路：
从大往小取，直到超过一半，硬币数最少。


代码

```python
n = int(input())
a = list(map(int, input().split()))
tmp = 0
for i in range(n): tmp += a[i]
m = tmp // 2
a = sorted(a, reverse = True)
cnt = 0
for i in range(n):
    cnt += a[i]
    if cnt > m:
        print(i + 1)
        break
```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChat069bfeda33780ebe886d1ddb6853e862-1.jpg)




### 1879B. Chips on the Board

constructive algorithms, greedy, 900, https://codeforces.com/problemset/problem/1879/B

思路：
显然要么每行各取1个，要么每列各取一个。这两种情况都可以对应贪心取最小的整列和最小的整行，再取min。


代码

```python
T = int(input())
for ___ in range(T):
    n = int(input())
    a = list(map(int, input().split())); b = list(map(int, input().split()))
    mn1 = float("inf"); mn2 = float("inf")
    s1 = 0; s2 = 0
    for i in range(n):
        mn1 = min(mn1, a[i]); mn2 = min(mn2, b[i])
        s1 += a[i]; s2 += b[i]
    print(min(s1 + mn2 * n, s2 + mn1 * n))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChat65b2d7c399072cb3adcc5040444507bf-1.jpg)




### 158B. Taxi

*special problem, greedy, implementation, 1100, https://codeforces.com/problemset/problem/158/B

思路：
显然3和1凑，2和2、1凑。类似6*6方格的那个选做题。


代码

```python
import math
n = int(input())
a = list(map(int, input().split()))
b = [0, 0, 0, 0, 0]
for i in range(n): b[a[i]] += 1
ans = b[4]
if b[1] <= b[3]: ans += b[3] + math.ceil(b[2] / 2)
else:
    b[1] -= b[3]
    if b[2] & 1:
        if b[1] >= 2: ans += b[3] + (b[2] + 1) // 2 + math.ceil((b[1] - 2) / 4)
        else: ans += b[3] + (b[2] + 1) // 2
    else: ans += b[3] + b[2] // 2 + math.ceil(b[1] / 4)
print(ans)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatd3f8abf61d01fec97f4bea7467c01226-1.jpg)




### *230B. T-primes（选做）

binary search, implementation, math, number theory, 1300, http://codeforces.com/problemset/problem/230/B

思路：
预处理:欧拉筛$O(n)$,可以顺便求出$\phi,$ $\mu$

代码

```python
import math
NR = 1000000
isprime = [True] * (NR + 1)
prime = []
phi = [1 for _ in range(NR + 1)]
cnt = 0; tmp = 0
isprime[0] = isprime[1] = False
for j in range(2, NR + 1):
    if isprime[j]:
        cnt += 1
        phi[j] = j - 1
        prime.append(j)
    for i in prime:
        tmp = i * j
        if tmp > NR: break
        isprime[tmp] = False
        if j % i == 0:
            phi[tmp] = i * phi[j]
            break
        phi[tmp] = phi[j] * phi[i]

"""ans = {}
for j in range(1, cnt + 1): ans[prime[j] * prime[j]] = True"""
n = int(input())
a = list(map(int, input().split()))
"""for i in range(n):
    if a[i] in ans: print("YES")
    else: print("NO")"""
for i in range(n):
    tmp = int(a[i] ** 0.5)
    if not isprime[tmp] or tmp ** 2 != a[i]:print("NO")
    else: print("YES")
```


```cpp
# include <cstdio>
# include <iostream>
# include <cmath>
# include <algorithm>
# include <cstring>
# include <queue>
# include <stack>
# include <map>
# include <set>

# define f(i,a,b) for (int i = a; i <= b; i++)
# define _f(i,a,b) for (int i = a; i >= b; i--)
using namespace std;
const int NR = 1e6;
map <long long, bool> check;
int n;
long long a[NR + 1];
bool isprime[NR + 1];
int prime[NR + 1], cnt = 0;
int phi[NR + 1];

int main() {
    scanf("%d", &n);
    phi[1] = 1;
    f(i,1,NR) isprime[i] = true;
    isprime[0] = isprime[1] = false;
    f(j,2,NR) {
        if (isprime[j]) {
            prime[++cnt] = j;
            phi[j] = j - 1;
        }
        for (int i = 1; i <= cnt && j * prime[i] <= NR; i++) {
            isprime[j * prime[i]] = false;
            if (j % prime[i] == 0) {
                phi[j * prime[i]] = phi[j] * prime[i];
                break;
            }
            phi[j * prime[i]] = phi[j] * phi[prime[i]];
        }
    }
    f(i,1,cnt) check[1ll * prime[i] * prime[i]] = true;
    f(i,1,n) {
        cin >> a[i];
        if (check.find(a[i]) != check.end()) printf("YES\n");
        else printf("NO\n");
    }
    return 0;
}
```


代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](WeChatf7254427b157d31e0ca47658ecb295d3-1.jpg)

![alt text](WeChat59633e9ae899d8a3fe288b4ed4ac5054-1.jpg)


### *12559: 最大最小整数 （选做）

greedy, strings, sortings, http://cs101.openjudge.cn/practice/12559

思路：
直接按照字典序排序显然不对(如：99，991，9912，992)，之后就猜测根据字典序x+y>y+x，
那么x排在y前面来排序，因为在前面所有位都相等时就是这么排的（前面有位不同时直接字典序已经可以区分，这个式子也是成立的）。
不过我也不太能证明这满足偏序关系($x<=y,y<=z$推出$x<=z$),
也不太能严格证明这个的正确性，但是确实对了。

之后我看到题解中写了可以将每个数复制一遍放后面，得到的新字符串可以按照字典序排序，巧妙。


代码

```python
from functools import cmp_to_key
def cmp(x, y):
    if x + y > y + x: return 1
    return -1
n = int(input())
a = list(input().split())
a = sorted(a, key = cmp_to_key(cmp), reverse = True)
b1 = "".join(a); b2 = "".join(reversed(a))
print("{} {}".format(b1, b2))
```


代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](WeChat3d3266e0cdbae949acdf6b6fbbf04712-1.jpg)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

本周我学习了贪心。学习了按结构体中特定元素排序、按指定函数排序的语法；学习了字典的语法。

这周每日选做等期中考试结束后再赶上来。

