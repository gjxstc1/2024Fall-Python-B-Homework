# Assign #3: Oct Mock Exam暨选做题目满百

Updated 1537 GMT+8 Oct 10, 2024

2024 fall, Complied by Hongfei Yan==高景行 数学科学学院==



**说明：**

1）Oct⽉考： AC==5（考场上）== 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++/C（已经在Codeforces/Openjudge上AC），截图（包含Accepted, 学号），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、作业评论有md或者doc。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E28674:《黑神话：悟空》之加密

http://cs101.openjudge.cn/practice/28674/



思路：
直接做，关键在于ord和chr函数


代码

```python
k = int(input())
s = input()
t = len(s)
k %= 26
for i in range(t):
    x = ord(s[i])
    if x >= 97:
        x -= k
        if x <= 96:
            x = x + 122 - 96
    else:
        x -= k
        if x <= 64:
            x = x + 90 - 64
    print(chr(x), end = "")

"""65-90; 97-122"""

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](WeChatbfa2ae655c0b8f992c9a9cb38373606e-1.jpg)



### E28691: 字符串中的整数求和

http://cs101.openjudge.cn/practice/28691/



思路：
直接做


代码

```python
s1, s2 = input().split()
sum1 = int(s1[0]) * 10 + int(s1[1])
sum2 = int(s2[0]) * 10 + int(s2[1])
print(sum1 + sum2)

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](WeChat623b61754b42fea822f2e140b2daf52c-1.jpg)



### M28664: 验证身份证号

http://cs101.openjudge.cn/practice/28664/



思路：
直接判断，把权记在列表里


代码

```python
a = [7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]
T = int(input())
for __ in range(T):
    s = input()
    cnt = 0
    for i in range(17):
        cnt += a[i] * int(s[i])
        cnt %= 11
    x = s[17]
    if not cnt or cnt == 1:
        if cnt + int(x) != 1: print("NO")
        else: print("YES")
        continue
    if cnt == 2:
        if x != "X": print("NO")
        else: print("YES")
        continue
    if cnt + int(x) != 12: print("NO")
    else: print("YES")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](WeChatc8e0808b6a584b2214520e620476f8c8-1.jpg)




### M28678: 角谷猜想

http://cs101.openjudge.cn/practice/28678/



思路：
直接做，关键在用format输出


代码

```python
n = int(input())
while n != 1:
    if n & 1:
        print("{}*3+1={}".format(n, n * 3 + 1))
        n = n * 3 + 1
    print("{}/2={}".format(n, n >> 1))
    n >>= 1
print("End")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](WeChat5e70c3c1f40147ade75e5f0b86d7b972-1.jpg)



### M28700: 罗马数字与整数的转换

http://cs101.openjudge.cn/practice/28700/



思路：
有些细节的模拟，但整体还好。未来应该学习一下字典。


##### 代码

```python
# 
s = input()
a = ["I", "V", "X", "L", "C", "D", "M"]
b = [1, 5, 10, 50, 100, 500, 1000]
w = [0 for i in range(100)]
w[ord("I")] = 0
w[ord("V")] = 1
w[ord("X")] = 2
w[ord("L")] = 3
w[ord("C")] = 4
w[ord("D")] = 5
w[ord("M")] = 6
t = len(s)
if ord(s[0]) < 65:
    s = int(s)
    ans = ""
    for i in range(6, -1, -1):
        x = s // b[i]
        y = s % b[i]
        if i == 1 or i == 3 or i == 5:
            if y >= b[i] - b[i - 1]:
                ans += a[i] * x + a[i - 1] + a[i]
                y -= b[i] - b[i - 1]
                s = y
                continue
        if i == 2 or i == 4 or i == 6:
            if y >= b[i] - b[i - 2]:
                ans += a[i] * x + a[i - 2] + a[i]
                y -= b[i] - b[i - 2]
                s = y
                continue
        ans += a[i] * x
        s = y
    print(ans)
else:
    ans = 0
    cut = [False for i in range(t + 1)]
    for i in range(t - 1):
        if w[ord(s[i])] < w[ord(s[i + 1])]: cut[i] = True
    for i in range(t):
        if i != 0 and cut[i - 1]: continue
        if not cut[i]:
            ans += b[w[ord(s[i])]]
            #
            continue
        ans += b[w[ord(s[i + 1])]] - b[w[ord(s[i])]]
        #i += 1
    print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](WeChatad9eae9abd75c680637271d1bf2ed424-1.jpg)



### *T25353: 排队 （选做）

http://cs101.openjudge.cn/practice/25353/



思路：

稳定$O(nlog^2n)$做法，和大佬学长做法不同。

本题可加入全单调递增（相邻两项差值大于d）和单调递减（相邻两项差值大于d）的数据，以卡住大部分最坏情况$O(n^2)$的算法。

（参考网上做法，非自己想到😭）答案一定由若干个递增段构成，我们只需要找这些段即可。显然只用考虑每个数往前跑。$\forall a[i],$ 找 $l,$使得$\left|a[l-1]-a[i]\right|>d,\left|a[k]-a[i]\right|<=d, \forall l \leq k \leq i-1$ 则 $a[i]$可活动范围为 $l$~$i,$并且$a[i]$移动到$p\in[l,i)$，其中$a[p] > a[i]$,保证字典序。

具体流程为维护这些段。每新来一个数看它最往前能到第几段，并且从这段往后的所有的段当中把这个数放到第一个使得段内$max$大于这个数的段，以保证字典序更小。用线段树维护区间最大最小值，那么$l,p$满足可二分性,二分+验证为$O(nlogn*logn)$，每一段内用$multiset$来在插入时排序，也为$O(nlogn)$（也可最后排序）

注：本题也可以按照身高差>d的连前到后的有向边，然后所求为最小拓扑排序，这个最坏是$O(n^2)$，所以还需要用一些数据结构优化，感觉不如直接这样写；本题也可以用 https://www.cnblogs.com/guoshaoyang/p/17824372.html 上大佬学长给的做法，更加自然

不知道为啥在加强版中C++就AC，python就TLE了。。。。。。

代码
python:

```python
#import bisect
#bisect.insort(a, x)时间复杂度是O(n)，坑人！
class Node:
    def __init__(self, mn, mx):
        self.mn = mn
        self.mx = mx
NR = int(1e5 + 10)
ls = lambda x: x << 1
rs = lambda x: x << 1 | 1
tree = [Node(float("inf"), float("-inf")) for i in range(NR * 4 + 1)]

def pushup(rt):
    tree[rt].mn = min(tree[ls(rt)].mn, tree[rs(rt)].mn)
    tree[rt].mx = max(tree[ls(rt)].mx, tree[rs(rt)].mx)

def upd(rt, l, r, x, MN, MX):
    if l == r:
        tree[rt].mn = MN
        tree[rt].mx = MX
        return
    mid = (l + r) >> 1
    if x <= mid: upd(ls(rt), l, mid, x, MN, MX)
    else: upd(rs(rt), mid + 1, r, x, MN, MX)
    pushup(rt)

def qmin(rt, l, r, qL, qR):
    if qL <= l and r <= qR: return tree[rt].mn
    mid = (l + r) >> 1
    ans = float('inf')
    if qL <= mid: ans = min(ans, qmin(ls(rt), l, mid, qL, qR))
    if mid < qR: ans = min(ans, qmin(rs(rt), mid + 1, r, qL, qR))
    return ans

def qmax(rt, l, r, qL, qR):
    if qL <= l and r <= qR: return tree[rt].mx
    mid = (l + r) >> 1
    ans = float('-inf')
    if qL <= mid: ans = max(ans, qmax(ls(rt), l, mid, qL, qR))
    if mid < qR: ans = max(ans, qmax(rs(rt), mid + 1, r, qL, qR))
    return ans

n, d = map(int, input().split())
#a = list(map(int, input().split()))
a = []
for i in range(n): a.append(int(input()))
b = [[] for i in range(n + 2)]
bmn = [float('inf') for i in range(n + 2)]
bmx = [float('-inf') for i in range(n + 2)]
upd(1, 1, n, 1, a[0], a[0])
now = 1
b[1].append(a[0])
bmn[1] = bmx[1] = a[0]
for i in range(1, n):
    l = 1; r = now; ans1 = now + 1
    while l <= r:
        mid = (l + r) >> 1
        mn = qmin(1, 1, n, mid, now)
        mx = qmax(1, 1, n, mid, now)
        if a[i] - mn > d or mx - a[i] > d: l = mid + 1
        else: ans1 = min(ans1, mid);r = mid - 1
    if ans1 == now + 1:
        now += 1
        b[now].append(a[i])
        bmn[now] = bmx[now] = a[i]
        upd(1, 1, n, now, a[i], a[i])
        continue
    l = ans1; r = now; ans2 = now + 1
    while l <= r:
        mid = (l + r) >> 1
        mx = qmax(1, 1, n, ans1, mid)
        if a[i] >= mx: l = mid + 1
        else: ans2 = min(ans2, mid);r = mid - 1
    if ans2 == now + 1:
        now += 1
        b[now].append(a[i])
        bmn[now] = bmx[now] = a[i]
        upd(1, 1, n, now, a[i], a[i])
        continue
    if a[i] < bmn[ans2]: upd(1, 1, n, ans2, a[i], bmx[ans2])
    b[ans2].append(a[i])
    bmn[ans2] = min(bmn[ans2], a[i])
    bmx[ans2] = max(bmx[ans2], a[i])
for i in range(1, now + 1):
    for j in sorted(b[i]): print(j)
```

C++:

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
# define ls(x) x << 1
# define rs(x) x << 1 | 1
using namespace std;

const int NR = 1e5 + 10;

struct Node {
    int mn, mx;
} tree[NR * 4 + 1];

void pushup(int rt) {
    tree[rt].mn = min(tree[ls(rt)].mn, tree[rs(rt)].mn);
    tree[rt].mx = max(tree[ls(rt)].mx, tree[rs(rt)].mx);
}

void upd(int rt, int l, int r, int x, int MN, int MX) {
    if (l == r) {
        tree[rt].mn = MN, tree[rt].mx = MX;
        return ;
    }
    int mid = (l + r) >> 1;
    if (x <= mid) upd(ls(rt), l, mid, x, MN, MX);
    else upd(rs(rt), mid + 1, r, x, MN, MX);
    pushup(rt);
}

int qmin(int rt, int l, int r, int qL, int qR) {
    if (qL <= l && r <= qR) return tree[rt].mn;
    int mid = (l + r) >> 1, ans = 2e9;
    if (qL <= mid) ans = min(ans, qmin(ls(rt), l, mid, qL,qR));
    if (mid < qR) ans = min(ans, qmin(rs(rt), mid + 1, r, qL, qR));
    return ans;
}

int qmax(int rt, int l, int r,int qL, int qR) {
    if (qL <= l && r <= qR) return tree[rt].mx;
    int mid = (l + r) >> 1, ans = -2e9;
    if (qL <= mid) ans = max(ans, qmax(ls(rt), l, mid, qL,qR));
    if (mid < qR) ans = max(ans, qmax(rs(rt), mid + 1, r, qL, qR));
    return ans;
}

int n, d;
int a[NR + 1];
int now = 0;
multiset <int> b[NR + 1];
int duan[NR + 1];

int main() {
    scanf("%d%d", &n, &d);
    f(i,1,n) scanf("%d", a + i);
    upd(1, 1, n, 1, a[1], a[1]);
    now = 1;
    b[1].insert(a[1]);
    duan[1] = 1;
    f(i,2,n) {
        int l = 1, r = now, mid, ans1 = now + 1, ans2 = now + 1, mn, mx;
        while (l <= r) {
            mid = (l + r) >> 1;
            mn = qmin(1, 1, n, mid, now);
            mx = qmax(1, 1, n, mid, now);
            if (a[i] - mn > d || mx - a[i] > d) l = mid + 1;
            else ans1 = min(ans1, mid), r = mid - 1;
        }
        if (ans1 == now + 1) {
            b[++now].insert(a[i]);
            duan[i] = now;
            upd(1, 1, n, now, a[i], a[i]);
            continue;
        }
        l = ans1, r = now;
        while (l <= r) {
            mid = (l + r) >> 1;
            mx = qmax(1, 1, n, ans1, mid);
            if (a[i] >= mx) l = mid + 1;
            else ans2 = min(ans2, mid), r = mid - 1;
        }
        if (ans2 == now + 1) {
            b[++now].insert(a[i]);
            duan[i] = now;
            upd(1, 1, n, now, a[i], a[i]);
            continue;
        }
        if (a[i] < *b[ans2].begin()) upd(1, 1, n, ans2, a[i], *b[ans2].rbegin());
        b[ans2].insert(a[i]);
    }
    f(i,1,now) {
        for(multiset<int>::iterator it = b[i].begin(); it != b[i].end(); it++) {
            printf("%d\n", *it);
        }
    }
    puts("");
    return 0;
}

```


代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

3个截图：

python普通版本题目：

![alt text](WeChat336649be956705fc21de7dfaa00d2229-1.jpg)

C++普通版本题目：
![alt text](WeChat22270b21debab1e64b786207c2719b23-1.jpg)

C++加强版本题目：
![alt text](WeChat1a48ffc429a503af48e9a25003da0d42-1.jpg)

## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。==

通过月考发现自己对很多python中常见函数和操作还是不了解（比如list有些常用的操作、字典（之前完全没用过）、aA1的ASCII码），极大影响做题速度，这是因为在之前做题时靠着笨方法绕开了很多这样的函数和操作。以后要注意做题时多学习一些python中方便好用的函数和操作。

题目有难度，没有独立想到排队能稳定不超时的算法，只是在看过题解后才会线段树做法。如果期末考试有类似这样的题目可能还需要学习一些 数据结构和常用的算法适用的模型 来优化时间复杂度。题目太难了😭。

选做题中约瑟夫问题有逆推的巧妙的思想，有一定难度。

OJ“计概2024fall每日选做”目前全部AC。
