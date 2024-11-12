## 1 保留几位小数（无import）
```python
ans = 0
for i in range(n):
    if w >= a[i].weight:
        w -= a[i].weight
        ans += a[i].value
    else:
        ans += a[i].value * w / a[i].weight
        break
print(f"{ans:.1f}")
```

-------

## 2 持续输入
```python
while True:
  try:
    输入
  except EOFError:
    break
```

------

## 3 取根号（有import math,也可以1/2次方）
```python
import math
T = int(input())
for _ in range(T): print(int(math.sqrt(int(input()))))
```

------

## 4 字符取ASCII,及常见ascii值
```python
k = int(input())
s = input()
t = len(s)
k %= 26
for i in range(t):
    x = ord(s[i]) #字符转整数
    if x >= 97:
        x -= k
        if x <= 96:
            x = x + 122 - 96
    else:
        x -= k
        if x <= 64:
            x = x + 90 - 64
    print(chr(x), end = "") #整数转字符
""" "1"-49, A 65-90, 97-122"""
```

------

## 5 结构体+排序（简单大小）
```python
NR = 10000
n, w = map(int, input().split())
class Node:
    def __init__(self, value, weight, ratio):
        self.ratio = ratio
        self.value = value
        self.weight = weight
a = []
for i in range(n):
    x, y = map(int, input().split())
    a.append(Node(x, y, x / y))
a = sorted(a, key=lambda _: _.ratio, reverse = True)
```

------

## 6 线段树
```python
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

```cpp
# include <bits/stdc++.h>
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

------

## 7 结构体排序（按照自己定义的函数，需要import）（复杂大小，推荐）
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
    return x.pos - y.poss
```
```python
from functools import cmp_to_key
def cmp(x, y): # return 1 <-> 应该交换(x,y),即y应该排在x的前面
    if x + y > y + x: return 1
    return -1 #注意这里不能写0.交换回去要写-1（也可以再补充==时是0）
n = int(input())
a = list(input().split())
a = sorted(a, key = cmp_to_key(cmp), reverse = True)
b1 = "".join(a); b2 = "".join(reversed(a)) # join函数可以将list中的string按顺序连接，中间用前面的""来连接
print("{} {}".format(b1, b2))
```
```python
from functools import cmp_to_key
class Person:
    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height

    def __repr__(self):
        return f"Person(name={self.name}, age={self.age}, height={self.height})"
# Custom comparison function
def compare_persons(p1, p2):
    if p1.age == p2.age:
        return p1.height - p2.height  # Sort by height if ages are equal
    return p1.age - p2.age  # Sort by age
# List of persons
people = [Person("Alice", 30, 165),Person("Bob", 25, 180),Person("Charlie", 30, 175),]
# Sorting using the custom comparison function
sorted_people = sorted(people, key=cmp_to_key(compare_persons))
```

------

## 8 欧拉筛

```cpp
# include <bits/stdc++.h>
# define f(i,a,b) for (int i = a; i <= b; i++)
# define _f(i,a,b) for (int i = a; i >= b; i--)
using namespace std;
const int NR = 1e6;
int n;
long long a[NR + 1];
bool isprime[NR + 1];
int prime[NR + 1], cnt = 0;
int phi[NR + 1], miu[NR + 1];

int main() {
    scanf("%d", &n);
    phi[1] = 1; miu[1] = 1;
    f(i,1,NR) isprime[i] = true;
    isprime[0] = isprime[1] = false;
    f(j,2,NR) {
        if (isprime[j]) {
            prime[++cnt] = j;
            miu[j] = -1;
            phi[j] = j - 1;
        }
        for (int i = 1; i <= cnt && j * prime[i] <= NR; i++) {
            isprime[j * prime[i]] = false;
            if (j % prime[i] == 0) {
                miu[j * prime[i]] = 0;
                phi[j * prime[i]] = phi[j] * prime[i];
                break;
            }
            phi[j * prime[i]] = phi[j] * phi[prime[i]];
            miu[j * prime[i]] = -miu[j];
        }
    }
    return 0;
}
```

------

## 9 最长上升子序列

```cpp
# include <bits/stdc++.h>
# define f(i,a,b) for (int i = a; i <= b; i++)
# define _f(i,a,b) for (int i = a; i >= b; i--)
using namespace std;
int g[100001];
int a[100001];

int main() {
    int n;
    scanf("%d", &n);
    int num = 0;
    g[0] = -1e9;
    f(i, 1, n) {
        scanf("%d", a + i);
        _f(j, num, 0)
            if (a[i] > g[j]) {
                g[j + 1] = a[i];
                num = max(num, j + 1);
                break;
            }
    }
    printf("%d", num); // num就是最长上升子序列长度
    return 0;
}
```

------

## 10 快速读入和输出（需import）

```python
import sys
input = sys.stdin.read # 快读
output = sys.stdout.write # 快写
def solve():
    data = input().split()
    n = int(data[0])
    results = []
    for i in range(1, n + 1):
    # 假设是简单的加法运算
        results.append(str(int(data[2*i - 1]) + int(data[2*i])))
    output("\n".join(results) + "\n") # 快写方式：将输出转成字符串
solve()
```

------

## 11 Exgcd

```cpp
int gcd(int a, int b, int &x, int &y) {
	if (b == 0) {
		x = 1;
		y = 0;
		return a;
	}
	int x1, y1, g = gcd(b, a % b, x1, y1);
	x = y1;
	y = x1 - a / b * y1;
	return g;
}
// exgcd中，ax+by=gcd(a,b),但x、y可能有负数
```

```python
def swap(a, b): return b, a #注意要写： x,y = swap(x, y)
def exgcd(a, b):
    if not b: return a, 1, 0
    d, x, y = exgcd(b, a % b)
    return d, y, x - a // b * y
n, a, b, c = map(int, input().split()) # exgcd(a, b)中，ax+by=d
d, x0, y0 = exgcd(a, b); ans = 0
```

------

## 12 rstrip
```python
a.rstrip()只删掉末位指定字符，没指明默认空格
```
------

## 13 整数转成字符串
```python
a = str(i) # i = 100, 则a = "100"
```

------

## 14 单调队列(deque)-有import

```cpp
# include <bits/stdc++.h>

# define f(i,a,b) for (int i = a; i <= b; i++)
# define _f(i,a,b) for (int i = a; i >= b; i--)
using namespace std;
const int NR = 1e6;
int n, k, x;

struct Node {
	int x, id;
} a[NR + 1];
deque <Node> q;

int main() {
	scanf("%d%d", &n, &k);
	f(i,1,n) {
		scanf("%d", &a[i].x);
		a[i].id = i;
	}
	// 求最小值， dq中储存的数单调上升，每存一个数要求比前一个数大，否则前面的数会被单调队列掉
	f(i,1,n) {
		while (!q.empty() && q.front().id <= i - k) q.pop_front();// 窗口长度
		while (!q.empty() && a[i].x <= q.back().x) q.pop_back();
		q.push_back(a[i]);
		if (i < k) continue;
		printf("%d ", q.front().x);
	}
	q.clear();
	puts(""); // 最大值
	f(i,1,n) {
		while (!q.empty() && q.front().id <= i - k) q.pop_front();
		while (!q.empty() && a[i].x >= q.back().x) q.pop_back();
		q.push_back(a[i]);
		if (i < k) continue;
		printf("%d ", q.front().x);
	}
	return 0;
}
```

```python
from collections import deque
import sys
output = sys.stdout.write
ans = ""
q = deque()
n, k = map(int, input().split())
a = list(map(int, input().split()))
for i in range(n):
    while q and q[0] <= i - k: q.popleft()
    while q and a[i] <= a[q[-1]]: q.pop()
    q.append(i)
    if i < k - 1: continue
    ans += str(a[q[0]]) + " "
ans += "\n"
q.clear()
for i in range(n):
    while q and q[0] <= i - k: q.popleft()
    while q and a[i] >= a[q[-1]]: q.pop()
    q.append(i)
    if i < k - 1: continue
    ans += str(a[q[0]]) + " "
ans += "\n"
output(ans)
```

------

## 15 带反悔贪心 (priority_queue & heapq)-有import (定义结构体的"<")

```cpp
# include <bitss/stdc++.h>

# define f(i,a,b) for (int i = a; i <= b; i++)
# define _f(i,a,b) for (int i = a; i >= b; i--)
# define ll long long
using namespace std;
const int NR = 2.5e5;
int n;
int a[NR + 1], b[NR + 1];
bool flag[NR + 1];

struct Node {
	int x, id;

	bool operator < (const Node &d) const { // 不要漏写const 
		return x < d.x;
	}
};
priority_queue <Node> q; // 默认从大到小排序
// 从小到大：priority_queue <int, vector <int>, greater <int> > q
int main() {
	scanf("%d", &n);
	f(i,1,n) scanf("%d", a + i);
	f(i,1,n) scanf("%d", b + i);
	ll sum = 0;
	int cnt = 0;
	Node tmp;
	f(i,1,n) {
		sum += a[i];
		if (sum >= b[i]) sum -= b[i], flag[i] = true, q.push((Node) {b[i], i}), cnt++;
		else if (!q.empty() && b[i] < q.top().x) {
			tmp = q.top(); q.pop(); q.push((Node) {b[i], i});
			sum += tmp.x - b[i];
			flag[i] = true; flag[tmp.id] = false;
		}
	}
	printf("%d\n", cnt);
	f(i,1,n) {
		if (flag[i]) printf("%d ", i);
	}
	puts("");
	return 0;
}
```

```python
import heapq
import sys
output = sys.stdout.write
class Node:
    def __init__(self, x, id):
        self.x = x
        self.id = id
    def __lt__(self, other): # 定义小于号！
        return self.x > other.x
ans = ""
q = [] # 和列表相同，只是语法上不同; q[0]是最小值！小根堆！
n = int(input())
flag = [False] * n
a = list(map(int, input().split()))
b = list(map(int, input().split()))
now = 0; cnt = 0
for i in range(n):
    now += a[i]
    if now >= b[i]:
        now -= b[i]
        heapq.heappush(q, Node(b[i], i))
        flag[i] = True
        cnt += 1
    elif q and b[i] < q[0].x:
        tmp = heapq.heappop(q) # 先返回值后pop
        flag[tmp.id] = False
        flag[i] = True
        now += tmp.x - b[i]
        heapq.heappush(q, Node(b[i], i))
ans += str(cnt) + "\n"
ans += " ".join(map(str, [i + 1 for i in range(n) if flag[i]]))
output(ans)
```

------

## 16 字典(map,初始化用{},直接赋值)

```python
month1 =\
    "pop, no, zip, zotz, tzec, xul, yoxkin, mol, chen, yax, zac, ceh, mac, kankin, muan, pax, koyab, cumhu".split(", ")
month2 =\
    "imix, ik, akbal, kan, chicchan, cimi, manik, lamat, muluk, ok, chuen, eb, ben, ix, mem, cib, caban, eznab, canac, ahau".split(", ")
chk1 = {}
for i in range(18): chk1[month1[i]] = i
chk1["uayet"] = 18
n = int(input()); print(n)
for i in range(n):
    x, y1, y2 = input().split()
    day = int(y2) * 365 + chk1[y1] * 20 + int(x.rstrip('.'))
    print("{} {} {}".format(day % 13 + 1, month2[day % 20], day // 260))
```

------

## 17 enumerate(生成(i, a[i]), 第一项为q[i][0], 第二项为q[i][1])

```python
n = int(input())
a = enumerate(list(map(int, input().split())))
a = sorted(a, key = lambda x: x[1])
for i, v in a: print(i + 1, end = " ")
print()
ans = 0
for i, v in enumerate(a): ans += v[1] * (n - 1 - i)
print(f"{ans / n:.2f}")
```

------

## 18 改变递归深度限制(默认1000，有时候要调大，否则RE)

```python
import sys
sys.setrecursionlimit(100000)
```

------

## 19 字典设置初始值（避免查询key是否有定义）

```python
a = defaultdict(list) # 默认a[所有] = []
b = defaultdict(int) # 默认a[所有] = 0
c = defaultdict(bool) # 默认a[所有] = False
```

------

## 20
