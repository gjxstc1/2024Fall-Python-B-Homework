# list指标从0开始，下标为0时再-1是回到最后一个元素！（前缀和时注意）
# reversed(a)只返回迭代器，若返回list应该用a = list(reversed(a))
# 在dfs时如果要用global，要写dfs前面声明这个变量（否则CE）：
```python
n = int(input())
ans = 0 # 先声明要global的变量再写dfs
a = list(map(int, input().split()))
def g(L, R):
    global ans
    ...
    return d
g(0, n - 1)
print(ans)
```

## 18 改变递归深度限制(默认1000，有时候要调大，否则RE)

```python
import sys
sys.setrecursionlimit(1 << 30)
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

## 19 字典设置初始值（避免查询key是否有定义,有import）

```python
from collections import defaultdict
a = defaultdict(list) # 默认a[所有] = []
b = defaultdict(int) # 默认a[所有] = 0
c = defaultdict(bool) # 默认a[所有] = False
dict.setdefault(key, default_value)
一般的字典初值：d = {key1 : value1, key2 : value2, key3 : value3 }
```

------

## 20 lower_bound, upper_bound, bisect二分查找(有import)
```python
bisect.bisect_left(a, target) = lower_bound，返回第一个数组a中>=target的位置,没有就返回len(a)；
bisect.bisect_right(a, target) = upper_bound，返回第一个数组a中>target的位置,没有就返回len(a)
一般也有（left,right）：bisect.bisect_right(a, x, lo=0, hi=len(a))，后两个表示数组范围
```
```python
import bisect
T = int(input())
for ___ in range(T):
    n, k = map(int, input().split())
    a = list(map(int, input().split())); b = list(map(int, input().split()))
    dp = [0] * n; mx = [0] * n # dp[i]必选i
    dp[0] = mx[0] = b[0]
    for i in range(1, n):
        d = bisect.bisect_left(a, a[i] - k) - 1 # bisect.bisect_left(a, target) = lower_bound，返回第一个数组a中>=target的位置；
        dp[i] = mx[d] + b[i] # bisect.bisect_right(a, target) = upper_bound，返回第一个数组a中>target的位置
        mx[i] = max(mx[i - 1], dp[i])
    print(mx[n - 1])
```

------

## 23 自动记忆化搜索，要求只与状态有关（有import）
```python
from functools import lru_cache
@lru_cache(maxsize=None)
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

## 24 辅助栈 & split后得到list
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

------

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
""" "1" 49, "A" 65-90, "a" 97-122"""
```

------
## 25 找二维list的索引
```python
for rowx, row in enumerate(a):
        try:
            coly = row.index('S')
            sx, sy = rowx, coly
            break
        except ValueError: continue
```

------

## 28 set

```python
n, m = map(int, input().split())
a = list(map(int, input().split()))
s = set() # 初始化
num = 1
for x in a:
    s.add(x) # add添加元素，已存在则无事
    if len(s) == m: # O(1)查找len
        num += 1
        s.clear()
print(num)
# 初始化也可以用：a = {1, 2, 3};也可以 a = set([1, 2, 4])
```
```python
判断元素是否在集合中存在： x in s
>>> thisset = set(("Google", "Runoob", "Taobao"))
>>> "Runoob" in thisset
True
>>> "Facebook" in thisset
False
>>> 
删除元素（要求存在）：s.remove(x)
集合的交并补：
>>> a = set('abracadabra')
>>> b = set('alacazam')
>>> a                                  
{'a', 'r', 'b', 'c', 'd'}
>>> a - b # 集合a中包含而集合b中不包含的元素
{'r', 'd', 'b'}
>>> a | b 并集# 集合a或b中包含的所有元素
{'a', 'c', 'r', 'd', 'b', 'm', 'z', 'l'}
>>> a & b 交集# 集合a和b中都包含了的元素
{'a', 'c'}
>>> a ^ b 对称差# 不同时包含于a和b的元素
{'r', 'd', 'b', 'm', 'z', 'l'}
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

## 10 一次性读入和输出（需import）

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

## 27 逆序对数（归并排序）
```python
n = int(input())
ans = 0
a = list(map(int, input().split()))

def g(L, R):
    global ans
    if L == R: return [a[L]]
    mid = (L + R) >> 1
    b = g(L, mid) + [float("inf")]; c = g(mid + 1, R) + [float("inf")]
    l, r = 0, 0
    d = []
    while l <= mid - L or r <= R - mid - 1:
        if b[l] < c[r]: d.append(b[l]); l += 1
        else: d.append(c[r]); r += 1; ans += mid - L - l + 1
    return d

g(0, n - 1)
print(ans)
```

------

## 12 rstrip
```python
a.rstrip()只删掉末位指定字符，没指明默认空格
```
------

## 13 整数转成字符串
```python
a = str(i) # i = 100, 则 a = "100"
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

## 15 带反悔贪心(p_queue&heapq)-有import(定义结构体的"<")

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

-----

## 21 拷贝不等于引用

```python
拷贝： a = b[:] # 两个独立的列表，对一个列表的修改不影响另一个列表时
引用： a = b # 两个变量引用同一个列表，对一个列表的修改同时反映在另一个列表上
```
------

## 22 bfs语法(!!popleft)

```python
from collections import deque
q = deque()
while q:
    tmp = q.popleft()
    for nx, ny in directions:
        tx, ty = tmp[0] + nx, tmp[1] + ny
        if 0 <= tx < n and 0 <= ty < n and not flag[tx][ty]:
            if a[tx][ty] == "1":
                print(tmp[2])
                exit(0)
            flag[tx][ty] = True
            q.append((tx, ty, tmp[2] + 1))
```

------

## 24 dijkstra (N+M)logN
```cpp
struct T {
    int x;
    ll dis;
    bool operator < (const T &b) const {
        return dis > b.dis;
    }
};
priority_queue <T> q;
void dij() {
    memset(d, 1, sizeof(d));
    memset(flag, false, sizeof(flag));
    q.push((T) {s, 0});
    d[s] = 0;
    while (!q.empty()) {
        int x = q.top().x; q.pop();
        if (flag[x]) continue;
        flag[x] = true;
        for (int i = fe[x]; i; i = g[i].nxt) {
            int y = g[i].to;
            if (d[y] <= d[x] + g[i].w) continue;
            d[y] = d[x] + g[i].w;
            q.push((T) {y, d[y]});
        }
    }
}

int main() {
    scanf("%d%d%d", &n, &m, &s);
    f(i,1,m) {
        int u, v, w;
        scanf("%d%d%d", &u, &v, &w);
        addedge(u, v, w);
    }
    dij();
    f(i,1,n) printf("%lld ", d[i]);
    return 0;
}
```

------

## 26 product (有import，生成笛卡尔积$A^B$)

```python
from itertools import product
a = [list(map(int, input().split())) for _ in range(5)]
movement = [[0] * 6 for _ in range(5)]
for tmp in product(range(2), repeat = 6):
    movement[0] = list(tmp)
```

------
### 29 哈希表

求得一个数组最长的和为0的连续子序列的方法（实例为求最长平均值为a的字串）

```python
def longest_subarray_with_avg(arr, a):
    # 步骤 1: 将每个元素减去a
    diff = [x - a for x in arr]
    # 步骤 2: 使用哈希表存储前缀和
    prefix_sum = 0
    prefix_map = {0: -1}  # 前缀和为0时，起始索引为-1
    max_len = 0
    start_index = -1
    # 步骤 3: 遍历数组并计算前缀和
    for i in range(len(diff)):
        prefix_sum += diff[i]
        # 步骤 4: 如果前缀和已经出现过，计算子数组的长度
        if prefix_sum in prefix_map:
            length = i - prefix_map[prefix_sum]
            if length > max_len:
                max_len = length
                start_index = prefix_map[prefix_sum] + 1
        else:
            prefix_map[prefix_sum] = i
    # 返回最长子数组
    if max_len > 0:
        return arr[start_index:start_index + max_len]
    else:
        return []
```

------

## 30 字符串互相转化

```python
upper()：将字符串中的所有小写字母转换为大写。
lower()：将字符串中的所有大写字母转换为小写。
swapcase()：将字符串中的所有大写字母转换为小写，同时将所有小写字母转换为大写。
# 定义一个包含大小写字母的字符串
text = "Hello, World!"
# 使用 upper() 方法将所有字母转为大写
upper_text = text.upper()
print(f"Upper case: {upper_text}")
# 使用 lower() 方法将所有字母转为小写
lower_text = text.lower()
print(f"Lower case: {lower_text}")
# 使用 swapcase() 方法将所有字母的大小写互换
swapped_text = text.swapcase()
print(f"Swapped case: {swapped_text}")
```

------

## 31 defaultdict

* 这是一种"如果访问的key不存在与字典中,就给它新建一个默认值`dict[key]=default_value`的字典

  ```python
  from collections import defaultdict
  a=defaultdict(list)#以空列表为默认值的defaultdict
  #类似地,括号里是int,set也可以
  ```

* 如果要自定义默认值,用`defaultdict(lambda: XXXX)`

### 32 双端队列(deque)

* 可以从两端弹出元素,但弹出中间元素会很慢

  ```python
  from collections import deque
  deque.popleft(item)#左弹出
  deque.pop(item)#右弹出
  deque.appendleft(item)#左加入
  deque.append(item)#右加入
  ```

### 33 小顶堆(heapq)

* 这是用一个列表实现小顶堆的数据结构.小顶堆是一种儿茶素(x)(二叉树),**根节点最小,每一个节点都比它的孩子小.**

* 你可以先建立一个列表,然后对它做堆操作,这样它就被你当成了一个堆

* (python没有大顶堆数据结构,想要做大顶堆可以把heapq里面所有元素加负号)

  ```python
  import heapq
  heap=[6,5,4,3,2,1]#注意现在它还只是一个普通的列表
  ```

* 如果它的元素并不是以堆的形式排列的,就给它"堆化":`heapq.heapify(heap)`

* 堆化后的a应为`[1,2,4,3,5,6]`(以堆的形式排列的方式显然不止一种)

* 然后可以对堆进行堆操作:

  ```python
  import heapq
  heapq.heappush(heap, item)#推入元素
  heapq.heappop(heap)#弹出堆顶
  heapq.heappushpop(heap,item)#先推再弹,更高效
  heapq.heapreplace(heap,item)#先弹再推(SN1)(bushi)
  ```

* 弹出和推入都是O(log n)

### 34 列表（deepcopy复制二维数组！）

* 构建二维列表:`[[],[],...]`(就是列表套列表)

* 清空:`l.clear()`,O(1)

* 切片:`l[a:b]`,**O(b-a)**,不是O(1)!

* 反转:`l.reverse()`,O(n)

* **浅拷贝问题**:当列表等被拷贝时,拷贝的是列表的地址而不是列表本身

  ```python
  a=[1,2,3]
  b=a
  ```

  此时a,b共用一个地址,其中任何一个被改变时,另一个也会相应改变.
  这称为"浅拷贝".(如果你知道FL Studio...这就相当于FL Studio里面复制一个pattern)

  与之相反的叫深拷贝,拷贝的是列表本身(相当于复制了pattern然后make unique)

  ```python
  from copy import deepcopy
  a=[1,2,3]
  b=a[:]
  c=[[1,2],[2,3],[3,4]]
  d=c[:]
  e=deepcopy(c)
  ```

  对于一维列表,用切片复制就可以实现深拷贝, `l.copy()`与之是等价的.
  对于>=2维的列表,如果用切片或者`l.copy()`,那里面那些小列表还是浅拷贝的.如果要完全深拷贝自身,需要用deepcopy函数

* 列表化:`list(x)`,时间复杂度取决于遍历x的长度.其中x可以是以下:

  * 字符串:`list('abc')==['a','b','c']`
  * 迭代器/生成器(range啊map啊这类的):
    * `list(range(4))==[0,1,2,3]`
    * `list(map(int,input().split())==[2,5,1]`(假设输入'2 5 1')
    * `list(enumerate([2,5,1]))==[(0,2),(1,5),(2,1)]`

### 35 集合(无重复的元素,自动去重)

* **集合内的元素只能是"零维的"标量.**
* 添加:`s.add(x)`
* 查询:`x (not) in s`,**O(1)**
* 删除:
  * `s.discard(x)`,**O(1)**:删除x,如果没找到x就无事发生
* 弹出:`s.pop()`,O(1):随弹出一项
* 清空:`s.clear()`,O(1)
* 集合运算:
  * 并:`a|b`,O(len(a)+len(b))
  * 交:`a&b`,O(len(a)+len(b))
  * 差:
    * `a-b`,O(len(a)+len(b)),注意这是表示∈a且∉b的元素的集合,
      如`{1}-{1,2}==set(),{1,2}-{1}=={2}`
    * `a^b`,对称差,等于a|b-a&b
  * `a>=b,a>b,a<=b,a<b`,O(len(小的那个))表示集合的包含关系,如果既不是子集也不是超集也返回False
* 构建集合:
  * 空集:`a=set()`
  * 集合化:set(x),x是列表\元组\字典(a是字典则会返回所有键组成的集合)

### 36 字典

* **字典的键只能是"零维"标量(int,float这种),而非"一维"以上的复合数据类型.**
* 删除:`del d[key]`
* 弹出:
  * `d.pop(key)`,O(1)
* 获取:`d.get(key,dft)`可以在key不在字典时返回默认值dft
* 清空:`d.clear()`,O(1)
* 键or值们:`d.keys()`,`d.values()`,O(1).

### 37 迭代器和生成器

* range就是一种迭代器,它在迭代时,每次生成一个量.
* **可以被列表/元组/字典/集合化.**
* python中常用的迭代器/生成器有以下几种:
  * `range(n)`:生成0,1,...,(n-1)的整数序列
  * `enumerate(list)`:依次生成列表/元组中每一项索引与元素的对应:
    * `list(enumerate(['a','b','c']))=[(0,'a'),(1,'b'),(2,'c')]`
    * `dict(enumerate(['a','b','c']))={0:'a',1:'b',2:'c'}`
  * `zip(a,b)`:依次生成列表/元组a,b中每一项的对应.如果有一方有多余项,则不管
    * `list(zip([1,2,3],[11,22]))=[(1,11),(2,22)]`
    * `dict`类似

    ### 38 next_permutation:

1）从后往前找第一组相邻的升序数对，记录左边的位置p。 2）从后往前找第一个比p位置的数大的数，将两个数交换。 3）把p位置后所有数字逆序。

```python
def next_permutation(b):
    pos1 = n - 2
    while pos1 >= 0 and b[pos1] >= b[pos1 + 1]: pos1 -= 1
    if pos1 == -1: return [i + 1 for i in range(n)]
    pos2 = n - 1
    while pos2 >= 0 and b[pos2] <= b[pos1]: pos2 -= 1
    b[pos2], b[pos1] = b[pos1], b[pos2]
    b[pos1 + 1:] = sorted(b[pos1 + 1:])
    return b
T = int(input())
for ___ in range(T):
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    for i in range(k): a = next_permutation(a)
    print(*a)
```

```cpp
int T, n, k;
int a[NR + 1];
int main() {
	scanf("%d", &T);
	while (T --) {
		scanf("%d%d", &n, &k);
		f(i,1,n) scanf("%d", a + i);
		f(i,1,k) next_permutation(a + 1, a + n + 1); // n,n-1,...,1下一个是1,2,...n
		f(i,1,n) printf("%d ", a[i]);
		puts("");
	}
	return 0;
}

```

------

## 39 取整(有import)

```python
import math
print(math.ceil(4.2)) # 5
print(math.floor(4.2)) # 4
```

------

## 40 二维数组拷贝(有import)

```python
import copy
original = [1, 2, [3, 4]]
copied = copy.deepcopy(original)
print(copied) # [1, 2, [3, 4]]
```

------

## 41 Dijkstra - 走山路

```python
from collections import deque
import heapq
def solve(sx, sy, ex, ey):
    q = []
    if a[sx][sy] == '#' or a[ex][ey] == '#':
        print("NO")
        return
    flag = [[False] * m for _ in range(n)] # 是否 in q
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
